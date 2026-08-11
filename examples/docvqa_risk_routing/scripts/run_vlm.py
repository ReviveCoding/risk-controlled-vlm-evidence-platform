from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from common import (
    contract_sha256,
    discover_shards,
    iter_rows,
    load_json,
    normalize_answers,
    normalize_types,
    open_image,
    sha256_file,
)
from huggingface_hub import model_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--question-manifest", type=Path)
    return parser.parse_args()


def load_question_manifest(path: Path | None) -> tuple[set[str] | None, dict | None]:
    if path is None:
        return None, None
    manifest = load_json(path)
    questions = manifest.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Question manifest must contain a non-empty 'questions' list")
    qids = [str(item["question_id"]) for item in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("Question manifest contains duplicate question IDs")
    return set(qids), manifest


def latest_compatible(
    runs_root: Path, mode: str, contract_hash: str, manifest_hash: str | None
) -> Path | None:
    candidates = []
    for state_path in runs_root.glob("*/run_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("mode") == mode
                and state.get("contract_sha256") == contract_hash
                and state.get("question_manifest_sha256") == manifest_hash
                and state.get("status") != "complete"
            ):
                candidates.append(state_path.parent)
        except Exception:
            continue
    return max(candidates, key=lambda path: path.name) if candidates else None


def write_state(path: Path, state: dict):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by the frozen contract")
    contract = load_json(args.contract)
    contract_hash = contract_sha256(contract)
    selected_qids, question_manifest = load_question_manifest(args.question_manifest)
    manifest_hash = sha256_file(args.question_manifest) if args.question_manifest else None
    if selected_qids is not None and args.limit:
        raise ValueError("Do not combine --question-manifest with --limit")
    if args.mode == "pilot" and selected_qids is None:
        raise ValueError("Pilot mode requires --question-manifest")
    shards = discover_shards(args.data_root)
    if len(shards) != contract["dataset"]["expected_shards"]:
        raise RuntimeError("Source admission failed: expected six validation shards. Run Audit first.")
    if args.run_id:
        run_dir = args.runs_root / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = latest_compatible(args.runs_root, args.mode, contract_hash, manifest_hash)
        if run_dir is None:
            suffix = "prompt-candidate" if args.mode == "pilot" else "qwen3vl4b"
            run_id = datetime.now(UTC).strftime(f"%Y%m%dT%H%M%SZ-{args.mode}-{suffix}")
            run_dir = args.runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.jsonl"
    state_path = run_dir / "run_state.json"
    previous_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if previous_state:
        if previous_state.get("contract_sha256") != contract_hash or previous_state.get("mode") != args.mode:
            raise RuntimeError("Existing run ID is incompatible with this contract or mode")
        if previous_state.get("question_manifest_sha256") != manifest_hash:
            raise RuntimeError("Existing run ID uses a different question manifest")
    completed: set[str] = set()
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("contract_sha256") != contract_hash:
                        raise RuntimeError("Existing predictions contain a different contract hash")
                    qid = str(record["question_id"])
                    if qid in completed:
                        raise RuntimeError(f"Existing predictions contain duplicate question ID: {qid}")
                    if selected_qids is not None and qid not in selected_qids:
                        raise RuntimeError(
                            "Existing predictions contain a question outside the pilot manifest"
                        )
                    completed.add(qid)

    seed = 20260804
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_cfg = contract["model"]
    image_cfg = contract["image"]
    gen_cfg = contract["generation"]
    if question_manifest and question_manifest.get("model_id") != model_cfg["model_id"]:
        raise ValueError("Question manifest model ID does not match the candidate contract")
    frozen_manifest_revision = question_manifest.get("model_revision") if question_manifest else None
    revision = (
        previous_state.get("model_revision")
        or frozen_manifest_revision
        or model_info(model_cfg["model_id"]).sha
    )
    target_questions = (
        len(selected_qids)
        if selected_qids is not None
        else (args.limit or contract["dataset"]["expected_questions"])
    )
    state = {
        "status": "loading_model",
        "mode": args.mode,
        "run_dir": str(run_dir),
        "contract_sha256": contract_hash,
        "model_id": model_cfg["model_id"],
        "model_revision": revision,
        "completed": len(completed),
        "limit": target_questions,
        "question_manifest": str(args.question_manifest) if args.question_manifest else None,
        "question_manifest_sha256": manifest_hash,
        "contract_version": contract["contract_version"],
        "started_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
    }
    write_state(state_path, state)
    processor = AutoProcessor.from_pretrained(
        model_cfg["model_id"],
        revision=revision,
        min_pixels=image_cfg["min_pixels"],
        max_pixels=image_cfg["max_pixels"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_cfg["model_id"],
        revision=revision,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=model_cfg["attention"],
    ).eval()
    loaded_revision = getattr(model.config, "_commit_hash", None)
    if loaded_revision and loaded_revision != revision:
        raise RuntimeError(f"Loaded model revision drift: {loaded_revision} != {revision}")
    state["status"] = "running"
    write_state(state_path, state)
    started = time.time()
    seen = 0
    with predictions_path.open("a", encoding="utf-8", buffering=1) as output:
        for shard, row_index, _, row in iter_rows(shards, batch_size=8):
            qid = str(row["question_id"])
            if selected_qids is not None and qid not in selected_qids:
                continue
            if args.limit and seen >= args.limit:
                break
            seen += 1
            if qid in completed:
                continue
            question = str(row["question"])
            image = open_image(row["image"], args.data_root)
            prompt = gen_cfg["prompt"].format(question=question)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            tick = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            input_length = inputs["input_ids"].shape[-1]
            tokens = generated.sequences[0, input_length:]
            prediction = processor.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            token_logps = []
            for token, logits in zip(tokens.tolist(), generated.scores, strict=False):
                token_logps.append(float(torch.log_softmax(logits[0].float(), dim=-1)[token].item()))
            confidence = math.exp(sum(token_logps) / len(token_logps)) if token_logps else 0.0
            record = {
                "question_id": qid,
                "ucsf_document_id": str(row["document_id"]),
                "page_id": str(row["page_id"] if row["page_id"] is not None else ""),
                "question": question,
                "answers": normalize_answers(row["answers"]),
                "question_types": normalize_types(row["question_types"]),
                "prediction": prediction,
                "confidence_raw": confidence,
                "token_count": len(tokens),
                "latency_seconds": time.time() - tick,
                "source_shard": shard.name,
                "source_row": row_index,
                "model_id": model_cfg["model_id"],
                "model_revision": revision,
                "contract_sha256": contract_hash,
                "contract_version": contract["contract_version"],
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed.add(qid)
            state["completed"] = len(completed)
            state["last_question_id"] = qid
            state["elapsed_seconds"] = time.time() - started
            state["updated_utc"] = datetime.now(UTC).isoformat()
            write_state(state_path, state)
            print(
                f"DOCVQA_PROGRESS={len(completed)}/{state['limit']} qid={qid} "
                f"latency={record['latency_seconds']:.2f}s",
                flush=True,
            )
            del inputs, generated, tokens, image
            if len(completed) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    if selected_qids is not None and seen != len(selected_qids):
        missing = len(selected_qids) - seen
        raise RuntimeError(f"Question manifest admission failed: {missing} question IDs were not found")
    state["status"] = "complete" if len(completed) >= state["limit"] else "partial"
    state["completed_utc"] = datetime.now(UTC).isoformat()
    state["elapsed_seconds"] = time.time() - started
    write_state(state_path, state)
    print(f"DOCVQA_RUN_STATUS={state['status'].upper()}")
    print(f"DOCVQA_RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
