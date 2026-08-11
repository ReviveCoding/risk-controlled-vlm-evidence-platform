from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from model_forge import __version__
from model_forge.artifacts import (
    load_model_bundle,
    predict_feature_csv,
    verify_inference_replay,
)
from model_forge.contracts.models import (
    DataContract,
    ReferenceRelease,
    ReferenceState,
    ReleaseEvidence,
)
from model_forge.contracts.sec_fsds import SecFsdsContract
from model_forge.data.acquisition import load_cfpb_export, publish_snapshot
from model_forge.data.fixture import generate_complaint_fixture
from model_forge.data.out_of_core import publish_cfpb_parquet_snapshot
from model_forge.data.sec_fsds import publish_sec_fsds_snapshot
from model_forge.data.sec_label_audit import publish_sec_label_feasibility_audit
from model_forge.data.sec_linkage_diagnostic import publish_sec_linkage_diagnostic
from model_forge.migration.parity import assess_manifest, validate_parity
from model_forge.monitoring.drift import monitoring_snapshot
from model_forge.pipeline import (
    compare_run_directories,
    finalize_replay_qualification,
    regenerate_report,
    run_demo,
    verify_output,
    verify_replay,
)
from model_forge.qualification.faults import default_reference_state, run_fault_benchmark
from model_forge.qualification.policies import PolicyEngine
from model_forge.registry.local import LocalRegistry
from model_forge.resources import bundled_config, bundled_config_names, resolve_config
from model_forge.utils import (
    governed_run_ancestor,
    load_bounded_json,
    load_numeric_npy,
    load_yaml,
    reject_symlink_components,
    write_json,
)
from model_forge.workloads.qwen_embeddings import encode_qwen_csv, qwen_gpu_probe


def _json_print(value: object, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str), file=stream)


def _reject_output_collision(output: Path | None, inputs: list[Path | None]) -> None:
    if output is None:
        return
    reject_symlink_components(output, "output")
    resolved_output = output.resolve(strict=False)
    if any(value is not None and value.resolve(strict=False) == resolved_output for value in inputs):
        raise ValueError("output must not overwrite an input file")
    governed_root = governed_run_ancestor(output)
    if governed_root is not None:
        raise ValueError(f"output must be outside the governed run ({governed_root})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-forge", description="ModelForge-R governed experimentation CLI"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="preserve a traceback for developer diagnosis after an execution error",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize versioned contracts in a target directory")
    init.add_argument("destination", type=Path)

    ingest = sub.add_parser("ingest", help="validate and publish a fixture, CFPB, or SEC FSDS snapshot")
    ingest.add_argument("--source", choices=["fixture", "cfpb-csv", "sec-fsds"], required=True)
    ingest.add_argument("--input", type=Path)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--data-contract", type=Path, default=Path("configs/data_contract.yaml"))
    ingest.add_argument("--sec-contract", type=Path, default=Path("configs/sec_fsds_contract.yaml"))
    ingest.add_argument("--rows", type=int, default=3200)
    ingest.add_argument("--seed", type=int, default=20260717)
    ingest.add_argument(
        "--cutoff",
        help=(
            "inclusive maturity-safe event-time boundary; subtract the data contract's "
            "label_maturity_days from the snapshot as-of date before supplying this value"
        ),
    )
    ingest.add_argument(
        "--storage-format",
        choices=["csv", "parquet-dataset"],
        default="csv",
        help="csv keeps legacy in-memory behavior; parquet-dataset scans out of core",
    )
    ingest.add_argument("--chunk-rows", type=int, default=100_000)
    ingest.add_argument("--train-sample-rows", type=int, default=100_000)
    ingest.add_argument("--evaluation-sample-rows", type=int, default=25_000)

    sec_audit = sub.add_parser(
        "sec-label-audit",
        help="audit maturity-safe SEC future-amendment labels without training",
    )
    sec_audit.add_argument("--historical-input", type=Path, required=True)
    sec_audit.add_argument("--recent-input", type=Path, required=True)
    sec_audit.add_argument("--output", type=Path, required=True)
    sec_audit.add_argument(
        "--historical-contract",
        type=Path,
        default=Path("configs/sec_fsds_history_contract.yaml"),
    )
    sec_audit.add_argument(
        "--recent-contract",
        type=Path,
        default=Path("configs/sec_fsds_contract.yaml"),
    )

    sec_linkage = sub.add_parser(
        "sec-linkage-diagnostic",
        help="diagnose SEC amendment linkage and split-level label readiness",
    )
    sec_linkage.add_argument("--historical-input", type=Path, required=True)
    sec_linkage.add_argument("--recent-input", type=Path, required=True)
    sec_linkage.add_argument("--output", type=Path, required=True)
    sec_linkage.add_argument(
        "--historical-contract",
        type=Path,
        default=Path("configs/sec_fsds_history_contract.yaml"),
    )
    sec_linkage.add_argument(
        "--recent-contract",
        type=Path,
        default=Path("configs/sec_fsds_contract.yaml"),
    )

    demo = sub.add_parser("demo", help="run complete offline fixture workflow")
    demo.add_argument("--config", type=Path, default=Path("configs/experiment_contract.yaml"))
    demo.add_argument("--output", type=Path)
    demo.add_argument("--clean", action="store_true")

    train = sub.add_parser("train", help="run the atomic governed train-through-evidence workflow")
    train.add_argument("--config", type=Path, default=Path("configs/experiment_contract.yaml"))
    train.add_argument("--output", type=Path)
    train.add_argument("--clean", action="store_true")
    train.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="canonical snapshot created by `model-forge ingest`",
    )
    train.add_argument(
        "--dataset-manifest",
        type=Path,
        help="snapshot manifest; defaults to <dataset>.manifest.json",
    )

    evaluate = sub.add_parser("evaluate", help="regenerate and verify evidence from a completed run")
    evaluate.add_argument("--output", type=Path, default=Path("outputs/latest"))

    report = sub.add_parser("report", help="regenerate canonical evidence report")
    report.add_argument("--output", type=Path, default=Path("outputs/latest"))

    verify = sub.add_parser("verify", help="verify required outputs and checksums")
    verify.add_argument("--output", type=Path, default=Path("outputs/latest"))

    replay_verify = sub.add_parser(
        "replay-verify",
        help="compare two independently executed complete runs under the replay contract",
    )
    replay_verify.add_argument("--first", type=Path, required=True)
    replay_verify.add_argument("--second", type=Path, required=True)
    replay_verify.add_argument("--output", type=Path)

    replay_compare = sub.add_parser(
        "replay-compare",
        help="test the run-directory comparator without claiming independent execution",
    )
    replay_compare.add_argument("--first", type=Path, required=True)
    replay_compare.add_argument("--second", type=Path, required=True)
    replay_compare.add_argument("--output", type=Path)

    artifact_verify = sub.add_parser(
        "artifact-verify", help="checksum and compatibility check, then explicit-trust reload"
    )
    artifact_verify.add_argument("--artifact", type=Path, required=True)
    artifact_verify.add_argument("--manifest", type=Path, required=True)
    artifact_verify.add_argument("--trust-local-joblib", action="store_true")

    predict = sub.add_parser("predict", help="run feature-table inference from a verified local artifact")
    predict.add_argument("--artifact", type=Path, required=True)
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--trust-local-joblib", action="store_true")

    inference_replay = sub.add_parser(
        "inference-replay-verify",
        help="reload a model and compare identity-aligned probabilities with the governed run",
    )
    inference_replay.add_argument("--artifact", type=Path, required=True)
    inference_replay.add_argument("--manifest", type=Path, required=True)
    inference_replay.add_argument("--fixture", type=Path, required=True)
    inference_replay.add_argument("--original-predictions", type=Path, required=True)
    inference_replay.add_argument("--candidate-id", default="W2")
    inference_replay.add_argument("--output", type=Path, required=True)
    inference_replay.add_argument("--trust-local-joblib", action="store_true")

    replay_qualify = sub.add_parser(
        "replay-qualify",
        help="create a final C4 decision from primary evidence and independent replay evidence",
    )
    replay_qualify.add_argument("--evidence", type=Path, required=True)
    replay_qualify.add_argument("--comparison", type=Path, required=True)
    replay_qualify.add_argument("--policy-config", type=Path, default=Path("configs/release_policy.yaml"))
    replay_qualify.add_argument("--output", type=Path, required=True)

    fault = sub.add_parser("fault-benchmark", help="run the independent 160-case fault corpus")
    fault.add_argument("--policy-config", type=Path, default=Path("configs/release_policy.yaml"))
    fault.add_argument("--fault-config", type=Path, default=Path("configs/fault_matrix.yaml"))
    fault.add_argument("--output", type=Path, default=Path("outputs/fault_benchmark"))
    fault.add_argument("--bootstrap", type=int, default=2000)

    qualify = sub.add_parser("qualify", help="qualify a release-evidence JSON record")
    qualify.add_argument("evidence", type=Path)
    qualify.add_argument("--policy", choices=["C0", "C1", "C2", "C3", "C4"], default="C4")
    qualify.add_argument("--policy-config", type=Path, default=Path("configs/release_policy.yaml"))
    qualify.add_argument("--reference", type=Path)
    qualify.add_argument("--output", type=Path)

    promote = sub.add_parser("promote", help="approval-gated local alias update")
    promote.add_argument("--registry", type=Path, required=True)
    promote.add_argument("--alias", choices=["Challenger", "Champion"], required=True)
    promote.add_argument("--version", required=True)
    promote.add_argument("--approve", action="store_true")
    promote.add_argument("--expected-current")

    sub.add_parser("gpu-probe", help="report READY or honest SKIP for the optional Qwen/CUDA path")

    embed = sub.add_parser("embed-qwen", help="create revision-tracked Qwen embeddings on local CUDA")
    embed.add_argument("--input", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--text-column", default="complaint_narrative")
    embed.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    embed.add_argument("--revision", required=True)
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--max-length", type=int, default=512)

    monitor = sub.add_parser("monitor", help="compute a score-drift snapshot")
    monitor.add_argument("reference", type=Path, help=".npy file with reference scores")
    monitor.add_argument("current", type=Path, help=".npy file with current scores")
    monitor.add_argument("--threshold", type=float, default=0.20)
    monitor.add_argument("--output", type=Path)

    migrate = sub.add_parser("migrate", help="assess a legacy manifest or validate prediction parity")
    migrate.add_argument("action", choices=["assess", "validate-parity"])
    migrate.add_argument("--manifest", type=Path)
    migrate.add_argument("--legacy", type=Path)
    migrate.add_argument("--migrated", type=Path)
    migrate.add_argument("--tolerance", type=float, default=1e-8)
    migrate.add_argument("--output", type=Path)
    return parser


def _load_reference(path: Path | None) -> ReferenceState:
    if path is None:
        return default_reference_state()
    raw = load_bounded_json(path, label="reference-state document")
    releases = [ReferenceRelease(**item) for item in raw["releases"]]
    return ReferenceState(raw["state_id"], raw["as_of"], releases, int(raw.get("window", 8)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: dict[str, Any]
    try:
        if args.command == "init":
            reject_symlink_components(args.destination, "initialization destination")
            destination = args.destination.resolve()
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError("destination exists and is not empty")
            destination.mkdir(parents=True, exist_ok=True)
            config_dir = destination / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            for name in bundled_config_names():
                (config_dir / name).write_text(
                    bundled_config(name).read_text(encoding="utf-8"), encoding="utf-8"
                )
            result = {
                "status": "PASS",
                "destination": str(destination),
                "created": "versioned_contracts",
            }
        elif args.command == "sec-label-audit":
            _reject_output_collision(args.output, [args.historical_input, args.recent_input])
            result = publish_sec_label_feasibility_audit(
                args.historical_input,
                args.recent_input,
                args.output,
                SecFsdsContract.from_yaml(args.historical_contract),
                SecFsdsContract.from_yaml(args.recent_contract),
            )
        elif args.command == "sec-linkage-diagnostic":
            _reject_output_collision(args.output, [args.historical_input, args.recent_input])
            result = publish_sec_linkage_diagnostic(
                args.historical_input,
                args.recent_input,
                args.output,
                SecFsdsContract.from_yaml(args.historical_contract),
                SecFsdsContract.from_yaml(args.recent_contract),
            )
        elif args.command == "ingest":
            if args.source == "sec-fsds":
                if not args.input:
                    raise ValueError("--input is required for sec-fsds")
                sec_contract = SecFsdsContract.from_yaml(resolve_config(args.sec_contract))
                result = publish_sec_fsds_snapshot(
                    args.input, args.output, sec_contract, chunk_rows=args.chunk_rows
                )
                _json_print(result)
                return 0
            data_contract = DataContract.from_yaml(resolve_config(args.data_contract))
            if args.storage_format == "parquet-dataset" and args.source != "cfpb-csv":
                raise ValueError("parquet-dataset storage is supported only for cfpb-csv")
            if args.source == "fixture":
                frame = generate_complaint_fixture(args.rows, args.seed)
                source = args.output.with_name(args.output.name + ".fixture-source.csv")
                source.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(source, index=False, lineterminator="\n")
                evidence_class = "synthetic_fixture"
            else:
                if not args.input:
                    raise ValueError("--input is required for cfpb-csv")
                if not args.cutoff:
                    raise ValueError("--cutoff is required for cfpb-csv label maturity")
                source = args.input
                if args.storage_format == "parquet-dataset":
                    result = publish_cfpb_parquet_snapshot(
                        source,
                        args.output,
                        data_contract,
                        cutoff=args.cutoff,
                        chunk_rows=args.chunk_rows,
                        train_sample_rows=args.train_sample_rows,
                        evaluation_sample_rows=args.evaluation_sample_rows,
                        seed=args.seed,
                    )
                else:
                    frame = load_cfpb_export(source, data_contract, args.cutoff)
                    evidence_class = "public_cfpb_snapshot"
            if args.source == "fixture" or args.storage_format == "csv":
                result = publish_snapshot(
                    frame,
                    args.output,
                    source,
                    data_contract,
                    evidence_class=evidence_class,
                )
        elif args.command == "demo":
            result = run_demo(args.config, args.output, args.clean)
        elif args.command == "train":
            if args.dataset_manifest is not None and args.dataset is None:
                raise ValueError("--dataset-manifest requires --dataset")
            result = run_demo(
                args.config,
                args.output,
                args.clean,
                dataset_path=args.dataset,
                dataset_manifest=args.dataset_manifest,
            )
        elif args.command == "evaluate" or args.command == "report":
            result = regenerate_report(args.output)
        elif args.command == "verify":
            result = verify_output(args.output.resolve())
        elif args.command == "replay-verify":
            result = verify_replay(args.first, args.second, args.output)
        elif args.command == "replay-compare":
            result = compare_run_directories(args.first, args.second, args.output)
        elif args.command == "artifact-verify":
            _, manifest = load_model_bundle(
                args.artifact.resolve(),
                args.manifest.resolve(),
                trusted_local_artifact=args.trust_local_joblib,
            )
            result = {
                "status": "PASS",
                "candidate_id": manifest["candidate_id"],
                "artifact_schema_version": manifest["artifact_schema_version"],
                "checksum_verified_before_load": True,
            }
        elif args.command == "predict":
            result = predict_feature_csv(
                args.artifact.resolve(),
                args.manifest.resolve(),
                args.input.resolve(),
                args.output.resolve(),
                trusted_local_artifact=args.trust_local_joblib,
            )
        elif args.command == "inference-replay-verify":
            _reject_output_collision(
                args.output,
                [
                    args.artifact,
                    args.manifest,
                    args.fixture,
                    args.original_predictions,
                ],
            )
            result = verify_inference_replay(
                args.artifact.resolve(),
                args.manifest.resolve(),
                args.fixture.resolve(),
                args.original_predictions.resolve(),
                candidate_id=args.candidate_id,
                trusted_local_artifact=args.trust_local_joblib,
            )
            write_json(args.output, result)
        elif args.command == "replay-qualify":
            _reject_output_collision(
                args.output,
                [args.evidence, args.comparison, args.policy_config],
            )
            result = finalize_replay_qualification(
                args.evidence.resolve(),
                args.comparison.resolve(),
                resolve_config(args.policy_config),
                args.output.resolve(),
            )
        elif args.command == "fault-benchmark":
            reject_symlink_components(args.output, "fault benchmark output")
            governed_root = governed_run_ancestor(args.output)
            if governed_root is not None:
                raise ValueError(f"fault benchmark output must be outside the governed run ({governed_root})")
            args.output.mkdir(parents=True, exist_ok=True)
            engine = PolicyEngine(load_yaml(resolve_config(args.policy_config)))
            frame, summary = run_fault_benchmark(
                engine,
                resamples=args.bootstrap,
                fault_config=load_yaml(resolve_config(args.fault_config)),
            )
            frame.to_csv(args.output / "fault_benchmark_results.csv", index=False, lineterminator="\n")
            write_json(args.output / "fault_benchmark_summary.json", summary)
            result = summary
        elif args.command == "qualify":
            _reject_output_collision(args.output, [args.evidence, args.reference, args.policy_config])
            raw = load_bounded_json(args.evidence, label="release-evidence document")
            evidence = ReleaseEvidence.from_dict(raw)
            decision = PolicyEngine(load_yaml(resolve_config(args.policy_config))).evaluate(
                args.policy, evidence, _load_reference(args.reference)
            )
            result = asdict(decision)
            if args.output:
                write_json(args.output, result)
        elif args.command == "promote":
            registry = LocalRegistry(args.registry)
            result = registry.set_alias(
                args.alias,
                args.version,
                approved=args.approve,
                expected_current=args.expected_current,
            )
        elif args.command == "gpu-probe":
            result = qwen_gpu_probe()
        elif args.command == "embed-qwen":
            result = encode_qwen_csv(
                args.input,
                args.output,
                text_column=args.text_column,
                model_name=args.model,
                revision=args.revision,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        elif args.command == "monitor":
            _reject_output_collision(args.output, [args.reference, args.current])
            result = monitoring_snapshot(
                load_numeric_npy(args.reference),
                load_numeric_npy(args.current),
                args.threshold,
            )
            if args.output:
                write_json(args.output, result)
        elif args.command == "migrate":
            if args.action == "assess":
                if not args.manifest:
                    raise ValueError("--manifest is required for assess")
                _reject_output_collision(args.output, [args.manifest])
                result = assess_manifest(load_bounded_json(args.manifest, label="migration manifest"))
            else:
                if not args.legacy or not args.migrated:
                    raise ValueError("--legacy and --migrated are required for validate-parity")
                _reject_output_collision(args.output, [args.legacy, args.migrated])
                result = validate_parity(
                    load_numeric_npy(args.legacy),
                    load_numeric_npy(args.migrated),
                    args.tolerance,
                )
            if args.output:
                write_json(args.output, result)
        else:
            raise AssertionError("unreachable")
        _json_print(result)
        return 0 if not isinstance(result, dict) or result.get("status") != "FAIL" else 1
    except Exception as error:
        if args.debug:
            raise
        _json_print(
            {"status": "ERROR", "error_type": type(error).__name__, "message": str(error)},
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
