from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import fitz

from .documents import Candidate
from .vlm import VLMContractError, VLMVerification

_JSON_FENCE = re.compile(r"^```json\s*(\{.*\})\s*```$", re.DOTALL | re.IGNORECASE)


def parse_single_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    match = _JSON_FENCE.fullmatch(stripped)
    if match:
        stripped = match.group(1)
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise VLMContractError("visual verifier did not return valid JSON") from exc
    if stripped[end:].strip():
        raise VLMContractError("visual verifier returned trailing prose or multiple values")
    if not isinstance(payload, dict):
        raise VLMContractError("visual verifier root must be one JSON object")
    return payload


def _render_pdf_page(source_path: str, page_number: int):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("install the project with the [vlm] extra") from exc
    path = Path(source_path)
    if path.suffix.casefold() != ".pdf":
        raise VLMContractError("Qwen3-VL runtime currently requires a PDF candidate source")
    with fitz.open(path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise VLMContractError("candidate page is outside the PDF")
        pixmap = document.load_page(page_number - 1).get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


class Qwen3VLVerifier:
    """Lazy optional runtime. Real inference requires the `vlm` extra and downloaded model weights."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", *, max_new_tokens: int = 384):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("install the project with the [vlm] extra") from exc
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.max_new_tokens = max_new_tokens

    def verify(self, slot: str, candidate: Candidate) -> VLMVerification:
        image = _render_pdf_page(candidate.source_path, candidate.page)
        prompt = (
            "Return exactly one JSON object and no prose. Verify only the supplied candidate. "
            "Required keys: candidate_id, document_id, page, text, bbox, polarity, confidence. "
            f"Slot: {slot}. Candidate ID: {candidate.candidate_id}. "
            f"Document ID: {candidate.document_id}. Page: {candidate.page}. "
            f"BBox: {candidate.bbox!r}. Candidate text: {candidate.text!r}."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return VLMVerification.model_validate(parse_single_json_object(text))
