from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .documents import Candidate, ParsedBlock, load_manifest, parse_document, retrieve_candidates
from .publication import sha256_file
from .schemas import AssessmentCase, Evidence, Polarity


class VLMContractError(RuntimeError):
    pass


class VLMVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    document_id: str
    page: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=50_000)
    bbox: tuple[float, float, float, float] | None = None
    polarity: Polarity
    confidence: float = Field(ge=0.0, le=1.0)


class VisualVerifier(Protocol):
    def verify(self, slot: str, candidate: Candidate) -> VLMVerification | dict: ...


def _verify_candidate_source(candidate: Candidate, *, phase: str) -> None:
    try:
        current_hash = sha256_file(Path(candidate.source_path))
    except OSError as exc:
        raise VLMContractError(f"candidate source is unavailable {phase} visual verification") from exc
    if current_hash != candidate.source_hash:
        raise VLMContractError(f"candidate source changed {phase} visual verification")


def build_case_from_documents(
    *,
    case_id: str,
    control_id: str,
    required_slots: tuple[str, ...],
    slot_queries: dict[str, str],
    target_system: str,
    target_region: str,
    manifest_path,
    verifier: VisualVerifier,
    confidence_threshold: float = 0.8,
    max_pack_blocks: int = 50_000,
    max_pack_text_chars: int = 20_000_000,
    top_k_per_slot: int = 5,
) -> AssessmentCase:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    if min(max_pack_blocks, max_pack_text_chars, top_k_per_slot) < 1:
        raise ValueError("VLM resource limits must be positive")
    if set(slot_queries) != set(required_slots):
        raise ValueError("slot_queries must exactly match required_slots")
    documents = load_manifest(manifest_path)
    blocks: list[ParsedBlock] = []
    text_chars = 0
    for document in documents:
        parsed = parse_document(document)
        if len(blocks) + len(parsed) > max_pack_blocks:
            raise VLMContractError("evidence pack block limit exceeded")
        text_chars += sum(len(block.text) for block in parsed)
        if text_chars > max_pack_text_chars:
            raise VLMContractError("evidence pack text limit exceeded")
        blocks.extend(parsed)
    evidence: list[Evidence] = []
    document_map = {document.document_id: document for document in documents}
    for slot in required_slots:
        candidates = retrieve_candidates(blocks, documents, slot_queries[slot], top_k=top_k_per_slot)
        if not candidates:
            continue
        accepted = None
        for candidate in candidates:
            _verify_candidate_source(candidate, phase="before")
            try:
                raw = verifier.verify(slot, candidate)
                verified = raw if isinstance(raw, VLMVerification) else VLMVerification.model_validate(raw)
            except Exception as exc:
                raise VLMContractError("visual verifier output is invalid") from exc
            _verify_candidate_source(candidate, phase="during")
            if (
                verified.candidate_id != candidate.candidate_id
                or verified.document_id != candidate.document_id
                or verified.page != candidate.page
                or verified.text != candidate.text
                or verified.bbox != candidate.bbox
            ):
                raise VLMContractError("visual verifier changed candidate identity or grounding")
            if verified.confidence >= confidence_threshold:
                accepted = (candidate, verified)
                break
        if accepted is None:
            continue
        candidate, verified = accepted
        document = document_map[candidate.document_id]
        evidence.append(
            Evidence(
                evidence_id=f"{case_id}:{slot}:{candidate.candidate_id}",
                slot=slot,
                text=verified.text,
                polarity=verified.polarity,
                system=document.system,
                region=document.region,
                age_days=document.age_days,
                page=verified.page,
                bbox=verified.bbox,
                source_hash=candidate.source_hash,
            )
        )
    return AssessmentCase(
        case_id=case_id,
        control_id=control_id,
        required_slots=required_slots,
        target_system=target_system,
        target_region=target_region,
        evidence=evidence,
    )
