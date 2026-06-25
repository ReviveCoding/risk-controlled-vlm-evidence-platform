from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from control_evidence.decision import assess
from control_evidence.documents import DocumentError, load_manifest, parse_document
from control_evidence.publication import sha256_file
from control_evidence.schemas import Polarity, Status
from control_evidence.vlm import VLMContractError, VLMVerification, build_case_from_documents


def _manifest(tmp_path: Path, source: Path, document_id: str = "doc-1") -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": document_id,
                        "source_path": source.name,
                        "sha256": sha256_file(source),
                        "document_type": source.suffix.lstrip("."),
                        "system": "payments-prod",
                        "region": "us-east-1",
                        "age_days": 10,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


class SupportingVerifier:
    def verify(self, slot, candidate):
        return VLMVerification(
            candidate_id=candidate.candidate_id,
            document_id=candidate.document_id,
            page=candidate.page,
            text=candidate.text,
            bbox=candidate.bbox,
            polarity=Polarity.SUPPORTS,
            confidence=0.95,
        )


def test_secure_manifest_json_parse_and_vlm_decision(tmp_path):
    source = tmp_path / "evidence.json"
    source.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "text": "Policy requires immutable audit logging.",
                                "bbox": [0.1, 0.1, 0.8, 0.2],
                            },
                            {
                                "text": "Configuration enables immutable audit logging.",
                                "bbox": [0.1, 0.3, 0.8, 0.4],
                            },
                            {
                                "text": "Operating samples show continuous audit retention.",
                                "bbox": [0.1, 0.5, 0.8, 0.6],
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path, source)
    case = build_case_from_documents(
        case_id="document-case",
        control_id="DEMO-AU-RETENTION",
        required_slots=("design", "implementation", "operating"),
        slot_queries={
            "design": "policy requires immutable audit logging",
            "implementation": "configuration enables immutable audit logging",
            "operating": "operating samples continuous audit retention",
        },
        target_system="payments-prod",
        target_region="us-east-1",
        manifest_path=manifest,
        verifier=SupportingVerifier(),
    )
    result = assess(case)
    assert result.status == Status.SATISFIED
    assert len(result.usable_evidence_ids) == 3
    assert all(item.bbox is not None for item in case.evidence)


def test_pdf_parser_produces_normalized_grounding(tmp_path):
    source = tmp_path / "evidence.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((72, 100), "Policy requires immutable audit logging.")
    document.save(source)
    manifest = _manifest(tmp_path, source)
    loaded = load_manifest(manifest)[0]
    blocks = parse_document(loaded)
    assert blocks
    assert blocks[0].page == 1
    assert blocks[0].bbox is not None
    assert all(0 <= value <= 1 for value in blocks[0].bbox)


def test_manifest_detects_post_validation_source_change(tmp_path):
    source = tmp_path / "evidence.txt"
    source.write_text("trusted evidence", encoding="utf-8")
    manifest = _manifest(tmp_path, source)
    loaded = load_manifest(manifest)[0]
    source.write_text("mutated evidence", encoding="utf-8")
    with pytest.raises(DocumentError, match="changed after manifest validation"):
        parse_document(loaded)


def test_vlm_cannot_change_candidate_identity(tmp_path):
    source = tmp_path / "evidence.txt"
    source.write_text("Policy requires immutable audit logging.", encoding="utf-8")
    manifest = _manifest(tmp_path, source)

    class BadVerifier:
        def verify(self, slot, candidate):
            return {
                "candidate_id": "different-candidate",
                "document_id": candidate.document_id,
                "page": candidate.page,
                "text": candidate.text,
                "bbox": candidate.bbox,
                "polarity": "SUPPORTS",
                "confidence": 0.99,
            }

    with pytest.raises(VLMContractError, match="changed candidate identity"):
        build_case_from_documents(
            case_id="bad-vlm",
            control_id="DEMO",
            required_slots=("design",),
            slot_queries={"design": "policy immutable audit logging"},
            target_system="payments-prod",
            target_region="us-east-1",
            manifest_path=manifest,
            verifier=BadVerifier(),
        )


def test_manifest_rejects_symlink_even_when_target_is_inside_root(tmp_path):
    source = tmp_path / "real.txt"
    source.write_text("approved evidence", encoding="utf-8")
    link = tmp_path / "linked.txt"
    try:
        link.symlink_to(source.name)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "doc-link",
                        "source_path": link.name,
                        "sha256": sha256_file(source),
                        "document_type": "text",
                        "system": "payments-prod",
                        "region": "us-east-1",
                        "age_days": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DocumentError, match="symlinks"):
        load_manifest(manifest)


def test_manifest_and_json_structure_limits_fail_closed(tmp_path):
    oversized = tmp_path / "oversized.json"
    oversized.write_text("{}" + " " * 32, encoding="utf-8")
    with pytest.raises(DocumentError, match="manifest size"):
        load_manifest(oversized, max_manifest_bytes=8)

    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"pages": [{"page": 1, "blocks": "not-a-list"}]}), encoding="utf-8")
    manifest = _manifest(tmp_path, source, document_id="bad-json")
    document = load_manifest(manifest)[0]
    with pytest.raises(DocumentError, match="blocks must be a list"):
        parse_document(document)


def test_vlm_pack_aggregate_block_limit_fails_closed(tmp_path):
    source = tmp_path / "many.json"
    source.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {"text": "policy immutable audit logging"},
                            {"text": "configuration immutable audit logging"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path, source)
    with pytest.raises(VLMContractError, match="block limit"):
        build_case_from_documents(
            case_id="bounded",
            control_id="DEMO",
            required_slots=("design",),
            slot_queries={"design": "policy immutable audit logging"},
            target_system="payments-prod",
            target_region="us-east-1",
            manifest_path=manifest,
            verifier=SupportingVerifier(),
            max_pack_blocks=1,
        )


def test_vlm_pack_aggregate_text_limit_fails_closed(tmp_path):
    source = tmp_path / "long.txt"
    source.write_text("policy immutable audit logging " * 20, encoding="utf-8")
    manifest = _manifest(tmp_path, source)
    with pytest.raises(VLMContractError, match="text limit"):
        build_case_from_documents(
            case_id="bounded-text",
            control_id="DEMO",
            required_slots=("design",),
            slot_queries={"design": "policy immutable audit logging"},
            target_system="payments-prod",
            target_region="us-east-1",
            manifest_path=manifest,
            verifier=SupportingVerifier(),
            max_pack_text_chars=10,
        )


def test_vlm_slot_queries_must_exactly_match_required_slots(tmp_path):
    source = tmp_path / "evidence.txt"
    source.write_text("policy immutable audit logging", encoding="utf-8")
    manifest = _manifest(tmp_path, source)
    with pytest.raises(ValueError, match="exactly match"):
        build_case_from_documents(
            case_id="query-contract",
            control_id="DEMO",
            required_slots=("design", "operating"),
            slot_queries={"design": "policy immutable audit logging"},
            target_system="payments-prod",
            target_region="us-east-1",
            manifest_path=manifest,
            verifier=SupportingVerifier(),
        )


def test_retrieval_rejects_empty_query_and_invalid_top_k(tmp_path):
    from control_evidence.documents import retrieve_candidates

    source = tmp_path / "evidence.txt"
    source.write_text("policy immutable audit logging", encoding="utf-8")
    manifest = _manifest(tmp_path, source)
    documents = load_manifest(manifest)
    blocks = parse_document(documents[0])
    with pytest.raises(ValueError, match="searchable tokens"):
        retrieve_candidates(blocks, documents, "---")
    with pytest.raises(ValueError, match="top_k"):
        retrieve_candidates(blocks, documents, "policy", top_k=0)


def test_vlm_verification_rejects_source_mutated_after_parsing(tmp_path):
    source = tmp_path / "evidence.txt"
    source.write_text("Policy requires immutable audit logging.", encoding="utf-8")
    manifest = _manifest(tmp_path, source)

    class MutatingVerifier:
        def verify(self, slot, candidate):
            source.write_text("Evidence was replaced during visual verification.", encoding="utf-8")
            return VLMVerification(
                candidate_id=candidate.candidate_id,
                document_id=candidate.document_id,
                page=candidate.page,
                text=candidate.text,
                bbox=candidate.bbox,
                polarity=Polarity.SUPPORTS,
                confidence=0.99,
            )

    with pytest.raises(VLMContractError, match="changed during visual verification"):
        build_case_from_documents(
            case_id="vlm-toctou",
            control_id="DEMO",
            required_slots=("design",),
            slot_queries={"design": "policy immutable audit logging"},
            target_system="payments-prod",
            target_region="us-east-1",
            manifest_path=manifest,
            verifier=MutatingVerifier(),
        )
