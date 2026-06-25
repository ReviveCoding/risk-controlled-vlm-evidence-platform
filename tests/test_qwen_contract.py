from __future__ import annotations

import json

import pytest

from control_evidence.qwen3_vl import parse_single_json_object
from control_evidence.vlm import VLMContractError


def test_strict_parser_accepts_one_object_and_json_fence():
    assert parse_single_json_object('{"confidence": 0.9}') == {"confidence": 0.9}
    assert parse_single_json_object('```json\n{"confidence": 0.9}\n```') == {"confidence": 0.9}


@pytest.mark.parametrize(
    "payload",
    [
        'Here is the result: {"confidence": 0.9}',
        '{"confidence": 0.9} trailing',
        '{"confidence": 0.9}\n{"confidence": 0.8}',
        '[{"confidence": 0.9}]',
        '```python\n{"confidence": 0.9}\n```',
    ],
)
def test_strict_parser_rejects_prose_multiple_values_and_nonobjects(payload):
    with pytest.raises(VLMContractError):
        parse_single_json_object(payload)


def test_qwen_prompt_contains_all_grounding_identity_fields(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import control_evidence.qwen3_vl as qwen
    from control_evidence.documents import Candidate

    captured = {}

    class FakeInputs(dict):
        def to(self, device):
            captured["device"] = device
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            captured["prompt"] = messages[0]["content"][1]["text"]
            return FakeInputs({"input_ids": SimpleNamespace(shape=(1, 3)), "token_type_ids": object()})

        def batch_decode(self, trimmed, **kwargs):
            return [
                json.dumps(
                    {
                        "candidate_id": "candidate-1",
                        "document_id": "document-1",
                        "page": 2,
                        "text": "candidate text",
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                        "polarity": "SUPPORTS",
                        "confidence": 0.95,
                    }
                )
            ]

    class FakeGenerated:
        def __getitem__(self, key):
            return self

    class FakeModel:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["model_kwargs"] = kwargs
            return cls()

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return FakeGenerated()

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeProcessor,
        Qwen3VLForConditionalGeneration=FakeModel,
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setattr(qwen, "_render_pdf_page", lambda *args: object())
    verifier = qwen.Qwen3VLVerifier("fake-model")
    candidate = Candidate(
        candidate_id="candidate-1",
        document_id="document-1",
        page=2,
        text="candidate text",
        bbox=(0.1, 0.2, 0.3, 0.4),
        score=1.0,
        source_hash="a" * 64,
        source_path=str(tmp_path / "source.pdf"),
    )
    result = verifier.verify("implementation", candidate)
    assert result.candidate_id == candidate.candidate_id
    assert "Document ID: document-1" in captured["prompt"]
    assert "Page: 2" in captured["prompt"]
    assert "BBox: (0.1, 0.2, 0.3, 0.4)" in captured["prompt"]
    assert "token_type_ids" not in captured["generate_kwargs"]
    assert captured["model_kwargs"]["attn_implementation"] == "sdpa"
