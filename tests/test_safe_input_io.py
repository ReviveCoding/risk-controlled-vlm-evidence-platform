from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from model_forge.utils import (
    MAX_JSON_DOCUMENT_BYTES,
    MAX_YAML_DOCUMENT_BYTES,
    load_bounded_json,
    load_numeric_npy,
    load_yaml,
)
from model_forge.workloads.qwen_embeddings import encode_qwen_csv


class SafeInputIoTests(unittest.TestCase):
    def test_json_and_yaml_control_inputs_are_size_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized_json = root / "evidence.json"
            oversized_yaml = root / "contract.yaml"
            with oversized_json.open("wb") as stream:
                stream.truncate(MAX_JSON_DOCUMENT_BYTES + 1)
            with oversized_yaml.open("wb") as stream:
                stream.truncate(MAX_YAML_DOCUMENT_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "supported bound"):
                load_bounded_json(oversized_json)
            with self.assertRaisesRegex(ValueError, "supported bound"):
                load_yaml(oversized_yaml)

    def test_numeric_npy_loader_rejects_pickle_archive_and_resource_excess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.npy"
            np.save(valid, np.array([0.1, 0.2], dtype=np.float64), allow_pickle=False)
            np.testing.assert_array_equal(load_numeric_npy(valid), np.array([0.1, 0.2]))

            objects = root / "objects.npy"
            np.save(objects, np.array([{"unsafe": True}], dtype=object), allow_pickle=True)
            with self.assertRaisesRegex(ValueError, "real numeric"):
                load_numeric_npy(objects)

            archive = root / "archive.npz"
            np.savez(archive, values=np.array([0.1]))
            with self.assertRaisesRegex(ValueError, "NPY"):
                load_numeric_npy(archive)

            with self.assertRaisesRegex(ValueError, "file-size limit"):
                load_numeric_npy(valid, max_file_bytes=8)
            with self.assertRaisesRegex(ValueError, "element limit"):
                load_numeric_npy(valid, max_elements=1)

            truncated = root / "truncated.npy"
            truncated.write_bytes(valid.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "payload size"):
                load_numeric_npy(truncated)

    def test_qwen_contracts_are_checked_before_optional_gpu_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, r"\.npy"):
                encode_qwen_csv(
                    root / "input.csv",
                    root / "embeddings.bin",
                    revision="exact-revision",
                )
            with self.assertRaisesRegex(ValueError, "batch_size"):
                encode_qwen_csv(
                    root / "input.csv",
                    root / "embeddings.npy",
                    revision="exact-revision",
                    batch_size=0,
                )
            output = root / "embeddings.npy"
            colliding_input = output.with_suffix(".npy.manifest.json")
            colliding_input.write_text("complaint_narrative\ntext\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be different"):
                encode_qwen_csv(colliding_input, output, revision="exact-revision")

    def test_qwen_output_is_validated_and_transactional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_csv = root / "input.csv"
            output_npy = root / "embeddings.npy"
            pd.DataFrame({"complaint_narrative": ["a", "b"]}).to_csv(input_csv, index=False)

            class FakeSentenceTransformer:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    self.max_seq_length = 0

                def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
                    return np.ones((len(texts), 3), dtype=np.float32)

            fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
            with (
                patch.dict("sys.modules", {"sentence_transformers": fake_module}),
                patch(
                    "model_forge.workloads.qwen_embeddings.qwen_gpu_probe",
                    return_value={"status": "READY", "reason": None, "device": "fake-cuda"},
                ),
            ):
                manifest = encode_qwen_csv(
                    input_csv,
                    output_npy,
                    revision="exact-revision",
                )
            self.assertEqual(manifest["embedding_shape"], [2, 3])
            expected = np.full((2, 3), 1.0 / np.sqrt(3.0), dtype=np.float32)
            np.testing.assert_allclose(load_numeric_npy(output_npy), expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(list(root.glob(".embeddings.*.npy")), [])

            output_npy.write_bytes(b"known-good")
            with (
                patch.dict("sys.modules", {"sentence_transformers": fake_module}),
                patch(
                    "model_forge.workloads.qwen_embeddings.qwen_gpu_probe",
                    return_value={"status": "READY", "reason": None, "device": "fake-cuda"},
                ),
                patch("model_forge.workloads.qwen_embeddings.np.save", side_effect=OSError("stop")),
                self.assertRaisesRegex(OSError, "stop"),
            ):
                encode_qwen_csv(input_csv, output_npy, revision="exact-revision")
            self.assertEqual(output_npy.read_bytes(), b"known-good")
            self.assertEqual(list(root.glob(".embeddings.*.npy")), [])

            manifest_path = output_npy.with_suffix(".npy.manifest.json")
            output_npy.write_bytes(b"known-good")
            manifest_path.write_bytes(b"known-manifest")
            with (
                patch.dict("sys.modules", {"sentence_transformers": fake_module}),
                patch(
                    "model_forge.workloads.qwen_embeddings.qwen_gpu_probe",
                    return_value={"status": "READY", "reason": None, "device": "fake-cuda"},
                ),
                patch(
                    "model_forge.workloads.qwen_embeddings.write_json",
                    side_effect=OSError("manifest-stop"),
                ),
                self.assertRaisesRegex(OSError, "manifest-stop"),
            ):
                encode_qwen_csv(input_csv, output_npy, revision="exact-revision")
            self.assertEqual(output_npy.read_bytes(), b"known-good")
            self.assertEqual(manifest_path.read_bytes(), b"known-manifest")


if __name__ == "__main__":
    unittest.main()
