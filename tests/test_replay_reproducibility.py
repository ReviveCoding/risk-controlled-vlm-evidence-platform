from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from model_forge.pipeline import verify_replay
from model_forge.utils import runtime_dependency_metadata
from model_forge.workloads.models import DeterministicTfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]


class ReplayReproducibilityTests(unittest.TestCase):
    def test_vectorizer_artifact_excludes_process_local_stop_word_cache(self) -> None:
        documents = np.array(["alpha beta", "beta gamma", "alpha gamma"])
        payloads = []
        for process_marker in (101, 202):
            vectorizer = DeterministicTfidfVectorizer().fit(documents)
            vectorizer._stop_words_id = process_marker
            stream = io.BytesIO()
            joblib.dump(vectorizer, stream)
            payloads.append(stream.getvalue())
            restored = joblib.load(io.BytesIO(stream.getvalue()))
            self.assertFalse(hasattr(restored, "_stop_words_id"))
            self.assertTrue(
                np.array_equal(
                    vectorizer.transform(documents).toarray(),
                    restored.transform(documents).toarray(),
                )
            )
        self.assertEqual(payloads[0], payloads[1])

    def test_runtime_identity_includes_effective_numeric_execution(self) -> None:
        libraries = [
            {
                "architecture": "Haswell",
                "filepath": "ignored/install/path.dll",
                "internal_api": "openblas",
                "num_threads": 1,
                "prefix": "libopenblas",
                "threading_layer": "pthreads",
                "user_api": "blas",
                "version": "1.2.3",
            }
        ]
        with (
            patch.dict(
                os.environ,
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
            ),
            patch("model_forge.utils.threadpool_info", return_value=libraries),
        ):
            first = runtime_dependency_metadata()
        numeric = first["numeric_execution"]
        self.assertEqual(numeric["libraries"][0]["num_threads"], 1)
        self.assertNotIn("filepath", numeric["libraries"][0])
        libraries[0]["num_threads"] = 2
        with patch("model_forge.utils.threadpool_info", return_value=libraries):
            second = runtime_dependency_metadata()
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_two_independent_fixture_processes_replay_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            environment = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            for output in (first, second):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "model_forge.cli",
                        "demo",
                        "--config",
                        "configs/experiment_test.yaml",
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            comparison = verify_replay(first, second)
            self.assertEqual(comparison["status"], "PASS", comparison)


if __name__ == "__main__":
    unittest.main()
