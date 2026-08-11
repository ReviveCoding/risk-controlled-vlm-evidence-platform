from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from model_forge.backends.base import Backend
from model_forge.utils import write_json


class LocalBackend(Backend):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError("path escapes backend root")
        return path

    def write_json(self, relative_path: str, value: Any) -> Path:
        path = self.resolve(relative_path)
        write_json(path, value)
        return path

    def write_table(self, relative_path: str, rows: Any) -> Path:
        path = self.resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            raise ValueError(f"unsupported table suffix: {suffix}")
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=str(path.parent))
        os.close(descriptor)
        temp = Path(temp_name)
        try:
            if suffix == ".csv":
                frame.to_csv(temp, index=False, lineterminator="\n")
            else:
                try:
                    frame.to_parquet(temp, index=False)
                except ImportError as error:
                    marker = path.with_suffix(path.suffix + ".unavailable.json")
                    write_json(
                        marker,
                        {
                            "status": "NOT_AVAILABLE",
                            "reason": (
                                "Parquet engine is an optional dependency; CSV is the canonical "
                                "fixture table."
                            ),
                            "expected_path": str(path.name),
                            "error_type": type(error).__name__,
                        },
                    )
                    return marker
            os.replace(temp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp.unlink()
        return path
