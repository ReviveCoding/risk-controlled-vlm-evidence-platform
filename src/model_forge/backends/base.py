from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Backend(ABC):
    @abstractmethod
    def write_table(self, relative_path: str, rows: Any) -> Path:
        raise NotImplementedError

    @abstractmethod
    def write_json(self, relative_path: str, value: Any) -> Path:
        raise NotImplementedError

    @abstractmethod
    def resolve(self, relative_path: str) -> Path:
        raise NotImplementedError
