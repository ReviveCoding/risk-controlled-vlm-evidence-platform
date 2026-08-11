"""Versioned, read-only resources shipped with the Python distribution."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

CONFIG_PACKAGE = "model_forge.resources.configs"


def bundled_config(name: str) -> Path:
    """Return a bundled config path for normal wheel/editable installations."""
    relative = Path(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) > 2
        or (len(relative.parts) == 2 and relative.parts[0] != "workloads")
        or not name.endswith(".yaml")
    ):
        raise ValueError(f"invalid bundled config name: {name}")
    resource = files(CONFIG_PACKAGE).joinpath(*relative.parts)
    if not resource.is_file():
        raise FileNotFoundError(f"bundled config does not exist: {name}")
    return Path(str(resource))


def bundled_config_names() -> list[str]:
    return sorted(item.name for item in files(CONFIG_PACKAGE).iterdir() if item.name.endswith(".yaml"))


def resolve_config(path: Path) -> Path:
    """Resolve a user path, falling back only for the known configs/<name> form."""
    if path.exists():
        return path.resolve()
    if path.parent == Path("configs"):
        return bundled_config(path.name)
    if path.parent == Path("configs/workloads"):
        return bundled_config(f"workloads/{path.name}")
    raise FileNotFoundError(path)
