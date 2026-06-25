from __future__ import annotations

import argparse
import gzip
import os
import tarfile
import tempfile
from pathlib import Path

DEFAULT_EPOCH = 1_704_067_200


def normalize_sdist(source: Path, output: Path, *, epoch: int = DEFAULT_EPOCH) -> None:
    source = source.resolve(strict=True)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tar_name = tempfile.mkstemp(prefix="normalized-sdist-", suffix=".tar")
    os.close(descriptor)
    temporary_tar = Path(tar_name)
    try:
        with (
            tarfile.open(source, "r:gz") as incoming,
            tarfile.open(temporary_tar, "w", format=tarfile.PAX_FORMAT) as outgoing,
        ):
            for member in sorted(incoming.getmembers(), key=lambda item: item.name):
                normalized = tarfile.TarInfo(member.name)
                normalized.type = member.type
                normalized.mode = member.mode
                normalized.uid = 0
                normalized.gid = 0
                normalized.uname = ""
                normalized.gname = ""
                normalized.mtime = epoch
                normalized.linkname = member.linkname
                normalized.devmajor = member.devmajor
                normalized.devminor = member.devminor
                normalized.pax_headers = {}
                file_object = None
                if member.isfile():
                    normalized.size = member.size
                    file_object = incoming.extractfile(member)
                    if file_object is None:
                        raise ValueError(f"could not read sdist member: {member.name}")
                outgoing.addfile(normalized, file_object)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        temporary_output = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as raw_output, temporary_tar.open("rb") as raw_tar:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_output, mtime=epoch, compresslevel=9
                ) as stream:
                    while chunk := raw_tar.read(1024 * 1024):
                        stream.write(chunk)
                raw_output.flush()
                os.fsync(raw_output.fileno())
            os.replace(temporary_output, output)
        except Exception:
            temporary_output.unlink(missing_ok=True)
            raise
    finally:
        temporary_tar.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    args = parser.parse_args()
    normalize_sdist(args.source, args.output, epoch=args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
