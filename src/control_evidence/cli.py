from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from . import __version__
from .archive_adapters import ArchiveSafetyError, inspect_docvqa, inspect_funsd, inspect_kleister
from .learned_routing import publish_learned_routing_experiment
from .pipeline import publish_benchmark
from .provenance import ProvenanceError, cyclone_dx_sbom
from .publication import PublicationError, TransactionalPublisher
from .vlm import VLMContractError

_EXPECTED_CLI_ERRORS = (
    ArchiveSafetyError,
    FileNotFoundError,
    json.JSONDecodeError,
    OSError,
    PermissionError,
    ProvenanceError,
    PublicationError,
    RuntimeError,
    ValueError,
    VLMContractError,
    zipfile.BadZipFile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="control-evidence")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show the original traceback for troubleshooting",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")

    pipeline = sub.add_parser("full-pipeline")
    pipeline.add_argument("--root", type=Path, default=Path("."))
    pipeline.add_argument("--seed", type=int, default=17)
    pipeline.add_argument("--run-id")

    routing = sub.add_parser("compare-risk-routing")
    routing.add_argument("--root", type=Path, default=Path("."))
    routing.add_argument("--seed", type=int, default=701)
    routing.add_argument("--n-per-family", type=int, default=360)
    routing.add_argument("--capacity-fraction", type=float, default=0.20)
    routing.add_argument("--bootstrap-samples", type=int, default=1000)
    routing.add_argument("--run-id")

    validate = sub.add_parser("validate-run")
    validate.add_argument("run_dir", type=Path)

    sbom = sub.add_parser("sbom")
    sbom.add_argument("--output", type=Path, default=Path("reports/cyclonedx-sbom.json"))

    funsd = sub.add_parser("inspect-funsd")
    funsd.add_argument("archive", type=Path)
    funsd.add_argument("--output", type=Path)

    kleister = sub.add_parser("inspect-kleister")
    kleister.add_argument("archive", type=Path)
    kleister.add_argument("--split", default="dev-0")
    kleister.add_argument("--output", type=Path)

    docvqa = sub.add_parser("inspect-docvqa")
    docvqa.add_argument("archive", type=Path)
    docvqa.add_argument("--output", type=Path)
    return parser


def _display_path(path: Path, base: Path | None = None) -> str:
    base = (base or Path.cwd()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return str(resolved)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "full-pipeline":
        output_root = args.root / "outputs"
        run_dir = publish_benchmark(output_root, seed=args.seed, run_id=args.run_id)
        latest = TransactionalPublisher(output_root).latest()
        payload = json.loads((latest / "release_gate.json").read_text(encoding="utf-8"))
        print(json.dumps({"run_dir": _display_path(run_dir), **payload}, indent=2, sort_keys=True))
        return 0 if payload["gate_status"] == "PASS" else 2
    if args.command == "compare-risk-routing":
        output_root = args.root / "outputs" / "routing_experiments"
        run_dir = publish_learned_routing_experiment(
            output_root,
            seed=args.seed,
            n_per_family=args.n_per_family,
            capacity_fraction=args.capacity_fraction,
            n_bootstrap=args.bootstrap_samples,
            run_id=args.run_id,
        )
        payload = json.loads((run_dir / "routing_experiment_summary.json").read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "run_dir": _display_path(run_dir),
                    "promotion": payload["promotion"],
                    "baseline": payload["baseline"],
                    "candidate": payload["candidate"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["promotion"]["decision"] == "PROMOTE_LEARNED_ROUTER" else 3
    if args.command == "validate-run":
        run_dir = args.run_dir.resolve()
        manifest = TransactionalPublisher(run_dir.parents[1]).validate_run(run_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "sbom":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(cyclone_dx_sbom(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(_display_path(args.output))
        return 0
    if args.command in {"inspect-funsd", "inspect-kleister", "inspect-docvqa"}:
        if args.command == "inspect-funsd":
            report = inspect_funsd(args.archive)
        elif args.command == "inspect-kleister":
            report = inspect_kleister(args.archive, args.split)
        else:
            report = inspect_docvqa(args.archive)
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except _EXPECTED_CLI_ERRORS as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        if args.debug:
            raise
        print("error: unexpected internal failure; rerun with --debug", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
