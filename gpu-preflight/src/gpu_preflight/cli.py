"""Command-line interface for the GPU preflight."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from gpu_preflight import __version__
from gpu_preflight.image import inspect_image_reference
from gpu_preflight.inspection import InspectionReport, has_threshold
from gpu_preflight.profiles import load_profiles, resolve_profile
from gpu_preflight.render import (
    render_html,
    render_json,
    render_profiles_text,
    render_text,
)

REPORT_FILENAMES = ("report.txt", "report.json", "report.html")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-preflight",
        description=(
            "Statically inspect a container image for readiness across Nebius GPU hardware. "
            "GPU Preflight pulls and exports the selected image platform but never starts it."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="load additional hardware profiles from a directory of TOML files",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profiles", help="list built-in hardware profiles")

    inspect_parser = subparsers.add_parser("inspect", help="inspect a container image")
    inspect_parser.add_argument("image", help="customer-owned container image reference")
    inspect_parser.add_argument(
        "--platform",
        type=_platform_argument,
        metavar="PLATFORM",
        help="platform to inspect explicitly; otherwise infer it per hardware target",
    )
    _add_target_arguments(inspect_parser)
    _add_report_arguments(inspect_parser)
    _add_scan_arguments(inspect_parser)
    return parser


def _platform_argument(value: str) -> str:
    aliases = {
        "amd64": "linux/amd64",
        "arm64": "linux/arm64",
        "linux/aarch64": "linux/arm64",
        "linux/amd64": "linux/amd64",
        "linux/arm64": "linux/arm64",
        "linux/x86_64": "linux/amd64",
    }
    try:
        return aliases[value.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "platform must be arm64, amd64, linux/arm64, or linux/amd64"
        ) from error


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target",
        "-t",
        action="append",
        default=[],
        help="hardware target or alias; required and repeatable for comparison",
    )


def _add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("all", "text", "json", "html"),
        default="all",
        help=(
            "report format; the default 'all' writes report.txt, report.json, and "
            "report.html together"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="output file for an explicitly selected single format",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "directory for the default report bundle; otherwise create a timestamped "
            "directory under ./gpu-preflight-reports"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing report files in an explicitly selected output directory",
    )
    parser.add_argument(
        "--fail-on",
        choices=("blocker", "warning", "never"),
        default="warning",
        help="exit non-zero when this severity is present (default: warning)",
    )


def _add_scan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-scan-mb",
        type=int,
        default=2048,
        help="maximum candidate file content read during a filesystem scan (default: 2048 MiB)",
    )
    parser.add_argument(
        "--cuobjdump",
        default="auto",
        metavar="AUTO|PATH",
        help=(
            "CUDA artifact analyzer: auto-discover cuobjdump (default) or provide an executable "
            "path"
        ),
    )
    parser.add_argument(
        "--cuobjdump-fallback",
        default="auto",
        metavar="AUTO|PATH|OFF",
        help=(
            "second analyzer retried when cuobjdump reports an error: auto uses "
            "/usr/local/cuda-13.0/bin/cuobjdump when installed (default), or provide a path"
        ),
    )
    parser.add_argument(
        "--cuobjdump-timeout",
        type=int,
        default=30,
        metavar="SECONDS",
        help="maximum time per cuobjdump artifact inspection (default: 30)",
    )
    parser.add_argument(
        "--max-cuobjdump-files",
        type=int,
        default=256,
        metavar="COUNT",
        help="maximum CUDA candidate files passed to cuobjdump (default: 256)",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help=(
            "remove the cuobjdump file-count cap; candidate-byte, per-artifact, and timeout "
            "safety limits still apply"
        ),
    )
    parser.add_argument(
        "--max-artifact-mb",
        type=int,
        default=768,
        help="maximum size of one file passed to cuobjdump (default: 768 MiB)",
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="timeout for individual Docker metadata/pull commands (default: 3600)",
    )


def _write_output(content: str, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)


def _report_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-") or "image"


def _default_report_dir(
    image: str,
    targets: tuple[str, ...],
    generated_at: str,
) -> Path:
    generated = datetime.fromisoformat(generated_at).astimezone(UTC)
    timestamp = generated.strftime("%Y%m%dT%H%M%S.%fZ")
    image_name = image.rsplit("/", 1)[-1]
    name = "-".join((_report_slug(image_name), *targets, timestamp))
    return Path("gpu-preflight-reports") / name


def _report_paths(output_dir: Path) -> tuple[Path, ...]:
    return tuple(output_dir / name for name in REPORT_FILENAMES)


def _validate_report_destination(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"report output directory is not a directory: {output_dir}")
    existing = tuple(path for path in _report_paths(output_dir) if path.exists())
    if existing and not overwrite:
        raise ValueError(
            "report output already exists: "
            + ", ".join(str(path) for path in existing)
            + "; choose another --output-dir or pass --overwrite"
        )


def _write_report_bundle(
    report: InspectionReport,
    output_dir: Path,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    paths = _report_paths(output_dir)
    _validate_report_destination(output_dir, overwrite=overwrite)

    rendered = (
        (paths[0], render_text(report)),
        (paths[1], render_json(report)),
        (paths[2], render_html(report)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in rendered:
        path.write_text(content, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profiles = load_profiles(args.profile_dir)
        ordered_profiles = tuple(profiles[key] for key in sorted(profiles))
        if args.command == "profiles":
            sys.stdout.write(render_profiles_text(ordered_profiles))
            return 0

        if not args.target:
            raise ValueError(
                "inspect requires at least one --target to avoid unintended multi-platform pulls"
            )
        if args.format == "all" and args.output is not None:
            raise ValueError(
                "--output requires --format text, json, or html; use --output-dir for the "
                "default report bundle"
            )
        if args.format != "all" and args.output_dir is not None:
            raise ValueError("--output-dir is only valid with --format all")
        if args.format != "all" and args.overwrite:
            raise ValueError("--overwrite is only valid with --format all")
        if args.overwrite and args.output_dir is None:
            raise ValueError("--overwrite requires --output-dir")
        if args.format == "all" and args.output_dir is not None:
            _validate_report_destination(args.output_dir, overwrite=args.overwrite)

        selected = (
            tuple(resolve_profile(profiles, target) for target in args.target)
            if args.target
            else ordered_profiles
        )
        if args.max_scan_mb <= 0:
            raise ValueError("--max-scan-mb must be greater than zero")
        if args.max_artifact_mb <= 0:
            raise ValueError("--max-artifact-mb must be greater than zero")
        if args.max_cuobjdump_files <= 0:
            raise ValueError("--max-cuobjdump-files must be greater than zero")
        if args.cuobjdump_timeout <= 0:
            raise ValueError("--cuobjdump-timeout must be greater than zero")
        if args.command_timeout <= 0:
            raise ValueError("--command-timeout must be greater than zero")
        scan_options = {
            "scan_mode": "filesystem",
            "max_scan_bytes": args.max_scan_mb * 1024 * 1024,
            "cuobjdump_mode": "auto" if args.cuobjdump.lower() == "auto" else args.cuobjdump,
            "cuobjdump_timeout_seconds": args.cuobjdump_timeout,
            "cuobjdump_fallback_mode": (
                args.cuobjdump_fallback.lower()
                if args.cuobjdump_fallback.lower() in {"auto", "off"}
                else args.cuobjdump_fallback
            ),
            "max_cuobjdump_files": None if args.exhaustive else args.max_cuobjdump_files,
            "max_artifact_bytes": args.max_artifact_mb * 1024 * 1024,
            "command_timeout_seconds": args.command_timeout,
        }
        report = inspect_image_reference(
            args.image,
            selected,
            platform=args.platform,
            **scan_options,
        )
        if args.format == "all":
            output_dir = args.output_dir or _default_report_dir(
                args.image,
                tuple(profile.id for profile in selected),
                report.generated_at,
            )
            paths = _write_report_bundle(report, output_dir, overwrite=args.overwrite)
            sys.stdout.write(render_text(report))
            print("\nGenerated report bundle:", file=sys.stderr)
            for path in paths:
                print(f"  {path.resolve()}", file=sys.stderr)
            return 2 if has_threshold(report, args.fail_on) else 0

        renderer = {
            "html": render_html,
            "json": render_json,
            "text": render_text,
        }[args.format]
        _write_output(renderer(report), args.output)
        return 2 if has_threshold(report, args.fail_on) else 0
    except (KeyError, OSError, ValueError) as error:
        print(f"gpu-preflight: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
