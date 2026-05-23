import argparse
from pathlib import Path

from gr_synth.config import load_settings
from gr_synth.prompts import PROMPTS
from gr_synth.upload import HubUploader

_PROMPT_NAMES = tuple(PROMPTS.keys())


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/shards"),
        help="Directory containing <prompt>.parquet files (default: data/shards).",
    )
    ap.add_argument(
        "--prompts",
        default=",".join(_PROMPT_NAMES),
        help="Comma-separated prompt names to upload (default: all four).",
    )
    ap.add_argument(
        "--dest-prefix",
        default="",
        help=(
            "Prefix under each prompt directory on the Hub. "
            "E.g. --dest-prefix v2 uploads to '{prompt}/v2/{filename}'. Default: ''."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded; do not push to the Hub.",
    )
    args = ap.parse_args()

    prompts = tuple(p.strip() for p in args.prompts.split(",") if p.strip())
    unknown = set(prompts) - set(_PROMPT_NAMES)
    if unknown:
        raise SystemExit(f"unknown prompts: {sorted(unknown)}")

    settings = load_settings()
    print(f"target repo: {settings.hf_repo_id}")

    plan: list[tuple[Path, str]] = []
    for prompt in prompts:
        matches = sorted(args.input.glob(f"{prompt}*.parquet"))
        if not matches:
            print(f"skip {prompt}: no {prompt}*.parquet under {args.input}")
            continue
        for local in matches:
            if args.dest_prefix:
                repo_path = f"{prompt}/{args.dest_prefix.strip('/')}/{local.name}"
            else:
                repo_path = f"{prompt}/{local.name}"
            plan.append((local, repo_path))

    if not plan:
        print("nothing to upload")
        return

    print("plan:")
    for local, repo_path in plan:
        print(f"  {local} ({_size_mb(local):.1f} MB) → {settings.hf_repo_id}:{repo_path}")

    if args.dry_run:
        print("dry-run: no uploads performed")
        return

    uploader = HubUploader(settings)
    uploader.ensure_readme()
    for local, repo_path in plan:
        uploader.upload(local, repo_path)
    print(f"done: uploaded {len(plan)} file(s)")


if __name__ == "__main__":
    main()
