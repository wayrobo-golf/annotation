from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_package.workflow.job_models import load_job_manifest
from my_package.workflow.job_store import JobStore
from my_package.workflow.pipeline import WorkflowPipeline


def load_dotenv_file(path: Path) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def load_default_env() -> None:
    load_dotenv_file(ROOT.parent / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotationctl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("manifest")

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("job_id")
    upload_parser.add_argument("--workspace", required=True)
    upload_parser.add_argument("--manifest")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("job_id")
    status_parser.add_argument("--workspace", required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("job_id")
    finalize_parser.add_argument("--workspace", required=True)

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("job_id")
    delete_parser.add_argument("--workspace", required=True)

    return parser


def run_submit(manifest_path: Path) -> int:
    manifest = load_job_manifest(manifest_path)
    pipeline = WorkflowPipeline(job_store=JobStore(manifest.workspace.root_dir))
    job_id = pipeline.prepare_local(manifest)

    job_dir = pipeline.job_store.jobs_root / job_id
    shutil.copy2(manifest_path, job_dir / "job.yaml")
    pipeline.upload_xtreme(job_id)

    state = pipeline.job_store.load_state(job_id)
    print(f"job_id={job_id}")
    print(f"status={state.status}")
    return 0


def run_upload(workspace: Path, job_id: str, manifest_path: Path | None = None) -> int:
    pipeline = WorkflowPipeline(job_store=JobStore(workspace))
    pipeline.upload_xtreme(job_id, manifest_path=manifest_path)
    state = pipeline.job_store.load_state(job_id)
    print(f"job_id={state.job_id}")
    print(f"status={state.status}")
    return 0


def run_status(workspace: Path, job_id: str) -> int:
    store = JobStore(workspace)
    state = store.load_state(job_id)
    print(f"job_id={state.job_id}")
    print(f"status={state.status}")
    print(f"current_step={state.current_step}")
    return 0


def run_finalize(workspace: Path, job_id: str) -> int:
    pipeline = WorkflowPipeline(job_store=JobStore(workspace))
    pipeline.finalize(job_id)
    state = pipeline.job_store.load_state(job_id)
    print(f"job_id={state.job_id}")
    print(f"status={state.status}")
    return 0


def run_delete(workspace: Path, job_id: str) -> int:
    pipeline = WorkflowPipeline(job_store=JobStore(workspace))
    pipeline.delete(job_id)
    print(f"job_id={job_id}")
    print("status=deleted")
    return 0


def main() -> int:
    load_default_env()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        return run_submit(Path(args.manifest))
    if args.command == "upload":
        return run_upload(
            Path(args.workspace),
            args.job_id,
            Path(args.manifest) if args.manifest is not None else None,
        )
    if args.command == "status":
        return run_status(Path(args.workspace), args.job_id)
    if args.command == "finalize":
        return run_finalize(Path(args.workspace), args.job_id)
    if args.command == "delete":
        return run_delete(Path(args.workspace), args.job_id)

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
