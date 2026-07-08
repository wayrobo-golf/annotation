from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import yaml

from .job_models import JobManifest, JobState, load_job_manifest
from .job_store import JobStore
from .extrinsics import build_default_tf_overrides
from .xtreme_gateway import (
    XtremeGateway,
    normalize_export_tree,
    rebuild_result_tree_from_export_data,
)

_UNSET = object()
_XTREME_REMOTE_STATE_KEYS = (
    "dataset_id",
    "import_task_serial",
    "export_serial_number",
)


def _default_extract_archives(*args, **kwargs):
    from scripts.extract_archives import extract_archives

    return extract_archives(*args, **kwargs)


def _default_auto_annotation_runner(*args, **kwargs):
    from scripts.replay_rosbag_main import run_auto_annotation_job

    return run_auto_annotation_job(*args, **kwargs)


def _default_get_bags_to_process(path):
    from scripts.replay_rosbag_main import get_bags_to_process

    return get_bags_to_process(path)


def _default_merge_final_dataset(*args, **kwargs):
    from scripts.MergeXtremeDynamic import build_final_dataset

    return build_final_dataset(*args, **kwargs)


def _default_unpack_archive(archive_path: Path, target_root: Path) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(target_root))
    return target_root


@dataclass
class WorkflowPipeline:
    job_store: JobStore
    gateway_factory: type = XtremeGateway
    extract_archives_fn: callable = _default_extract_archives
    get_bags_to_process_fn: callable = _default_get_bags_to_process
    auto_annotation_runner: callable = _default_auto_annotation_runner
    unpack_archive_fn: callable = _default_unpack_archive
    normalize_export_tree_fn: callable = normalize_export_tree
    rebuild_result_tree_fn: callable = rebuild_result_tree_from_export_data
    merge_final_dataset_fn: callable = _default_merge_final_dataset

    @classmethod
    def for_tests(cls, workspace_root: Path, **overrides) -> "WorkflowPipeline":
        return cls(job_store=JobStore(workspace_root), **overrides)

    def _save_state(
        self,
        job_id: str,
        state: JobState,
        *,
        status: str | None = None,
        current_step: str | None = None,
        last_error=_UNSET,
    ) -> None:
        if status is not None:
            state.status = status
        if current_step is not None:
            state.current_step = current_step
        if last_error is not _UNSET:
            state.last_error = last_error
        self.job_store.save_state(job_id, state)

    def _prepare_auto_annotation_yaml(
        self,
        manifest: JobManifest,
        job_dir: Path,
    ) -> Path | None:
        if manifest.auto_annotation is None:
            return None

        payload = yaml.safe_load(
            manifest.auto_annotation.base_yaml_path.read_text(encoding="utf-8")
        )
        params = payload["automatic_annotation_node"]["ros__parameters"]
        if manifest.auto_annotation.extrinsics_source is not None:
            extrinsics_overrides = build_default_tf_overrides(
                manifest.auto_annotation.extrinsics_source.path,
                quaternion_order=(
                    manifest.auto_annotation.extrinsics_source.quaternion_order
                ),
            )
            params.update(extrinsics_overrides)
        params.update(manifest.auto_annotation.overrides)

        runtime_dir = job_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_yaml_path = runtime_dir / "auto_annotation.yaml"
        runtime_yaml_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return runtime_yaml_path

    def _upload_xtreme_for_job(
        self,
        job_id: str,
        manifest: JobManifest,
        state: JobState,
        xtreme_zip_path: Path,
    ) -> None:
        self._save_state(
            job_id,
            state,
            current_step="uploading_xtreme",
            last_error=None,
        )
        gateway = self.gateway_factory(
            manifest.xtreme.base_url,
            manifest.xtreme.token_env,
        )
        dataset_id = gateway.create_or_get_dataset(
            manifest.xtreme.dataset_name,
            manifest.xtreme.dataset_type.strip().upper(),
        )
        upload_url, access_url = gateway.request_upload_url(
            xtreme_zip_path.name,
            dataset_id,
        )
        gateway.upload_archive(upload_url, xtreme_zip_path)
        import_task_serial = gateway.import_archive(dataset_id, access_url)
        gateway.wait_import_done(import_task_serial)

        state.xtreme.update(
            {
                "dataset_id": dataset_id,
                "import_task_serial": import_task_serial,
            }
        )
        self._save_state(
            job_id,
            state,
            status="waiting_human_annotation",
            current_step="waiting_human_annotation",
            last_error=None,
        )

    def prepare_local(self, manifest: JobManifest) -> str:
        job_id = uuid4().hex[:12]
        job_dir = self.job_store.initialize_job_dir(job_id)
        extracted_dir = job_dir / "artifacts" / "extracted_bags"
        xtreme_upload_dir = job_dir / "artifacts" / "xtreme_upload"
        raw_origin_root = job_dir / "artifacts" / "raw_origin"
        runtime_auto_annotation_yaml = self._prepare_auto_annotation_yaml(
            manifest,
            job_dir,
        )
        state = JobState(
            job_id=job_id,
            status="running",
            current_step="preparing_inputs",
            paths={
                "extracted_dir": str(extracted_dir),
                "xtreme_upload_dir": str(xtreme_upload_dir),
                **(
                    {"auto_annotation_yaml_path": str(runtime_auto_annotation_yaml)}
                    if runtime_auto_annotation_yaml is not None
                    else {}
                ),
            },
        )
        self.job_store.save_state(job_id, state)

        try:
            self._save_state(
                job_id,
                state,
                current_step="extracting_archives",
                last_error=None,
            )
            self.extract_archives_fn(
                manifest.input.source_dir,
                extracted_dir,
                recursive=manifest.input.recursive,
                overwrite=manifest.input.overwrite_extract,
            )

            if self.get_bags_to_process_fn(extracted_dir):
                auto_annotation_input = extracted_dir
            elif self.get_bags_to_process_fn(manifest.input.source_dir):
                auto_annotation_input = manifest.input.source_dir
            else:
                auto_annotation_input = extracted_dir

            runner_kwargs = {
                "workspace_path": Path(__file__).resolve().parents[3],
                "xtreme1_output_dir": xtreme_upload_dir,
                "raw_data_archive_dir": raw_origin_root,
                "preserve_full_raw_copy": False,
                "cleanup_share_data": False,
            }
            if manifest.input.qos_yaml_path is not None:
                runner_kwargs["qos_yaml_path"] = manifest.input.qos_yaml_path
            if runtime_auto_annotation_yaml is not None:
                runner_kwargs["yaml_config_path"] = runtime_auto_annotation_yaml

            self._save_state(job_id, state, current_step="running_auto_annotation")
            artifacts = self.auto_annotation_runner(
                auto_annotation_input,
                manifest.location,
                **runner_kwargs,
            )
            state.paths.update(
                {
                    "xtreme_upload_dir": str(artifacts["xtreme_upload_dir"]),
                    "xtreme_zip_path": str(artifacts["xtreme_zip_path"]),
                    "raw_archive_dir": str(artifacts["raw_archive_dir"]),
                    "submit_manifest_path": str(artifacts["submit_manifest_path"]),
                }
            )
            self._save_state(
                job_id,
                state,
                status="prepared_for_upload",
                current_step="prepared_for_upload",
                last_error=None,
            )
            return job_id
        except Exception as exc:
            self._save_state(
                job_id,
                state,
                status="failed",
                last_error=str(exc),
            )
            raise

    def submit(self, manifest: JobManifest) -> str:
        job_id = self.prepare_local(manifest)
        state = self.job_store.load_state(job_id)
        try:
            self._upload_xtreme_for_job(
                job_id,
                manifest,
                state,
                Path(state.paths["xtreme_zip_path"]),
            )
            return job_id
        except Exception as exc:
            self._save_state(
                job_id,
                state,
                status="failed",
                last_error=str(exc),
            )
            raise

    def upload_xtreme(self, job_id: str, manifest_path: Path | None = None) -> None:
        job_dir = self.job_store.jobs_root / job_id
        job_manifest_path = job_dir / "job.yaml"
        selected_manifest_path = job_manifest_path
        if manifest_path is not None:
            selected_manifest_path = Path(manifest_path)
            if not selected_manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest override: {selected_manifest_path}")
            if not job_manifest_path.exists():
                shutil.copy2(selected_manifest_path, job_manifest_path)
                selected_manifest_path = job_manifest_path
        if not selected_manifest_path.exists():
            workspace_manifest = self.job_store.workspace_root / "job.yaml"
            if workspace_manifest.exists():
                shutil.copy2(workspace_manifest, job_manifest_path)
                selected_manifest_path = job_manifest_path
            else:
                raise FileNotFoundError(
                    f"Missing job manifest: {job_manifest_path}. "
                    f"Provide a manifest copy at {workspace_manifest} or rerun submit with the updated workflow."
                )
        manifest = load_job_manifest(selected_manifest_path)
        state = self.job_store.load_state(job_id)
        xtreme_zip_path = Path(state.paths["xtreme_zip_path"])
        if not xtreme_zip_path.exists():
            raise FileNotFoundError(f"Missing Xtreme upload archive: {xtreme_zip_path}")
        try:
            self._upload_xtreme_for_job(job_id, manifest, state, xtreme_zip_path)
        except Exception as exc:
            self._save_state(
                job_id,
                state,
                status="failed",
                last_error=str(exc),
            )
            raise

    def seed_waiting_job(self, job_id: str) -> str:
        self.job_store.initialize_job_dir(job_id)
        self.job_store.save_state(
            job_id,
            JobState(
                job_id=job_id,
                status="waiting_human_annotation",
                current_step="waiting_human_annotation",
            ),
        )
        return job_id

    def delete(self, job_id: str) -> None:
        job_dir = self.job_store.jobs_root / job_id
        state = self.job_store.load_state(job_id)
        self._delete_remote_xtreme_dataset(job_dir, state)
        self._delete_job_dir(job_dir)

    def delete_xtreme(self, job_id: str) -> None:
        job_dir = self.job_store.jobs_root / job_id
        state = self.job_store.load_state(job_id)
        self._delete_remote_xtreme_dataset(job_dir, state)
        for key in _XTREME_REMOTE_STATE_KEYS:
            state.xtreme.pop(key, None)
        self._save_state(
            job_id,
            state,
            status="prepared_for_upload",
            current_step="prepared_for_upload",
            last_error=None,
        )

    def _delete_remote_xtreme_dataset(self, job_dir: Path, state: JobState) -> None:
        dataset_id = state.xtreme.get("dataset_id")
        if not dataset_id:
            return

        manifest = load_job_manifest(job_dir / "job.yaml")
        gateway = self.gateway_factory(
            manifest.xtreme.base_url,
            manifest.xtreme.token_env,
        )
        try:
            gateway.delete_dataset(dataset_id)
        except RuntimeError as exc:
            if not _is_xtreme_missing_resource_error(exc):
                raise

    def _delete_job_dir(self, job_dir: Path) -> None:
        jobs_root = self.job_store.jobs_root.resolve()
        target = job_dir.resolve()
        if target == jobs_root or jobs_root not in target.parents:
            raise ValueError(f"Refusing to delete path outside jobs root: {job_dir}")
        shutil.rmtree(target)

    def finalize(self, job_id: str) -> None:
        job_dir = self.job_store.jobs_root / job_id
        manifest = load_job_manifest(job_dir / "job.yaml")
        state = self.job_store.load_state(job_id)
        gateway = self.gateway_factory(
            manifest.xtreme.base_url,
            manifest.xtreme.token_env,
        )

        export_archive_path = job_dir / "artifacts" / "xtreme_export" / "result.zip"
        export_unpack_dir = job_dir / "artifacts" / "xtreme_export" / "download"
        normalized_export_dir = job_dir / "artifacts" / "xtreme_export" / "normalized"
        final_output_dir = job_dir / "artifacts" / "final_dataset"
        try:
            submit_manifest_path = state.paths.get("submit_manifest_path")
            if not submit_manifest_path:
                raise FileNotFoundError(
                    f"Missing submit frame manifest in job state: {job_id}"
                )
            export_serial_number = gateway.request_export(
                dataset_id=state.xtreme["dataset_id"]
            )
            state.xtreme["export_serial_number"] = export_serial_number
            self._save_state(
                job_id,
                state,
                status="exporting_xtreme",
                current_step="exporting_xtreme",
                last_error=None,
            )

            download_url = gateway.wait_export_done(export_serial_number)
            gateway.download_export(download_url, export_archive_path)
            self.unpack_archive_fn(export_archive_path, export_unpack_dir)
            self.normalize_export_tree_fn(export_unpack_dir, normalized_export_dir)
            self.rebuild_result_tree_fn(
                normalized_export_dir,
                gateway.fetch_annotations_by_data_ids,
            )

            self._save_state(
                job_id,
                state,
                status="merging_final_dataset",
                current_step="merging_final_dataset",
            )

            merge_result = self.merge_final_dataset_fn(
                manifest.location,
                xtreme_export_root=normalized_export_dir,
                raw_archive_root=Path(state.paths["raw_archive_dir"]),
                final_kitti_output_dir=final_output_dir,
                submit_manifest_path=Path(submit_manifest_path),
            )
            state.paths["export_archive_path"] = str(export_archive_path)
            state.paths["normalized_export_dir"] = str(normalized_export_dir)
            state.paths["final_dataset_dir"] = str(
                merge_result.get("dataset_root_dir", final_output_dir)
            )
            if merge_result.get("zip_path") is not None:
                state.paths["final_zip_path"] = str(merge_result["zip_path"])
            self._save_state(
                job_id,
                state,
                status="completed",
                current_step="completed",
            )
        except Exception as exc:
            self._save_state(
                job_id,
                state,
                status="failed",
                last_error=str(exc),
            )
            raise


def _is_xtreme_missing_resource_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "http 404" in message or "not found" in message
