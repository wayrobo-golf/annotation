import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml

from my_package.workflow.job_models import JobState, load_job_manifest
from my_package.workflow.job_store import JobStore


def load_replay_module():
    fake_io_modules = types.ModuleType("my_package.io_modules")
    fake_nusc_exporter = types.ModuleType("my_package.io_modules.nusc_json_exporter")
    fake_nusc_exporter.generate_nuscenes_metadata = lambda *_args, **_kwargs: None
    fake_io_modules.nusc_json_exporter = fake_nusc_exporter
    sys.modules["my_package.io_modules"] = fake_io_modules
    sys.modules["my_package.io_modules.nusc_json_exporter"] = fake_nusc_exporter

    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "replay_rosbag_main.py"
    )
    spec = importlib.util.spec_from_file_location("replay_rosbag_main", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_annotationctl_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "annotationctl.py"
    )
    spec = importlib.util.spec_from_file_location("annotationctl", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_submit_manifest_file(
    manifest_path: Path, items: list[dict] | None = None
) -> Path:
    if items is None:
        items = [{"scene_name": "Scene_01", "frame_id": "frame_001", "img_ext": ".png"}]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(items), encoding="utf-8")
    return manifest_path


def test_annotationctl_loads_dotenv_from_repo_root(tmp_path: Path, monkeypatch):
    module = load_annotationctl_module()
    env_path = tmp_path / ".env"
    env_path.write_text("XTREME1_TOKEN=dotenv-token\n", encoding="utf-8")

    monkeypatch.delenv("XTREME1_TOKEN", raising=False)

    module.load_dotenv_file(env_path)

    assert os.environ["XTREME1_TOKEN"] == "dotenv-token"


def test_annotationctl_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch):
    module = load_annotationctl_module()
    env_path = tmp_path / ".env"
    env_path.write_text("XTREME1_TOKEN=dotenv-token\n", encoding="utf-8")

    monkeypatch.setenv("XTREME1_TOKEN", "shell-token")

    module.load_dotenv_file(env_path)

    assert os.environ["XTREME1_TOKEN"] == "shell-token"


def test_extract_archives_returns_output_dir_when_no_archives(tmp_path: Path):
    from scripts.extract_archives import extract_archives

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    result = extract_archives(source, output, recursive=False, overwrite=False)

    assert result == output
    assert output.is_dir()


def test_build_xtreme_upload_bundle_forces_kitti_disabled():
    module = load_replay_module()
    called = {}

    def fake_build(_workspace_path, _location_str):
        called["generate_kitti"] = module.GENERATE_KITTI_DATASET

    module.build_datasets = fake_build

    module.run_xtreme_upload_bundle("/tmp/workspace", "Demo_20260416")

    assert called["generate_kitti"] is False


def test_write_submit_frame_manifest_records_scene_and_frame(tmp_path):
    module = load_replay_module()
    manifest_path = tmp_path / "submit_frames.json"
    tasks = [
        (
            "1773905257000077056",
            "cfg.json",
            "label.txt",
            "pcd.pcd",
            "img.jpg",
            "Scene_01",
            "/tmp/Scene_01",
        ),
        (
            "1773905402200267008",
            "cfg2.json",
            "label2.txt",
            "pcd2.pcd",
            "img2.png",
            "Scene_02",
            "/tmp/Scene_02",
        ),
    ]

    result = module.write_submit_frame_manifest(manifest_path, tasks)

    assert result == manifest_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == [
        {
            "scene_name": "Scene_01",
            "frame_id": "1773905257000077056",
            "img_ext": ".jpg",
        },
        {
            "scene_name": "Scene_02",
            "frame_id": "1773905402200267008",
            "img_ext": ".png",
        },
    ]


def test_select_xtreme_upload_tasks_keeps_empty_frames_when_no_positive_frames():
    module = load_replay_module()

    valid_tasks = []
    empty_tasks = [("frame_1",), ("frame_2",), ("frame_3",)]

    kept_empty_tasks, discarded_empty_tasks, tasks = module.select_xtreme_upload_tasks(
        valid_tasks,
        empty_tasks,
        max_empty_frame_ratio=0.1,
        random_seed=42,
    )

    assert kept_empty_tasks == empty_tasks
    assert discarded_empty_tasks == []
    assert sorted(tasks) == sorted(empty_tasks)


def test_submit_runs_extract_then_upload_then_waiting_state(
    tmp_path: Path, monkeypatch
):
    from my_package.workflow.pipeline import WorkflowPipeline

    manifest_path = tmp_path / "job.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {tmp_path / 'source'}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "source").mkdir()
    monkeypatch.setenv("XTREME1_TOKEN", "token")

    manifest = load_job_manifest(manifest_path)
    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            return "dataset-001"

        def request_upload_url(self, *_args, **_kwargs):
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, *_args, **_kwargs):
            return None

        def import_archive(self, *_args, **_kwargs):
            return "import-001"

        def wait_import_done(self, *_args, **_kwargs):
            return None

    def fake_extract(_source, target, **_kwargs):
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_run_auto_annotation(_target_path, _location_str, **kwargs):
        xtreme_zip = kwargs["xtreme1_output_dir"] / "upload.zip"
        xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
        xtreme_zip.write_bytes(b"zip")
        raw_origin_dir = kwargs["raw_data_archive_dir"] / "origin"
        raw_origin_dir.mkdir(parents=True, exist_ok=True)
        submit_manifest = write_submit_manifest_file(
            kwargs["xtreme1_output_dir"] / "submit_frames.json"
        )
        return {
            "xtreme_zip_path": xtreme_zip,
            "raw_archive_dir": raw_origin_dir,
            "xtreme_upload_dir": kwargs["xtreme1_output_dir"],
            "submit_manifest_path": submit_manifest,
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        extract_archives_fn=fake_extract,
        auto_annotation_runner=fake_run_auto_annotation,
    )

    job_id = pipeline.submit(manifest)
    state = pipeline.job_store.load_state(job_id)

    assert state.status == "waiting_human_annotation"
    assert "dataset_id" in state.xtreme


def test_finalize_passes_submit_manifest_path_to_merge(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    manifest_path = tmp_path / "job.yaml"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {source_dir}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_job_manifest(manifest_path)
    monkeypatch.setenv("XTREME1_TOKEN", "token")

    submit_manifest = tmp_path / "job_submit_frames.json"
    submit_manifest.write_text(
        json.dumps(
            [{"scene_name": "Scene_01", "frame_id": "f1", "img_ext": ".png"}]
        ),
        encoding="utf-8",
    )
    export_archive = tmp_path / "export.zip"
    export_archive.write_bytes(b"zip")
    normalized_export_dir = tmp_path / "normalized"
    normalized_export_dir.mkdir(parents=True, exist_ok=True)
    captured = {}

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            return "dataset-001"

        def request_upload_url(self, *_args, **_kwargs):
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, *_args, **_kwargs):
            return None

        def import_archive(self, *_args, **_kwargs):
            return "import-001"

        def wait_import_done(self, *_args, **_kwargs):
            return None

        def request_export(self, dataset_id):
            assert dataset_id == "dataset-001"
            return "export-001"

        def wait_export_done(self, export_serial_number):
            assert export_serial_number == "export-001"
            return "http://download.local/result.zip"

        def download_export(self, _download_url, archive_path):
            Path(archive_path).parent.mkdir(parents=True, exist_ok=True)
            Path(archive_path).write_bytes(export_archive.read_bytes())

        def fetch_annotations_by_data_ids(self, *_args, **_kwargs):
            return []

    def fake_extract(_source, target, **_kwargs):
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_run_auto_annotation(_target_path, _location_str, **kwargs):
        xtreme_zip = kwargs["xtreme1_output_dir"] / "upload.zip"
        xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
        xtreme_zip.write_bytes(b"zip")
        raw_origin_dir = kwargs["raw_data_archive_dir"] / "origin"
        raw_origin_dir.mkdir(parents=True, exist_ok=True)
        return {
            "xtreme_zip_path": xtreme_zip,
            "raw_archive_dir": raw_origin_dir,
            "xtreme_upload_dir": kwargs["xtreme1_output_dir"],
            "submit_manifest_path": submit_manifest,
        }

    def fake_unpack_archive(_archive_path, target_root):
        target_root.mkdir(parents=True, exist_ok=True)
        return target_root

    def fake_normalize(_download_dir, normalized_dir):
        normalized_dir.mkdir(parents=True, exist_ok=True)
        return normalized_dir

    def fake_rebuild(_normalized_dir, _fetch_fn):
        return None

    def fake_merge(_location, **kwargs):
        captured["submit_manifest_path"] = kwargs["submit_manifest_path"]
        dataset_root = tmp_path / "final_dataset"
        dataset_root.mkdir(parents=True, exist_ok=True)
        zip_path = tmp_path / "final.zip"
        zip_path.write_bytes(b"zip")
        return {"dataset_root_dir": dataset_root, "zip_path": zip_path}

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        extract_archives_fn=fake_extract,
        auto_annotation_runner=fake_run_auto_annotation,
        unpack_archive_fn=fake_unpack_archive,
        normalize_export_tree_fn=fake_normalize,
        rebuild_result_tree_fn=fake_rebuild,
        merge_final_dataset_fn=fake_merge,
    )

    job_id = pipeline.submit(manifest)
    shutil.copy2(manifest_path, pipeline.job_store.jobs_root / job_id / "job.yaml")
    pipeline.finalize(job_id)

    assert Path(captured["submit_manifest_path"]) == submit_manifest


def test_submit_wires_adapters_and_gateway(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    manifest_path = tmp_path / "job.yaml"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {source_dir}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_job_manifest(manifest_path)
    monkeypatch.setenv("XTREME1_TOKEN", "token")
    calls = []

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, name, dataset_type):
            calls.append(("dataset", name, dataset_type))
            return "dataset-001"

        def request_upload_url(self, filename, dataset_id):
            calls.append(("upload_url", filename, dataset_id))
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, upload_url, archive_path):
            calls.append(("upload_archive", upload_url, Path(archive_path).name))

        def import_archive(self, dataset_id, access_url):
            calls.append(("import_archive", dataset_id, access_url))
            return "import-001"

        def wait_import_done(self, import_task_serial):
            calls.append(("wait_import", import_task_serial))

    def fake_extract(source, target, recursive=False, overwrite=False, remove_archive=False):
        calls.append(("extract", Path(source), Path(target)))
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_run_auto_annotation(target_path, location_str, **kwargs):
        calls.append(("auto_annotation", Path(target_path), location_str))
        xtreme_zip = kwargs["xtreme1_output_dir"] / "upload.zip"
        xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
        xtreme_zip.write_bytes(b"zip")
        raw_origin_dir = kwargs["raw_data_archive_dir"] / "origin"
        raw_origin_dir.mkdir(parents=True, exist_ok=True)
        submit_manifest = write_submit_manifest_file(
            kwargs["xtreme1_output_dir"] / "submit_frames.json"
        )
        return {
            "xtreme_zip_path": xtreme_zip,
            "raw_archive_dir": raw_origin_dir,
            "xtreme_upload_dir": kwargs["xtreme1_output_dir"],
            "submit_manifest_path": submit_manifest,
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        extract_archives_fn=fake_extract,
        get_bags_to_process_fn=lambda path: [path] if Path(path) == source_dir else [],
        auto_annotation_runner=fake_run_auto_annotation,
    )

    job_id = pipeline.submit(manifest)
    state = pipeline.job_store.load_state(job_id)

    assert state.status == "waiting_human_annotation"
    assert state.xtreme["dataset_id"] == "dataset-001"
    assert state.xtreme["import_task_serial"] == "import-001"
    assert "xtreme_zip_path" in state.paths
    assert calls[0][0] == "extract"
    assert calls[1][0] == "auto_annotation"
    assert calls[2] == ("dataset", "DemoDataset", "LIDAR_FUSION")
    assert calls[3] == ("upload_url", "upload.zip", "dataset-001")


def test_submit_uses_original_source_dir_when_extract_output_has_no_bags(
    tmp_path: Path, monkeypatch
):
    from my_package.workflow.pipeline import WorkflowPipeline

    manifest_path = tmp_path / "job.yaml"
    source_dir = tmp_path / "source"
    bag_dir = source_dir / "continuous_2026_03_05-13_58_14"
    bag_dir.mkdir(parents=True)
    (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information: {}", encoding="utf-8")
    (bag_dir / "data_0.db3").write_text("", encoding="utf-8")

    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {source_dir}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_job_manifest(manifest_path)
    monkeypatch.setenv("XTREME1_TOKEN", "token")
    calls = []

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            return "dataset-001"

        def request_upload_url(self, *_args, **_kwargs):
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, *_args, **_kwargs):
            return None

        def import_archive(self, *_args, **_kwargs):
            return "import-001"

        def wait_import_done(self, *_args, **_kwargs):
            return None

    def fake_extract(_source, target, **_kwargs):
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_run_auto_annotation(target_path, location_str, **kwargs):
        calls.append(("auto_annotation", Path(target_path), location_str))
        xtreme_zip = kwargs["xtreme1_output_dir"] / "upload.zip"
        xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
        xtreme_zip.write_bytes(b"zip")
        raw_origin_dir = kwargs["raw_data_archive_dir"] / "origin"
        raw_origin_dir.mkdir(parents=True, exist_ok=True)
        submit_manifest = write_submit_manifest_file(
            kwargs["xtreme1_output_dir"] / "submit_frames.json"
        )
        return {
            "xtreme_zip_path": xtreme_zip,
            "raw_archive_dir": raw_origin_dir,
            "xtreme_upload_dir": kwargs["xtreme1_output_dir"],
            "submit_manifest_path": submit_manifest,
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        extract_archives_fn=fake_extract,
        get_bags_to_process_fn=lambda path: [path] if Path(path) == source_dir else [],
        auto_annotation_runner=fake_run_auto_annotation,
    )

    pipeline.submit(manifest)

    assert calls == [("auto_annotation", source_dir, "Demo_20260416")]


def test_submit_generates_job_specific_auto_annotation_yaml(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    base_yaml = tmp_path / "default.yaml"
    base_yaml.write_text(
        "\n".join(
            [
                "automatic_annotation_node:",
                "  ros__parameters:",
                "    global_pc_map_addr: /maps/default.pcd",
                "    pc_annotation_file_addr: /maps/default.xml",
                "    default_tf_lidar_to_ins: [0, 0, 0, 0, 0, 0]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "job.yaml"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {source_dir}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "auto_annotation:",
                f"  base_yaml_path: {base_yaml}",
                "  overrides:",
                "    global_pc_map_addr: /maps/job.pcd",
                "    default_tf_lidar_to_ins: [1, 2, 3, 4, 5, 6]",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_job_manifest(manifest_path)
    monkeypatch.setenv("XTREME1_TOKEN", "token")
    calls = []

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            return "dataset-001"

        def request_upload_url(self, *_args, **_kwargs):
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, *_args, **_kwargs):
            return None

        def import_archive(self, *_args, **_kwargs):
            return "import-001"

        def wait_import_done(self, *_args, **_kwargs):
            return None

    def fake_extract(_source, target, **_kwargs):
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_run_auto_annotation(target_path, location_str, **kwargs):
        calls.append(("auto_annotation", Path(target_path), location_str, Path(kwargs["yaml_config_path"])))
        runtime_yaml = Path(kwargs["yaml_config_path"])
        assert runtime_yaml.exists()
        content = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
        params = content["automatic_annotation_node"]["ros__parameters"]
        assert params["global_pc_map_addr"] == "/maps/job.pcd"
        assert params["default_tf_lidar_to_ins"] == [1, 2, 3, 4, 5, 6]

        xtreme_zip = kwargs["xtreme1_output_dir"] / "upload.zip"
        xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
        xtreme_zip.write_bytes(b"zip")
        raw_origin_dir = kwargs["raw_data_archive_dir"] / "origin"
        raw_origin_dir.mkdir(parents=True, exist_ok=True)
        submit_manifest = write_submit_manifest_file(
            kwargs["xtreme1_output_dir"] / "submit_frames.json"
        )
        return {
            "xtreme_zip_path": xtreme_zip,
            "raw_archive_dir": raw_origin_dir,
            "xtreme_upload_dir": kwargs["xtreme1_output_dir"],
            "submit_manifest_path": submit_manifest,
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        extract_archives_fn=fake_extract,
        get_bags_to_process_fn=lambda path: [path] if Path(path) == source_dir else [],
        auto_annotation_runner=fake_run_auto_annotation,
    )

    job_id = pipeline.submit(manifest)
    state = pipeline.job_store.load_state(job_id)

    assert calls
    assert Path(state.paths["auto_annotation_yaml_path"]).exists()


def test_submit_generates_auto_annotation_yaml_from_extrinsics_source(
    tmp_path: Path, monkeypatch
):
    from my_package.workflow.pipeline import WorkflowPipeline

    base_yaml = tmp_path / "default.yaml"
    base_yaml.write_text(
        "\n".join(
            [
                "automatic_annotation_node:",
                "  ros__parameters:",
                "    global_pc_map_addr: /maps/default.pcd",
                "    default_tf_lcam_to_lidar: [0, 0, 0, 0, 0, 0]",
                "    default_tf_lidar_to_ins: [0, 0, 0, 0, 0, 0]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    extrinsics_yaml = tmp_path / "raw_extrinsics.yaml"
    extrinsics_yaml.write_text(
        "\n".join(
            [
                "quat_cam_to_lidar: [1.0, 0, 0, 0]",
                "trans_cam_to_lidar: [1.0, 2.0, 3.0]",
                "quat_lidar_to_imu: [1.0, 0, 0, 0]",
                "trans_lidar_to_imu: [0.5, 0.25, 1.0]",
                "quat_imu_to_ins: [1.0, 0, 0, 0]",
                "trans_imu2ins: [10.0, 20.0, 30.0]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "job.yaml"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {source_dir}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "auto_annotation:",
                f"  base_yaml_path: {base_yaml}",
                "  extrinsics_source:",
                f"    path: {extrinsics_yaml}",
                "  overrides:",
                "    global_pc_map_addr: /maps/job.pcd",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_job_manifest(manifest_path)
    monkeypatch.setenv("XTREME1_TOKEN", "token")

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            return "dataset-001"

        def request_upload_url(self, *_args, **_kwargs):
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, *_args, **_kwargs):
            return None

        def import_archive(self, *_args, **_kwargs):
            return "import-001"

        def wait_import_done(self, *_args, **_kwargs):
            return None

    def fake_extract(_source, target, **_kwargs):
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_run_auto_annotation(_target_path, _location_str, **kwargs):
        runtime_yaml = Path(kwargs["yaml_config_path"])
        content = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
        params = content["automatic_annotation_node"]["ros__parameters"]
        assert params["global_pc_map_addr"] == "/maps/job.pcd"
        assert params["default_tf_lcam_to_lidar"] == pytest.approx(
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0], abs=1e-6
        )
        assert params["default_tf_lidar_to_ins"] == pytest.approx(
            [10.5, 20.25, 31.0, 0.0, 0.0, 0.0], abs=1e-6
        )

        xtreme_zip = kwargs["xtreme1_output_dir"] / "upload.zip"
        xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
        xtreme_zip.write_bytes(b"zip")
        raw_origin_dir = kwargs["raw_data_archive_dir"] / "origin"
        raw_origin_dir.mkdir(parents=True, exist_ok=True)
        submit_manifest = write_submit_manifest_file(
            kwargs["xtreme1_output_dir"] / "submit_frames.json"
        )
        return {
            "xtreme_zip_path": xtreme_zip,
            "raw_archive_dir": raw_origin_dir,
            "xtreme_upload_dir": kwargs["xtreme1_output_dir"],
            "submit_manifest_path": submit_manifest,
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        extract_archives_fn=fake_extract,
        get_bags_to_process_fn=lambda path: [path] if Path(path) == source_dir else [],
        auto_annotation_runner=fake_run_auto_annotation,
    )

    pipeline.submit(manifest)


def test_load_job_manifest_parses_optional_input_qos_yaml_path(tmp_path: Path):
    manifest_path = tmp_path / "job.yaml"
    qos_yaml_path = tmp_path / "rosbag_qos.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {tmp_path / 'source'}",
                f"  qos_yaml_path: {qos_yaml_path}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_job_manifest(manifest_path)

    assert manifest.input.qos_yaml_path == qos_yaml_path


def test_submit_passes_manifest_qos_yaml_path_to_auto_annotation_runner(
    tmp_path: Path, monkeypatch
):
    from my_package.workflow.pipeline import WorkflowPipeline

    manifest_path = tmp_path / "job.yaml"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    qos_yaml_path = tmp_path / "rosbag_qos.yaml"
    qos_yaml_path.write_text("topics: {}\n", encoding="utf-8")
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {source_dir}",
                f"  qos_yaml_path: {qos_yaml_path}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_job_manifest(manifest_path)
    monkeypatch.setenv("XTREME1_TOKEN", "token")
    calls = []

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            return "dataset-001"

        def request_upload_url(self, *_args, **_kwargs):
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, *_args, **_kwargs):
            return None

        def import_archive(self, *_args, **_kwargs):
            return "import-001"

        def wait_import_done(self, *_args, **_kwargs):
            return None

    def fake_extract(_source, target, **_kwargs):
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_run_auto_annotation(target_path, location_str, **kwargs):
        calls.append(
            (
                Path(target_path),
                location_str,
                Path(kwargs["qos_yaml_path"]),
            )
        )
        xtreme_zip = kwargs["xtreme1_output_dir"] / "upload.zip"
        xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
        xtreme_zip.write_bytes(b"zip")
        raw_origin_dir = kwargs["raw_data_archive_dir"] / "origin"
        raw_origin_dir.mkdir(parents=True, exist_ok=True)
        submit_manifest = write_submit_manifest_file(
            kwargs["xtreme1_output_dir"] / "submit_frames.json"
        )
        return {
            "xtreme_zip_path": xtreme_zip,
            "raw_archive_dir": raw_origin_dir,
            "xtreme_upload_dir": kwargs["xtreme1_output_dir"],
            "submit_manifest_path": submit_manifest,
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        extract_archives_fn=fake_extract,
        get_bags_to_process_fn=lambda path: [path] if Path(path) == source_dir else [],
        auto_annotation_runner=fake_run_auto_annotation,
    )

    pipeline.submit(manifest)

    assert calls == [(source_dir, "Demo_20260416", qos_yaml_path)]


def test_submit_records_failed_state_when_xtreme_request_fails(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    manifest_path = tmp_path / "job.yaml"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {source_dir}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_job_manifest(manifest_path)
    monkeypatch.setenv("XTREME1_TOKEN", "token")

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            raise RuntimeError("xtreme request failed hard")

    def fake_extract(_source, target, **_kwargs):
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_run_auto_annotation(_target_path, _location_str, **kwargs):
        xtreme_zip = kwargs["xtreme1_output_dir"] / "upload.zip"
        xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
        xtreme_zip.write_bytes(b"zip")
        raw_origin_dir = kwargs["raw_data_archive_dir"] / "origin"
        raw_origin_dir.mkdir(parents=True, exist_ok=True)
        submit_manifest = write_submit_manifest_file(
            kwargs["xtreme1_output_dir"] / "submit_frames.json"
        )
        return {
            "xtreme_zip_path": xtreme_zip,
            "raw_archive_dir": raw_origin_dir,
            "xtreme_upload_dir": kwargs["xtreme1_output_dir"],
            "submit_manifest_path": submit_manifest,
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        extract_archives_fn=fake_extract,
        get_bags_to_process_fn=lambda path: [path] if Path(path) == source_dir else [],
        auto_annotation_runner=fake_run_auto_annotation,
    )

    with pytest.raises(RuntimeError, match="xtreme request failed hard"):
        pipeline.submit(manifest)

    job_dirs = sorted((tmp_path / "workspace" / "jobs").iterdir())
    assert len(job_dirs) == 1
    state = JobStore(tmp_path / "workspace").load_state(job_dirs[0].name)
    assert state.status == "failed"
    assert state.current_step == "uploading_xtreme"
    assert state.last_error == "xtreme request failed hard"
    assert "xtreme_zip_path" in state.paths


def test_upload_xtreme_uses_existing_zip_and_marks_waiting(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    monkeypatch.setenv("XTREME1_TOKEN", "token")

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            return "dataset-001"

        def request_upload_url(self, *_args, **_kwargs):
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, *_args, **_kwargs):
            return None

        def import_archive(self, *_args, **_kwargs):
            return "import-001"

        def wait_import_done(self, *_args, **_kwargs):
            return None

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
    )
    job_id = pipeline.seed_waiting_job("job-123")
    job_dir = pipeline.job_store.jobs_root / job_id
    (job_dir / "job.yaml").write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {tmp_path / 'source'}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8290",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    xtreme_zip = job_dir / "artifacts" / "xtreme_upload" / "upload.zip"
    xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
    xtreme_zip.write_bytes(b"zip")
    raw_origin_dir = job_dir / "artifacts" / "raw_origin" / "origin"
    raw_origin_dir.mkdir(parents=True, exist_ok=True)
    state = pipeline.job_store.load_state(job_id)
    state.status = "failed"
    state.current_step = "uploading_xtreme"
    state.last_error = "temporary gateway failure"
    state.paths["xtreme_zip_path"] = str(xtreme_zip)
    state.paths["xtreme_upload_dir"] = str(xtreme_zip.parent)
    state.paths["raw_archive_dir"] = str(raw_origin_dir)
    pipeline.job_store.save_state(job_id, state)

    pipeline.upload_xtreme(job_id)

    state = pipeline.job_store.load_state(job_id)
    assert state.status == "waiting_human_annotation"
    assert state.current_step == "waiting_human_annotation"
    assert state.last_error is None
    assert state.xtreme["dataset_id"] == "dataset-001"
    assert state.xtreme["import_task_serial"] == "import-001"


def test_upload_xtreme_uses_workspace_manifest_when_job_manifest_missing(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    monkeypatch.setenv("XTREME1_TOKEN", "token")

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_or_get_dataset(self, *_args, **_kwargs):
            return "dataset-001"

        def request_upload_url(self, *_args, **_kwargs):
            return ("http://upload.local/file", "http://access.local/file")

        def upload_archive(self, *_args, **_kwargs):
            return None

        def import_archive(self, *_args, **_kwargs):
            return "import-001"

        def wait_import_done(self, *_args, **_kwargs):
            return None

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
    )
    job_id = pipeline.seed_waiting_job("job-legacy")
    workspace_manifest = tmp_path / "workspace" / "job.yaml"
    workspace_manifest.parent.mkdir(parents=True, exist_ok=True)
    workspace_manifest.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {tmp_path / 'source'}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8290",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    job_dir = pipeline.job_store.jobs_root / job_id
    xtreme_zip = job_dir / "artifacts" / "xtreme_upload" / "upload.zip"
    xtreme_zip.parent.mkdir(parents=True, exist_ok=True)
    xtreme_zip.write_bytes(b"zip")
    raw_origin_dir = job_dir / "artifacts" / "raw_origin" / "origin"
    raw_origin_dir.mkdir(parents=True, exist_ok=True)
    state = pipeline.job_store.load_state(job_id)
    state.status = "failed"
    state.current_step = "uploading_xtreme"
    state.last_error = "temporary gateway failure"
    state.paths["xtreme_zip_path"] = str(xtreme_zip)
    state.paths["xtreme_upload_dir"] = str(xtreme_zip.parent)
    state.paths["raw_archive_dir"] = str(raw_origin_dir)
    pipeline.job_store.save_state(job_id, state)

    pipeline.upload_xtreme(job_id)

    assert (job_dir / "job.yaml").exists()
    state = pipeline.job_store.load_state(job_id)
    assert state.status == "waiting_human_annotation"


def test_finalize_runs_export_and_merge_to_completed(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    monkeypatch.setenv("XTREME1_TOKEN", "token")
    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def request_export(self, *_args, **_kwargs):
            return "export-serial-001"

        def wait_export_done(self, *_args, **_kwargs):
            return "http://download.local/result.zip"

        def download_export(self, _download_url, archive_path):
            archive_path.write_bytes(b"zip")

        def fetch_annotations_by_data_ids(self, _data_ids):
            return []

    def fake_unpack_archive(_archive_path, target_root):
        target_root.mkdir(parents=True, exist_ok=True)

    def fake_normalize(_download_root, target_root):
        scene_dir = target_root / "Scene_01" / "data"
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "1772690563879224064.json").write_text(
            '{"dataId": 3807, "name": "1772690563879224064"}',
            encoding="utf-8",
        )
        return target_root

    def fake_merge(_location_str, **kwargs):
        return {
            "dataset_root_dir": kwargs["final_kitti_output_dir"] / "dataset",
            "zip_path": kwargs["final_kitti_output_dir"] / "dataset.zip",
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        unpack_archive_fn=fake_unpack_archive,
        normalize_export_tree_fn=fake_normalize,
        merge_final_dataset_fn=fake_merge,
    )
    job_id = pipeline.seed_waiting_job("job-123")
    job_dir = pipeline.job_store.jobs_root / job_id
    (job_dir / "job.yaml").write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {tmp_path / 'source'}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    state = pipeline.job_store.load_state(job_id)
    raw_origin_dir = job_dir / "artifacts" / "raw_origin" / "origin"
    raw_origin_dir.mkdir(parents=True, exist_ok=True)
    submit_manifest = write_submit_manifest_file(
        job_dir / "artifacts" / "xtreme_upload" / "submit_frames.json",
        [{"scene_name": "Scene_01", "frame_id": "1772690563879224064", "img_ext": ".png"}],
    )
    state.xtreme["dataset_id"] = "dataset-001"
    state.last_error = "previous failure"
    state.paths["raw_archive_dir"] = str(raw_origin_dir)
    state.paths["submit_manifest_path"] = str(submit_manifest)
    pipeline.job_store.save_state(job_id, state)

    pipeline.finalize(job_id)
    state = pipeline.job_store.load_state(job_id)

    assert state.status == "completed"
    assert state.last_error is None
    assert "final_dataset_dir" in state.paths
    assert state.xtreme["export_serial_number"] == "export-serial-001"


def test_finalize_wires_export_download_and_merge(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    monkeypatch.setenv("XTREME1_TOKEN", "token")
    calls = []

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def request_export(self, dataset_id, data_format="XTREME1", select_model_run_ids="-1"):
            calls.append(("request_export", dataset_id, data_format, select_model_run_ids))
            return "export-serial-001"

        def wait_export_done(self, export_serial_number):
            calls.append(("wait_export", export_serial_number))
            return "http://download.local/result.zip"

        def download_export(self, download_url, archive_path):
            calls.append(("download_export", download_url, Path(archive_path).name))
            archive_path.write_bytes(b"zip")

        def fetch_annotations_by_data_ids(self, data_ids):
            calls.append(("fetch_annotations", tuple(data_ids)))
            return [
                {
                    "dataId": 3807,
                    "objects": [
                        {
                            "classAttributes": {
                                "type": "3D_BOX",
                                "modelClass": "Car",
                                "contour": {
                                    "size3D": {"x": 4.0, "y": 2.0, "z": 1.5},
                                    "center3D": {"x": 10.0, "y": 0.0, "z": -1.0},
                                    "rotation3D": {"x": 0.0, "y": 0.0, "z": 0.0},
                                },
                            }
                        }
                    ],
                }
            ]

    def fake_normalize(download_root, target_root):
        calls.append(("normalize", Path(download_root), Path(target_root)))
        scene_dir = target_root / "Scene_01" / "data"
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "1772690563879224064.json").write_text(
            '{"dataId": 3807, "name": "1772690563879224064"}',
            encoding="utf-8",
        )
        return target_root

    def fake_unpack_archive(archive_path, target_root):
        calls.append(("unpack", Path(archive_path), Path(target_root)))
        target_root.mkdir(parents=True, exist_ok=True)

    def fake_merge(location_str, **kwargs):
        calls.append(("merge", location_str, kwargs["xtreme_export_root"], kwargs["raw_archive_root"]))
        result_path = (
            Path(kwargs["xtreme_export_root"])
            / "Scene_01"
            / "result"
            / "1772690563879224064.json"
        )
        assert result_path.exists()
        return {
            "dataset_root_dir": kwargs["final_kitti_output_dir"] / "dataset",
            "zip_path": kwargs["final_kitti_output_dir"] / "dataset.zip",
        }

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
        unpack_archive_fn=fake_unpack_archive,
        normalize_export_tree_fn=fake_normalize,
        merge_final_dataset_fn=fake_merge,
    )
    job_id = pipeline.seed_waiting_job("job-123")
    job_dir = pipeline.job_store.jobs_root / job_id
    (job_dir / "job.yaml").write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {tmp_path / 'source'}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    state = pipeline.job_store.load_state(job_id)
    raw_origin_dir = job_dir / "artifacts" / "raw_origin" / "origin"
    raw_origin_dir.mkdir(parents=True, exist_ok=True)
    submit_manifest = write_submit_manifest_file(
        job_dir / "artifacts" / "xtreme_upload" / "submit_frames.json",
        [{"scene_name": "Scene_01", "frame_id": "1772690563879224064", "img_ext": ".png"}],
    )
    state.xtreme["dataset_id"] = "dataset-001"
    state.paths["raw_archive_dir"] = str(raw_origin_dir)
    state.paths["submit_manifest_path"] = str(submit_manifest)
    pipeline.job_store.save_state(job_id, state)

    pipeline.finalize(job_id)
    state = pipeline.job_store.load_state(job_id)

    assert state.status == "completed"
    assert state.xtreme["export_serial_number"] == "export-serial-001"
    assert "final_dataset_dir" in state.paths
    assert calls[0] == ("request_export", "dataset-001", "XTREME1", "-1")
    assert ("fetch_annotations", ("3807",)) in calls


def test_finalize_updates_step_before_waiting_for_export(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    monkeypatch.setenv("XTREME1_TOKEN", "token")

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def request_export(self, *_args, **_kwargs):
            return "export-serial-001"

        def wait_export_done(self, *_args, **_kwargs):
            state = pipeline.job_store.load_state(job_id)
            assert state.status == "exporting_xtreme"
            assert state.current_step == "exporting_xtreme"
            assert state.xtreme["export_serial_number"] == "export-serial-001"
            raise RuntimeError("export still running")

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
    )
    job_id = pipeline.seed_waiting_job("job-123")
    job_dir = pipeline.job_store.jobs_root / job_id
    (job_dir / "job.yaml").write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {tmp_path / 'source'}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    state = pipeline.job_store.load_state(job_id)
    raw_origin_dir = job_dir / "artifacts" / "raw_origin" / "origin"
    raw_origin_dir.mkdir(parents=True, exist_ok=True)
    submit_manifest = write_submit_manifest_file(
        job_dir / "artifacts" / "xtreme_upload" / "submit_frames.json",
        [{"scene_name": "Scene_01", "frame_id": "frame_001", "img_ext": ".png"}],
    )
    state.xtreme["dataset_id"] = "dataset-001"
    state.paths["raw_archive_dir"] = str(raw_origin_dir)
    state.paths["submit_manifest_path"] = str(submit_manifest)
    pipeline.job_store.save_state(job_id, state)

    with pytest.raises(RuntimeError, match="export still running"):
        pipeline.finalize(job_id)


def test_finalize_records_failed_state_and_error_message(tmp_path: Path, monkeypatch):
    from my_package.workflow.pipeline import WorkflowPipeline

    monkeypatch.setenv("XTREME1_TOKEN", "token")

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        def request_export(self, *_args, **_kwargs):
            return "export-serial-001"

        def wait_export_done(self, *_args, **_kwargs):
            raise RuntimeError("export failed hard")

    pipeline = WorkflowPipeline.for_tests(
        tmp_path / "workspace",
        gateway_factory=FakeGateway,
    )
    job_id = pipeline.seed_waiting_job("job-123")
    job_dir = pipeline.job_store.jobs_root / job_id
    (job_dir / "job.yaml").write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {tmp_path / 'source'}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    state = pipeline.job_store.load_state(job_id)
    raw_origin_dir = job_dir / "artifacts" / "raw_origin" / "origin"
    raw_origin_dir.mkdir(parents=True, exist_ok=True)
    submit_manifest = write_submit_manifest_file(
        job_dir / "artifacts" / "xtreme_upload" / "submit_frames.json",
        [{"scene_name": "Scene_01", "frame_id": "frame_001", "img_ext": ".png"}],
    )
    state.xtreme["dataset_id"] = "dataset-001"
    state.paths["raw_archive_dir"] = str(raw_origin_dir)
    state.paths["submit_manifest_path"] = str(submit_manifest)
    pipeline.job_store.save_state(job_id, state)

    with pytest.raises(RuntimeError, match="export failed hard"):
        pipeline.finalize(job_id)

    state = pipeline.job_store.load_state(job_id)
    assert state.status == "failed"
    assert state.current_step == "exporting_xtreme"
    assert state.last_error == "export failed hard"


def test_annotationctl_status_prints_saved_state(tmp_path: Path):
    module = load_annotationctl_module()
    calls = []

    def fake_run_status(workspace, job_id):
        calls.append((Path(workspace), job_id))
        return 0

    module.run_status = fake_run_status
    previous_argv = sys.argv[:]
    sys.argv = [
        "annotationctl",
        "status",
        "job-001",
        "--workspace",
        str(tmp_path / "workspace"),
    ]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv

    assert result == 0
    assert calls == [(tmp_path / "workspace", "job-001")]


def test_annotationctl_submit_creates_job_and_prints_job_id(
    tmp_path: Path, monkeypatch
):
    module = load_annotationctl_module()
    manifest = tmp_path / "job.yaml"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                f"  source_dir: {source_dir}",
                "workspace:",
                f"  root_dir: {tmp_path / 'workspace'}",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                f"  final_dataset_dir: {tmp_path / 'final'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run_submit(manifest_path):
        calls.append(Path(manifest_path))
        return 0

    module.run_submit = fake_run_submit
    previous_argv = sys.argv[:]
    sys.argv = ["annotationctl", "submit", str(manifest)]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv

    assert result == 0
    assert calls == [manifest]


def test_annotationctl_upload_dispatches_job_id_and_workspace(tmp_path: Path):
    module = load_annotationctl_module()
    calls = []

    def fake_run_upload(workspace, job_id, manifest_path=None):
        calls.append((Path(workspace), job_id, manifest_path))
        return 0

    module.run_upload = fake_run_upload
    previous_argv = sys.argv[:]
    sys.argv = [
        "annotationctl",
        "upload",
        "job-001",
        "--workspace",
        str(tmp_path / "workspace"),
    ]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv

    assert result == 0
    assert calls == [(tmp_path / "workspace", "job-001", None)]


def test_annotationctl_upload_dispatches_manifest_when_provided(tmp_path: Path):
    module = load_annotationctl_module()
    calls = []
    manifest = tmp_path / "job.yaml"

    def fake_run_upload(workspace, job_id, manifest_path=None):
        calls.append((Path(workspace), job_id, Path(manifest_path) if manifest_path else None))
        return 0

    module.run_upload = fake_run_upload
    previous_argv = sys.argv[:]
    sys.argv = [
        "annotationctl",
        "upload",
        "job-001",
        "--workspace",
        str(tmp_path / "workspace"),
        "--manifest",
        str(manifest),
    ]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv

    assert result == 0
    assert calls == [(tmp_path / "workspace", "job-001", manifest)]


def test_annotationctl_finalize_marks_job_completed(tmp_path: Path):
    module = load_annotationctl_module()
    calls = []

    def fake_run_finalize(workspace, job_id):
        calls.append((Path(workspace), job_id))
        return 0

    module.run_finalize = fake_run_finalize
    previous_argv = sys.argv[:]
    sys.argv = [
        "annotationctl",
        "finalize",
        "job-001",
        "--workspace",
        str(tmp_path / "workspace"),
    ]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv

    assert result == 0
    assert calls == [(tmp_path / "workspace", "job-001")]
