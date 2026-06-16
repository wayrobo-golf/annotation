from pathlib import Path

from my_package.workflow.job_models import load_job_manifest


def test_load_job_manifest_reads_required_fields(tmp_path: Path):
    manifest_path = tmp_path / "job.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                "  source_dir: /data/source",
                "  recursive: false",
                "  overwrite_extract: true",
                "workspace:",
                "  root_dir: /tmp/workspace",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                "  final_dataset_dir: /tmp/final",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_job_manifest(manifest_path)

    assert manifest.job_name == "demo_job"
    assert manifest.location == "Demo_20260416"
    assert manifest.input.source_dir == Path("/data/source")
    assert manifest.input.overwrite_extract is True
    assert manifest.xtreme.dataset_name == "DemoDataset"


def test_load_job_manifest_reads_auto_annotation_section(tmp_path: Path):
    manifest_path = tmp_path / "job.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                "  source_dir: /data/source",
                "workspace:",
                "  root_dir: /tmp/workspace",
                "auto_annotation:",
                "  base_yaml_path: /configs/default.yaml",
                "  overrides:",
                "    global_pc_map_addr: /maps/colored_lidar_merged.pcd",
                "    default_tf_lidar_to_ins: [1, 2, 3, 4, 5, 6]",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                "  final_dataset_dir: /tmp/final",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_job_manifest(manifest_path)

    assert manifest.auto_annotation is not None
    assert manifest.auto_annotation.base_yaml_path == Path("/configs/default.yaml")
    assert manifest.auto_annotation.overrides["global_pc_map_addr"] == "/maps/colored_lidar_merged.pcd"
    assert manifest.auto_annotation.overrides["default_tf_lidar_to_ins"] == [1, 2, 3, 4, 5, 6]


def test_load_job_manifest_reads_auto_annotation_extrinsics_source(tmp_path: Path):
    manifest_path = tmp_path / "job.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "job_name: demo_job",
                "location: Demo_20260416",
                "input:",
                "  source_dir: /data/source",
                "workspace:",
                "  root_dir: /tmp/workspace",
                "auto_annotation:",
                "  base_yaml_path: /configs/default.yaml",
                "  extrinsics_source:",
                "    path: /configs/raw_extrinsics.yaml",
                "    quaternion_order: wxyz",
                "xtreme:",
                "  base_url: http://127.0.0.1:8190",
                "  token_env: XTREME1_TOKEN",
                "  dataset_name: DemoDataset",
                "  dataset_type: lidar_fusion",
                "output:",
                "  final_dataset_dir: /tmp/final",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_job_manifest(manifest_path)

    assert manifest.auto_annotation is not None
    assert manifest.auto_annotation.extrinsics_source is not None
    assert manifest.auto_annotation.extrinsics_source.path == Path(
        "/configs/raw_extrinsics.yaml"
    )
    assert manifest.auto_annotation.extrinsics_source.quaternion_order == "wxyz"
