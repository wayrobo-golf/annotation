from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    source_dir: Path
    recursive: bool = False
    overwrite_extract: bool = False
    qos_yaml_path: Path | None = None


@dataclass
class WorkspaceConfig:
    root_dir: Path


@dataclass
class XtremeConfig:
    base_url: str
    token_env: str
    dataset_name: str
    dataset_type: str


@dataclass
class OutputConfig:
    final_dataset_dir: Path


@dataclass
class ExtrinsicsSourceConfig:
    path: Path
    quaternion_order: str = "wxyz"


@dataclass
class AutoAnnotationConfig:
    base_yaml_path: Path
    overrides: dict[str, Any] = field(default_factory=dict)
    extrinsics_source: ExtrinsicsSourceConfig | None = None


@dataclass
class JobManifest:
    job_name: str
    location: str
    input: InputConfig
    workspace: WorkspaceConfig
    auto_annotation: AutoAnnotationConfig | None
    xtreme: XtremeConfig
    output: OutputConfig


@dataclass
class JobState:
    job_id: str
    status: str
    current_step: str
    last_error: str | None = None
    xtreme: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobState":
        return cls(**payload)


def load_job_manifest(path: Path) -> JobManifest:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required_sections = ("input", "workspace", "xtreme", "output")
    for key in required_sections:
        if key not in payload:
            raise ValueError(f"Missing required manifest section: {key}")

    return JobManifest(
        job_name=payload["job_name"],
        location=payload["location"],
        input=InputConfig(
            source_dir=Path(payload["input"]["source_dir"]),
            recursive=payload["input"].get("recursive", False),
            overwrite_extract=payload["input"].get("overwrite_extract", False),
            qos_yaml_path=(
                Path(payload["input"]["qos_yaml_path"])
                if payload["input"].get("qos_yaml_path") is not None
                else None
            ),
        ),
        workspace=WorkspaceConfig(root_dir=Path(payload["workspace"]["root_dir"])),
        auto_annotation=(
            AutoAnnotationConfig(
                base_yaml_path=Path(payload["auto_annotation"]["base_yaml_path"]),
                overrides=payload["auto_annotation"].get("overrides", {}),
                extrinsics_source=(
                    ExtrinsicsSourceConfig(
                        path=Path(
                            payload["auto_annotation"]["extrinsics_source"]["path"]
                        ),
                        quaternion_order=payload["auto_annotation"][
                            "extrinsics_source"
                        ].get("quaternion_order", "wxyz"),
                    )
                    if payload["auto_annotation"].get("extrinsics_source") is not None
                    else None
                ),
            )
            if "auto_annotation" in payload
            else None
        ),
        xtreme=XtremeConfig(**payload["xtreme"]),
        output=OutputConfig(
            final_dataset_dir=Path(payload["output"]["final_dataset_dir"])
        ),
    )
