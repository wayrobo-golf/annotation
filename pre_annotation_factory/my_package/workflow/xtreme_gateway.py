from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class XtremeGateway:
    DATASET_FIND_PATH = "/api/dataset/findByPage"
    DATASET_CREATE_PATH = "/api/dataset/create"
    FILE_UPLOAD_URL_PATH = "/api/data/generatePresignedUrl"
    DATASET_IMPORT_PATH = "/api/data/upload"
    DATASET_IMPORT_STATUS_PATH = "/api/data/findUploadRecordBySerialNumbers"
    EXPORT_CREATE_PATH = "/api/data/export"
    EXPORT_STATUS_PATH = "/api/data/findExportRecordBySerialNumbers"
    ANNOTATION_LIST_PATH = "/api/annotate/data/listByDataIds"
    TRANSIENT_STATUS_CODES = {502, 503, 504}
    REQUEST_RETRY_ATTEMPTS = 3
    REQUEST_RETRY_DELAY_SECONDS = 1.0
    IMPORT_SUCCESS_STATUSES = {
        "COMPLETED",
        "IMPORT_COMPLETED",
        "IMPORT_SUCCESS",
        "PARSE_COMPLETED",
        "PARSE_SUCCESS",
        "SUCCESS",
        "DONE",
        "FINISHED",
    }
    IMPORT_FAILURE_STATUSES = {"FAIL", "FAILED", "ERROR"}

    def __init__(self, base_url: str, token_env: str):
        token = os.environ.get(token_env)
        if not token:
            raise ValueError(f"Missing Xtreme token in environment: {token_env}")

        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _request_json(self, method: str, path: str, payload=None, params=None):
        normalized_path = self._normalize_request_path(path)
        url = f"{self.base_url}{normalized_path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(url, data=data, method=method)
        for key, value in self.headers.items():
            request.add_header(key, value)

        last_error = None
        for attempt in range(1, self.REQUEST_RETRY_ATTEMPTS + 1):
            self._log_request_event(f"attempt {attempt}/{self.REQUEST_RETRY_ATTEMPTS}: {method} {url}")
            try:
                with urllib.request.urlopen(request) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in self.TRANSIENT_STATUS_CODES and attempt < self.REQUEST_RETRY_ATTEMPTS:
                    self._log_request_event(
                        f"retrying after HTTP {exc.code}: {method} {url}"
                    )
                    time.sleep(self.REQUEST_RETRY_DELAY_SECONDS)
                    continue
                body = _safe_read_error_body(exc)
                raise RuntimeError(
                    f"Xtreme request failed: {method} {url} -> HTTP {exc.code} {exc.reason}; "
                    f"response_body={body or '<empty>'}"
                ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < self.REQUEST_RETRY_ATTEMPTS:
                    self._log_request_event(
                        f"retrying after URLError {exc.reason}: {method} {url}"
                    )
                    time.sleep(self.REQUEST_RETRY_DELAY_SECONDS)
                    continue
                raise RuntimeError(
                    f"Xtreme request failed: {method} {url} -> {exc.reason}"
                ) from exc

        raise RuntimeError(f"Xtreme request failed: {method} {url} -> {last_error}")

    def _log_request_event(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[XtremeGateway] {timestamp} {message}")

    def _normalize_request_path(self, path: str) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.port in {8080, 8290} and path.startswith("/api/"):
            return path[len("/api") :]
        return path

    def create_or_get_dataset(self, name: str, dataset_type: str) -> str:
        response = self._request_json(
            "GET",
            self.DATASET_FIND_PATH,
            params={"name": name, "pageNo": 1, "pageSize": 100},
        )
        for item in response.get("data", {}).get("list", []):
            if item.get("name") == name:
                return item["id"]

        response = self._request_json(
            "POST",
            self.DATASET_CREATE_PATH,
            payload={"name": name, "type": dataset_type.strip().upper()},
        )
        return response["data"]["id"]

    def request_upload_url(self, filename: str, dataset_id: str) -> tuple[str, str]:
        response = self._request_json(
            "GET",
            self.FILE_UPLOAD_URL_PATH,
            params={"fileName": filename, "datasetId": dataset_id},
        )
        data = response.get("data", {})
        return data["presignedUrl"], data["accessUrl"]

    def upload_archive(self, upload_url: str, archive_path: Path) -> None:
        archive_path = Path(archive_path)
        request = urllib.request.Request(
            upload_url,
            data=archive_path.read_bytes(),
            method="PUT",
        )
        with urllib.request.urlopen(request):
            return

    def import_archive(self, dataset_id: str, access_url: str) -> str:
        response = self._request_json(
            "POST",
            self.DATASET_IMPORT_PATH,
            payload={
                "datasetId": dataset_id,
                "fileUrl": access_url,
                "source": "LOCAL",
                "resultType": "GROUND_TRUTH",
                "dataFormat": "XTREME1",
            },
        )
        return response["data"]

    def wait_import_done(
        self,
        import_task_serial: str,
        max_attempts: int = 180,
        sleep_seconds: float = 5.0,
    ) -> None:
        last_response = None
        last_status = None
        last_logged_status = None
        for _ in range(max_attempts):
            response = self._request_json(
                "GET",
                self.DATASET_IMPORT_STATUS_PATH,
                params={"serialNumbers": import_task_serial},
            )
            last_response = response
            records = response.get("data", [])
            if not records:
                time.sleep(sleep_seconds)
                continue
            status = records[0].get("status")
            last_status = status
            if status != last_logged_status:
                self._log_request_event(
                    f"import task {import_task_serial} status={status}"
                )
                last_logged_status = status
            if self._is_import_success_status(status):
                return
            if self._is_import_failure_status(status):
                raise RuntimeError(f"Xtreme import failed: {response}")
            time.sleep(sleep_seconds)
        raise TimeoutError(
            f"Timed out waiting for import task {import_task_serial}; "
            f"last_status={last_status}; last_response={last_response}"
        )

    def _is_import_success_status(self, status) -> bool:
        normalized = str(status or "").upper()
        return (
            normalized in self.IMPORT_SUCCESS_STATUSES
            or normalized.endswith("_COMPLETED")
            or normalized.endswith("_SUCCESS")
        )

    def _is_import_failure_status(self, status) -> bool:
        normalized = str(status or "").upper()
        return normalized in self.IMPORT_FAILURE_STATUSES

    def request_export(
        self,
        dataset_id: str,
        data_format: str = "XTREME1",
        select_model_run_ids: str = "-1",
    ) -> str:
        response = self._request_json(
            "GET",
            self.EXPORT_CREATE_PATH,
            params={
                "datasetId": dataset_id,
                "selectModelRunIds": select_model_run_ids,
                "dataFormat": data_format,
            },
        )
        return response["data"]

    def fetch_annotations_by_data_ids(self, data_ids) -> list[dict]:
        response = self._request_json(
            "GET",
            self.ANNOTATION_LIST_PATH,
            params={"dataIds": ",".join(str(data_id) for data_id in data_ids)},
        )
        return response.get("data", [])

    def wait_export_done(
        self,
        export_task_id: str,
        max_attempts: int = 60,
        sleep_seconds: float = 5.0,
    ) -> str:
        for _ in range(max_attempts):
            response = self._request_json(
                "GET",
                self.EXPORT_STATUS_PATH,
                params={"serialNumbers": export_task_id},
            )
            records = response.get("data", [])
            if not records:
                time.sleep(sleep_seconds)
                continue
            data = records[0]
            status = data.get("status")
            if status in {"COMPLETED", "SUCCESS", "DONE", "FINISHED"}:
                return data["filePath"]
            if status in {"FAIL", "FAILED", "ERROR"}:
                raise RuntimeError(f"Xtreme export failed: {response}")
            time.sleep(sleep_seconds)
        raise TimeoutError(f"Timed out waiting for export task {export_task_id}")

    def download_export(self, download_url: str, archive_path: Path) -> Path:
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(download_url) as response:
            archive_path.write_bytes(response.read())
        return archive_path


def _safe_read_error_body(exc: urllib.error.HTTPError) -> str:
    if exc.fp is None:
        return ""
    try:
        return exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def normalize_export_tree(download_root: Path, target_root: Path) -> Path:
    download_root = Path(download_root)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    entries = list(download_root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        source_root = entries[0]
    else:
        source_root = download_root

    for child in source_root.iterdir():
        destination = target_root / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)

    return target_root


def rebuild_result_tree_from_export_data(export_root: Path, fetch_annotations_fn) -> Path:
    export_root = Path(export_root)
    scene_to_entries: dict[Path, list[tuple[Path, dict]]] = {}

    for data_json_path in sorted(export_root.glob("Scene_*/data/*.json")):
        with data_json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        scene_to_entries.setdefault(data_json_path.parent.parent, []).append(
            (data_json_path, data)
        )

    for scene_dir, entries in scene_to_entries.items():
        result_dir = scene_dir / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        data_ids = [str(entry["dataId"]) for _, entry in entries]
        annotations_by_id = {
            str(item.get("dataId")): item
            for item in fetch_annotations_fn(data_ids)
        }

        for data_json_path, entry in entries:
            annotation = annotations_by_id.get(str(entry["dataId"]), {})
            objects = []
            for obj in annotation.get("objects", []):
                class_attributes = obj.get("classAttributes", {})
                result_obj = {
                    "type": class_attributes.get("type"),
                    "className": obj.get("className"),
                    "modelClass": class_attributes.get("modelClass"),
                    "contour": class_attributes.get("contour"),
                }
                track_id = obj.get("trackId") or class_attributes.get("trackId")
                track_name = obj.get("trackName") or class_attributes.get("trackName")
                if track_id and track_name:
                    result_obj["trackId"] = track_id
                    result_obj["trackName"] = track_name
                objects.append(result_obj)

            frame_name = entry.get("name") or data_json_path.stem
            result_path = result_dir / f"{frame_name}.json"
            result_payload = [{"objects": objects}]
            result_path.write_text(
                json.dumps(result_payload, ensure_ascii=False),
                encoding="utf-8",
            )

    return export_root
