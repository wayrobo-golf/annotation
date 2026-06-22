from pathlib import Path
import io
import urllib.error

import pytest

from my_package.workflow.xtreme_gateway import XtremeGateway, normalize_export_tree


def test_gateway_reads_token_from_environment(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")

    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    assert gateway.headers["Authorization"] == "Bearer secret-token"


def test_gateway_raises_when_token_missing(monkeypatch):
    monkeypatch.delenv("XTREME1_TOKEN", raising=False)

    with pytest.raises(ValueError, match="XTREME1_TOKEN"):
        XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")


def test_request_json_retries_transient_502(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    class FakeResponse:
        def __init__(self, body: str):
            self._body = body.encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    responses = [
        urllib.error.HTTPError(
            "http://127.0.0.1:8190/api/dataset/findByPage?pageNo=1&pageSize=100",
            502,
            "Bad Gateway",
            hdrs=None,
            fp=io.BytesIO(b"temporary gateway failure"),
        ),
        FakeResponse('{"code":"OK","data":{"list":[]}}'),
    ]

    def fake_urlopen(_request):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = gateway._request_json(
        "GET",
        gateway.DATASET_FIND_PATH,
        params={"name": "DemoDataset", "pageNo": 1, "pageSize": 100},
    )

    assert response == {"code": "OK", "data": {"list": []}}


def test_request_json_logs_attempt_and_retry(monkeypatch, capsys):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    class FakeResponse:
        def __init__(self, body: str):
            self._body = body.encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    responses = [
        urllib.error.HTTPError(
            "http://127.0.0.1:8190/api/dataset/findByPage?pageNo=1&pageSize=100",
            502,
            "Bad Gateway",
            hdrs=None,
            fp=io.BytesIO(b"temporary gateway failure"),
        ),
        FakeResponse('{"code":"OK","data":{"list":[]}}'),
    ]

    def fake_urlopen(_request):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    gateway._request_json(
        "GET",
        gateway.DATASET_FIND_PATH,
        params={"name": "DemoDataset", "pageNo": 1, "pageSize": 100},
    )

    output = capsys.readouterr().out
    assert "XtremeGateway" in output
    assert "attempt 1/3" in output
    assert "retrying after HTTP 502" in output
    assert "/api/dataset/findByPage" in output


def test_request_json_keeps_api_prefix_for_nginx_base_url(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    captured = {}

    class FakeResponse:
        def read(self):
            return b'{"code":"OK","data":{"list":[]}}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request):
        captured["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    gateway._request_json("GET", gateway.DATASET_FIND_PATH, params={"pageNo": 1})

    assert captured["url"] == "http://127.0.0.1:8190/api/dataset/findByPage?pageNo=1"


def test_request_json_strips_api_prefix_for_backend_base_url(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8290", "XTREME1_TOKEN")

    captured = {}

    class FakeResponse:
        def read(self):
            return b'{"code":"OK","data":{"list":[]}}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request):
        captured["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    gateway._request_json("GET", gateway.DATASET_FIND_PATH, params={"pageNo": 1})

    assert captured["url"] == "http://127.0.0.1:8290/dataset/findByPage?pageNo=1"


def test_normalize_export_tree_flattens_single_outer_directory(tmp_path: Path):
    source = tmp_path / "download"
    nested = source / "outer" / "Scene_01" / "result"
    nested.mkdir(parents=True)
    (nested / "frame_1.json").write_text("{}", encoding="utf-8")

    target = tmp_path / "normalized"
    normalize_export_tree(source, target)

    assert (target / "Scene_01" / "result" / "frame_1.json").exists()


def test_create_or_get_dataset_returns_existing_dataset_id(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    calls = []

    def fake_request(method, path, payload=None, params=None):
        calls.append((method, path, payload, params))
        if path == "/api/dataset/findByPage":
            return {"data": {"list": [{"id": "dataset-123", "name": "DemoDataset"}]}}
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    dataset_id = gateway.create_or_get_dataset("DemoDataset", "lidar_fusion")

    assert dataset_id == "dataset-123"
    assert calls[0][0] == "GET"
    assert calls[0][1] == "/api/dataset/findByPage"


def test_create_or_get_dataset_normalizes_dataset_type_for_create(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    calls = []

    def fake_request(method, path, payload=None, params=None):
        calls.append((method, path, payload, params))
        if path == "/api/dataset/findByPage":
            return {"data": {"list": []}}
        if path == "/api/dataset/create":
            return {"data": {"id": "dataset-456"}}
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    dataset_id = gateway.create_or_get_dataset("DemoDataset", "lidar_fusion")

    assert dataset_id == "dataset-456"
    assert calls[1] == (
        "POST",
        "/api/dataset/create",
        {"name": "DemoDataset", "type": "LIDAR_FUSION"},
        None,
    )


def test_request_upload_url_uses_data_presigned_endpoint(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    calls = []

    def fake_request(method, path, payload=None, params=None):
        calls.append((method, path, payload, params))
        return {
            "data": {
                "presignedUrl": "http://upload.local/file.zip",
                "accessUrl": "http://minio.internal/file.zip",
            }
        }

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    presigned_url, access_url = gateway.request_upload_url("upload.zip", "1")

    assert presigned_url == "http://upload.local/file.zip"
    assert access_url == "http://minio.internal/file.zip"
    assert calls == [
        (
            "GET",
            "/api/data/generatePresignedUrl",
            None,
            {"fileName": "upload.zip", "datasetId": "1"},
        )
    ]


def test_import_archive_and_wait_import_done_use_v06_data_endpoints(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    responses = [
        {"data": "upload-serial-001"},
        {"data": [{"status": "PARSING"}]},
        {"data": [{"status": "PARSE_COMPLETED"}]},
    ]

    calls = []

    def fake_request(method, path, payload=None, params=None):
        calls.append((method, path, payload, params))
        return responses.pop(0)

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    upload_serial = gateway.import_archive("dataset-123", "http://minio.internal/file.zip")
    gateway.wait_import_done(upload_serial, sleep_seconds=0)

    assert upload_serial == "upload-serial-001"
    assert calls[0] == (
        "POST",
        "/api/data/upload",
        {
            "datasetId": "dataset-123",
            "fileUrl": "http://minio.internal/file.zip",
            "source": "LOCAL",
            "resultType": "GROUND_TRUTH",
            "dataFormat": "XTREME1",
        },
        None,
    )
    assert calls[1] == (
        "GET",
        "/api/data/findUploadRecordBySerialNumbers",
        None,
        {"serialNumbers": "upload-serial-001"},
    )


def test_wait_import_done_accepts_completed_status(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    def fake_request(_method, _path, payload=None, params=None):
        return {"data": [{"status": "COMPLETED"}]}

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    gateway.wait_import_done("upload-serial-001", max_attempts=1, sleep_seconds=0)


def test_wait_import_done_accepts_upload_completed_status(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    def fake_request(_method, _path, payload=None, params=None):
        return {"data": [{"status": "UPLOAD_COMPLETED"}]}

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    gateway.wait_import_done("upload-serial-001", max_attempts=1, sleep_seconds=0)


def test_wait_import_done_timeout_reports_last_status_and_response(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    def fake_request(_method, _path, payload=None, params=None):
        return {"data": [{"status": "QUEUED", "progress": 30}]}

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    with pytest.raises(TimeoutError) as exc_info:
        gateway.wait_import_done("upload-serial-001", max_attempts=1, sleep_seconds=0)

    message = str(exc_info.value)
    assert "last_status=QUEUED" in message
    assert "progress" in message


def test_request_export_and_wait_export_done_use_dataset_export(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    responses = [
        {"data": "export-serial-001"},
        {"data": [{"status": "GENERATING"}]},
        {
            "data": [
                {
                    "status": "COMPLETED",
                    "filePath": "http://download/result.zip",
                }
            ]
        },
    ]

    calls = []

    def fake_request(method, path, payload=None, params=None):
        calls.append((method, path, payload, params))
        return responses.pop(0)

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    export_serial = gateway.request_export(dataset_id="dataset-123")
    download_url = gateway.wait_export_done(export_serial, sleep_seconds=0)

    assert export_serial == "export-serial-001"
    assert download_url == "http://download/result.zip"
    assert calls[0] == (
        "GET",
        "/api/data/export",
        None,
        {
            "datasetId": "dataset-123",
            "selectModelRunIds": "-1",
            "dataFormat": "XTREME1",
        },
    )
    assert calls[1] == (
        "GET",
        "/api/data/findExportRecordBySerialNumbers",
        None,
        {"serialNumbers": "export-serial-001"},
    )


def test_fetch_annotations_by_data_ids_uses_annotate_endpoint(monkeypatch):
    monkeypatch.setenv("XTREME1_TOKEN", "secret-token")
    gateway = XtremeGateway("http://127.0.0.1:8190", "XTREME1_TOKEN")

    calls = []

    def fake_request(method, path, payload=None, params=None):
        calls.append((method, path, payload, params))
        return {"data": [{"dataId": 3807, "objects": []}]}

    monkeypatch.setattr(gateway, "_request_json", fake_request)

    records = gateway.fetch_annotations_by_data_ids([3807, 3808])

    assert records == [{"dataId": 3807, "objects": []}]
    assert calls == [
        (
            "GET",
            "/api/annotate/data/listByDataIds",
            None,
            {"dataIds": "3807,3808"},
        )
    ]
