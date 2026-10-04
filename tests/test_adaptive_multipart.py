"""Current browser FormData through ASGI; all generation and media I/O are mocked."""
import asyncio
import base64
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from backend.adaptive import routes
from backend.adaptive.session import AdaptiveSession
from backend.config import MAX_IMAGE_BYTES, MAX_PROMPT_BYTES
from backend.frames import verify_image
from test_backend import harness, png


@pytest.fixture(scope="module")
def browser_form():
    def build(opening="start"):
        command = ["node", "tests/adaptive-request.test.js", "--multipart-fixture"]
        if opening == "opening":
            command.append("--opening-video")
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        return dict(content=base64.b64decode(value["body"]),
                    headers={"Content-Type": value["contentType"]})
    return build


def full_fields():
    return dict(use_saved_sequence="0", premise="Two explorers find a box.",
                timeline="0-5 Ana: Look at the box.", tracker="color",
                playback_mode="stream", eegRunMode="cumulative_prior_clips",
                characters=json.dumps([dict(name="Ana"), dict(name="Bea")]),
                duration="15", resolution="480P", objects='[{"name":"box"}]')


@pytest.mark.asyncio
async def test_actual_browser_form_reproduces_previous_eight_field_failure(tmp_path, monkeypatch, browser_form):
    monkeypatch.setattr(routes, "MAX_SESSION_FIELDS", 8)
    async with harness(tmp_path) as (engine, adapter, client):
        response = await client.post("/api/adaptive/sessions", **browser_form())
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == "Too many fields. Maximum number of fields is 8."
        assert not engine.jobs and not adapter.uploads and not adapter.submissions


@pytest.mark.asyncio
async def test_actual_browser_form_creates_and_finishes_mock_submission(tmp_path, monkeypatch, browser_form):
    import backend.adaptive.local_tracker as local_tracker
    import backend.progressive_media as progressive
    warmups, probes = [], []

    async def close():
        pass

    async def prewarm(provider, on_diagnostic):
        warmups.append(provider)
        on_diagnostic(dict(diagnostic_event="warm_ready"))
        return SimpleNamespace(request_close=lambda: None, close=close)

    async def probe(url):
        probes.append(url)
        return dict(bytes=1000, metadata="early_moov", rangeSupported=True)

    monkeypatch.setattr(local_tracker, "prewarm_local_tracker", prewarm)
    monkeypatch.setattr(progressive, "probe_video", probe)
    monkeypatch.setattr(AdaptiveSession, "start_detection", lambda *args: None)
    async with harness(tmp_path) as (engine, adapter, client):
        async def media(job):
            return tmp_path / "mock-validated.mp4"

        async def extract(path):
            return verify_image(png())

        monkeypatch.setattr(engine, "media_path", media)
        monkeypatch.setattr(engine, "extractor", extract)
        response = await client.post("/api/adaptive/sessions", **browser_form())
        assert response.status_code == 202, response.text
        public = response.json()
        assert public["playbackMode"] == "stream"
        assert public["eegRunMode"] == "cumulative_prior_clips"
        session = client._transport.app.state.sensors.session
        await asyncio.wait_for(session.task, 5)
        await asyncio.gather(*session.local_media_tasks.values())
        assert session.status == "running", session.error
        assert session.tracker == "color" and warmups == ["color"]
        assert session.story["premise"] == full_fields()["premise"]
        assert session.timeline == full_fields()["timeline"]
        assert session.objects[0]["name"] == "box"
        assert len(adapter.uploads) == len(adapter.submissions) == len(engine.jobs) == 1
        model, payload = adapter.submissions[0]
        assert payload["duration"] == 15 and payload["resolution"] == "480P"
        assert payload["image_url"].endswith("frame-1.png")
        assert probes == ["https://fal.media/request-1.mp4"]
        clip = session.clips[0]
        assert clip["status"] == "ready" and clip["playbackDelivery"] == "stream"
        assert clip["localMediaStatus"] == "validated"
        assert next(iter(engine.jobs.values()))["status"] == "completed"
        saved = json.loads((session.dir / "session.json").read_text())
        assert saved["playbackMode"] == "stream" and saved["eegRunMode"] == "cumulative_prior_clips"


@pytest.mark.asyncio
async def test_actual_browser_video_form_and_supported_optional_fields(tmp_path, monkeypatch, browser_form):
    async def no_generation(self):
        pass
    monkeypatch.setattr(AdaptiveSession, "start", no_generation)
    async with harness(tmp_path) as (engine, adapter, client):
        response = await client.post("/api/adaptive/sessions", **browser_form("opening"))
        assert response.status_code == 202, response.text
        session = client._transport.app.state.sensors.session
        assert session.opening_video.read_bytes() == b"mock-video"
        assert session.playback_mode == "stream" and session.eeg_run_mode == "cumulative_prior_clips"
        session.stop()
        fields = full_fields() | dict(scene_limit="1", eeg_run_mode="baseline")
        response = await client.post("/api/adaptive/sessions", data=fields,
                                     files=[("start", ("opening.png", png(), "image/png")),
                                            ("opening", ("opening.mp4", b"mock-video", "video/mp4"))])
        assert response.status_code == 202, response.text
        assert client._transport.app.state.sensors.session.max_scenes == 1
        assert not engine.jobs and not adapter.submissions


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["fields", "files", "part_size", "image_size", "video_size",
                                    "body_size", "unexpected_file", "duplicate_images", "malformed", "bad_json", "bad_mode"])
async def test_invalid_full_forms_are_bounded_before_generation(tmp_path, monkeypatch, case):
    async with harness(tmp_path) as (engine, adapter, client):
        fields = full_fields()
        files = [("start", ("opening.png", png(), "image/png"))]
        expected_status, expected_text = 400, None
        if case == "fields":
            # Count repeated fields as well as unique names, exactly at limit + 1.
            files = [(key, (None, value)) for key, value in fields.items()] + files
            files += [("premise", (None, "extra"))] * (routes.MAX_SESSION_FIELDS + 1 - len(fields))
            fields = {}
            expected_text = "Too many fields"
        elif case == "files":
            files *= routes.MAX_SESSION_FILES + 1
            expected_text = "Too many files"
        elif case == "part_size":
            fields["timeline"] = "x" * (MAX_PROMPT_BYTES + 1)
            expected_text = "Part exceeded maximum size"
        elif case == "image_size":
            files = [("start", ("large.png", b"x" * (MAX_IMAGE_BYTES + 1), "image/png"))]
        elif case == "video_size":
            monkeypatch.setattr(routes, "MAX_OPENING_BYTES", 16)
            files = [("opening", ("large.mp4", b"x" * 17, "video/mp4"))]
            expected_text = "at most 100 MB"
        elif case == "unexpected_file":
            files.append(("unrecognized", ("other.png", png(), "image/png")))
            expected_text = "Unexpected upload field"
        elif case == "duplicate_images":
            files *= 2
            expected_text = "Too many start images"
        elif case == "bad_json":
            fields["characters"] = "not-json"
        elif case == "bad_mode":
            fields["eegRunMode"] = "unsupported"
            expected_text = "Choose prior played clips"
        if case == "malformed":
            response = await client.post("/api/adaptive/sessions", content=b'--bad\r\nContent-Disposition: form-data\r\n\r\nx\r\n--bad--\r\n',
                                         headers={"Content-Type": "multipart/form-data; boundary=bad"})
            expected_text = 'name'
        elif case == "body_size":
            response = await client.post("/api/adaptive/sessions", content=b"small",
                                         headers={"Content-Length": str(13 * MAX_IMAGE_BYTES + MAX_PROMPT_BYTES + 12001)})
            expected_status, expected_text = 413, "Upload batch is too large"
        else:
            response = await client.post("/api/adaptive/sessions", data=fields, files=files)
        assert response.status_code == expected_status, response.text
        if expected_text:
            assert expected_text in response.text
        assert not engine.jobs and not adapter.uploads and not adapter.submissions


@pytest.mark.asyncio
async def test_other_multipart_routes_keep_eight_field_limit(tmp_path):
    async with harness(tmp_path) as (engine, adapter, client):
        response = await client.post("/api/jobs", data={f"field{i}": "x" for i in range(9)},
                                     files={"start": ("opening.png", png(), "image/png")})
        assert response.status_code == 400 and "Maximum number of fields is 8" in response.text
        assert not adapter.submissions
