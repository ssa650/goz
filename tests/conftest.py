import pytest


@pytest.fixture(autouse=True)
def no_live_sensors(monkeypatch):
    monkeypatch.setenv("GOZ_GAZE", "off")
    monkeypatch.setenv("GOZ_EEG", "off")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
