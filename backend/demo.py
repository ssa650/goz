"""Explicit local demo adapter. No credentials or provider requests."""
import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4
from .frames import ffmpeg


class DemoAdapter:
    demo = True

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.requests = {}

    def configured(self):
        return True

    def set_key(self, key):
        raise ValueError("Demo mode does not use an API key.")

    def redact(self, text):
        return text

    async def upload(self, image):
        return "demo-image:"+hashlib.sha256(image.data).hexdigest()

    async def submit(self, model, payload):
        request_id = str(uuid4())
        self.requests[request_id] = payload
        return {"request_id": request_id}

    async def status(self, model, request_id):
        await asyncio.sleep(.6)
        return {"status": "COMPLETED"}

    async def result(self, model, request_id):
        payload = self.requests[request_id]
        path = self.directory/f"{request_id}.mp4"
        hue = ["0x5478b5", "0x8b5a91", "0x4c9586"][len(list(self.directory.glob('*.mp4'))) % 3]
        await ffmpeg("-f", "lavfi", "-i", f"color=c={hue}:s=640x360:r=24",
                     "-t", str(payload["duration"]), "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path)
        return {"video": {"url": "demo:"+request_id}, "timings": {"inference": .6}}

    async def cancel(self, model, request_id):
        return {}

    async def close(self):
        pass
