"""Fal storage SDK + one-shot HTTP queue submission (no paid POST retries)."""
import httpx
import fal_client
from urllib.parse import quote


class FalError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class FalAdapter:
    demo = False

    def __init__(self, key="", transport=None):
        self.key = key.strip()
        self.http = httpx.AsyncClient(timeout=60, transport=transport, follow_redirects=False)

    def configured(self):
        return bool(self.key)

    def set_key(self, key):
        self.key = key

    def redact(self, text):
        return text.replace(self.key, "[redacted]") if self.key else text

    async def upload(self, image):
        return await fal_client.AsyncClient(key=self.key, default_timeout=60).upload(
            image.data, image.content_type, file_name=image.name)

    async def request(self, method, url, **kwargs):
        response = await self.http.request(method, url, headers={"Authorization": f"Key {self.key}"}, **kwargs)
        if not response.is_success:
            try:
                detail = response.json().get("detail", "Fal request failed.")
            except ValueError:
                detail = "Fal request failed."
            raise FalError(str(detail), response.status_code)
        return response.json() if response.content else {}

    async def submit(self, model, payload):
        # httpx does not retry this POST; ambiguous responses block further runs.
        return await self.request("POST", f"https://queue.fal.run/{model}", json=payload)

    def request_url(self, model, request_id):
        # Fal queue request routes use owner/model, excluding the endpoint subpath.
        owner, alias, *_ = model.split("/")
        return f"https://queue.fal.run/{owner}/{alias}/requests/{quote(request_id, safe='')}"

    async def status(self, model, request_id):
        return await self.request("GET", self.request_url(model, request_id)+"/status", params={"logs": "false"})

    async def result(self, model, request_id):
        return await self.request("GET", self.request_url(model, request_id))

    async def cancel(self, model, request_id):
        return await self.request("PUT", self.request_url(model, request_id)+"/cancel", timeout=15)

    async def close(self):
        await self.http.aclose()
