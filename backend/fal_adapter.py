"""Fal storage SDK + one-shot HTTP queue submission (no paid POST retries)."""
import re
import unicodedata
import httpx
import fal_client
from urllib.parse import quote
from .provider_errors import diagnostic, safe_text, selected_detail, user_message


def ascii_name(name):
    # Fal sends the file name in an HTTP header, which must be ASCII.
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or "upload"


class FalError(Exception):
    def __init__(self, message='', status=None, *, provider_error=None):
        super().__init__(user_message(provider_error) if provider_error else message)
        self.status = status
        self.provider_error = provider_error


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
        return safe_text(text, (self.key,), limit=1800)

    async def upload(self, image):
        try:
            return await fal_client.AsyncClient(key=self.key, default_timeout=60).upload(
                image.data, image.content_type, file_name=ascii_name(image.name))
        except Exception as error:
            response = getattr(error, 'response', None)
            if isinstance(response, httpx.Response):
                raise self.response_error(response, 'upload') from None
            info = diagnostic(str(error), stage='upload', secrets=(self.key,),
                              status=getattr(error, 'status_code', None))
            raise FalError(provider_error=info, status=info['httpStatus']) from None

    def response_error(self, response, stage, request_id=None):
        detail = ''
        body = None
        if len(response.content) <= 64 * 1024:
            try:
                body = response.json()
                detail = selected_detail(body)
            except ValueError:
                # Plain diagnostic prose only; never render an HTML error page.
                if '<' not in response.text and not re.search(r'(?m)^\S+:\s', response.text):
                    detail = response.text
        response_id = body.get('request_id') if isinstance(body, dict) else None
        info = diagnostic(detail, status=response.status_code, stage=stage,
                          request_id=request_id or response_id or response.headers.get('x-request-id'),
                          secrets=(self.key,))
        return FalError(status=response.status_code if not response.is_success else None, provider_error=info)

    async def request(self, method, url, *, stage='unknown', request_id=None, **kwargs):
        try:
            response = await self.http.request(method, url, headers={"Authorization": f"Key {self.key}"}, **kwargs)
        except httpx.RequestError as error:
            info = diagnostic(f'Provider transport interrupted ({type(error).__name__}).', stage=stage,
                              request_id=request_id, category='transport_error', secrets=(self.key,))
            raise FalError(provider_error=info) from None
        if not response.is_success:
            raise self.response_error(response, stage, request_id)
        try:
            body = response.json() if response.content else {}
        except ValueError:
            info = diagnostic('Provider returned invalid JSON.', status=response.status_code,
                              stage=stage, request_id=request_id, category='malformed_response', secrets=(self.key,))
            raise FalError(provider_error=info) from None
        if isinstance(body, dict) and body.get('error'):
            raise self.response_error(response, stage, request_id)
        return body

    async def submit(self, model, payload):
        # httpx does not retry this POST; ambiguous responses block further runs.
        return await self.request("POST", f"https://queue.fal.run/{model}", stage='submit', json=payload)

    def request_url(self, model, request_id):
        # Fal queue request routes use owner/model, excluding the endpoint subpath.
        owner, alias, *_ = model.split("/")
        return f"https://queue.fal.run/{owner}/{alias}/requests/{quote(request_id, safe='')}"

    async def status(self, model, request_id):
        return await self.request("GET", self.request_url(model, request_id)+"/status", stage='status', request_id=request_id, params={"logs": "false"})

    async def result(self, model, request_id):
        return await self.request("GET", self.request_url(model, request_id), stage='result', request_id=request_id)

    async def cancel(self, model, request_id):
        return await self.request("PUT", self.request_url(model, request_id)+"/cancel", stage='cancel', request_id=request_id, timeout=15)

    async def close(self):
        await self.http.aclose()
