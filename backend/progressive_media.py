"""Same-origin range delivery of completed Fal MP4s; never unfinished rendering."""
import asyncio
import re
import struct

import httpx

from .config import MAX_VIDEO_BYTES
from .frames import video_url

PROBE_BYTES = 32768


def range_bounds(value, total):
    if value is None:
        return 0, total-1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise ValueError("Use one valid media byte range.")
    first, last = match.groups()
    if not first:
        if not int(last):
            raise ValueError("Empty media range.")
        return max(0, total-int(last)), total-1
    start = int(first)
    end = min(int(last), total-1) if last else total-1
    if start > end or start >= total:
        raise ValueError("Media range is outside the completed video.")
    return start, end


async def open_video(url, byte_range=None, *, transport=None):
    """Own the client/response until the caller closes both, including on errors."""
    client = httpx.AsyncClient(timeout=30, follow_redirects=False, transport=transport)
    response = None
    try:
        for redirects in range(4):
            url = video_url(url)
            request = client.build_request("GET", url, headers={"Accept-Encoding":"identity", **({"Range":byte_range} if byte_range else {})})
            response = await client.send(request, stream=True)
            if response.status_code in (301,302,303,307,308):
                from urllib.parse import urljoin
                location = response.headers.get("location")
                await response.aclose()
                if redirects == 3 or not location:
                    raise ValueError("Too many media redirects.")
                url = video_url(urljoin(url,location))
                continue
            if response.status_code not in (200,206):
                raise ValueError(f"Completed stream unavailable (HTTP {response.status_code}).")
            if response.headers.get("content-type","").split(";",1)[0].strip().lower() not in ("video/mp4", "application/mp4"):
                raise ValueError("Completed stream did not return an MP4.")
            return client,response
        raise ValueError("Media stream unavailable.")
    except BaseException as error:
        if response is not None:
            await response.aclose()
        await client.aclose()
        if isinstance(error,httpx.RequestError):
            raise ValueError(f"Completed stream transport failed ({type(error).__name__}).") from None
        raise


def validate_range(response, start, end, total, requested=True):
    expected = end-start+1
    if int(response.headers.get("content-length", "-1")) != expected:
        raise ValueError("Completed media length changed or is unavailable.")
    if response.status_code == 206:
        if response.headers.get("content-range") != f"bytes {start}-{end}/{total}":
            raise ValueError("Completed stream returned an inconsistent byte range.")
    elif response.status_code != 200 or requested or start != 0 or end != total-1:
        raise ValueError("Completed stream does not support the requested byte range.")


def early_mp4_metadata(data):
    """Require a complete moov before mdat in a bounded prefix; otherwise download."""
    at, ftyp = 0, False
    while at+8 <= len(data):
        size,kind = struct.unpack_from(">I4s",data,at)
        header=8
        if size==1:
            if at+16>len(data): return False
            size=struct.unpack_from(">Q",data,at+8)[0];header=16
        if size<header or at+size>len(data): return False
        if kind==b'ftyp': ftyp=True
        if kind==b'mdat': return False
        if kind==b'moov': return ftyp and size>header
        at+=size
    return False


async def probe_video(url, *, transport=None):
    client,response = await open_video(url,f"bytes=0-{PROBE_BYTES-1}",transport=transport)
    try:
        content_range = response.headers.get('content-range','')
        match = re.fullmatch(r'bytes 0-(\d+)/(\d+)',content_range)
        if response.status_code != 206 or not match:
            raise ValueError("Progressive media ranges are unavailable; use download playback.")
        end,total = map(int,match.groups())
        if not 0 < total <= MAX_VIDEO_BYTES or end != min(PROBE_BYTES,total)-1:
            raise ValueError("Progressive media size is invalid or exceeds the download limit.")
        validate_range(response,0,end,total)
        data=bytearray()
        async with asyncio.timeout(15):
            async for chunk in response.aiter_bytes(chunk_size=8192):
                data.extend(chunk)
                if len(data)>PROBE_BYTES:
                    raise ValueError("Media probe exceeded its bounded prefix.")
        if len(data)!=end+1 or not early_mp4_metadata(data):
            raise ValueError("MP4 metadata is not available in the bounded prefix; use download playback.")
        return dict(bytes=total,probeBytes=len(data),metadata='early_moov',rangeSupported=True)
    finally:
        await response.aclose()
        await client.aclose()
