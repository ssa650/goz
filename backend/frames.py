"""Bounded Fal media downloads and actual final-frame decoding."""
import asyncio
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import tempfile
from urllib.parse import urlparse, urljoin
import httpx
import imageio_ffmpeg
from PIL import Image as PILImage
from .config import MAX_IMAGE_BYTES, MAX_VIDEO_BYTES


@dataclass(frozen=True)
class Image:
    data: bytes
    content_type: str
    name: str


def verify_image(data, name="frame.png"):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Each image must be at most 10 MB.")
    try:
        with PILImage.open(BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP") or image.width*image.height > 40_000_000:
                raise ValueError("Use a valid PNG, JPEG, or WebP image, up to 40 megapixels.")
            content_type = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[image.format]
            image.verify()
    except Exception as error:
        raise ValueError("Use a valid PNG, JPEG, or WebP image, up to 40 megapixels.") from error
    return Image(data, content_type, Path(name).name)


def video_url(value):
    url = urlparse(value)
    if (url.scheme != "https" or url.username or url.password or url.port not in (None, 443)
        or not any(url.hostname == host or (url.hostname or "").endswith("."+host) for host in ("fal.media", "fal.ai"))):
        raise ValueError("Video download must use a Fal HTTPS media URL.")
    return value


async def download_video(url, destination):
    video_url(url)
    try:
        async with asyncio.timeout(120), httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            for redirects in range(4):
                async with client.stream("GET", url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        if redirects == 3 or "location" not in response.headers:
                            raise ValueError("Too many video redirects.")
                        url = video_url(urljoin(url, response.headers["location"]))
                        continue
                    response.raise_for_status()
                    if int(response.headers.get("content-length", "0")) > MAX_VIDEO_BYTES:
                        raise ValueError("Video exceeds the 100 MB download limit.")
                    size = 0
                    with destination.open("wb") as output:
                        destination.chmod(0o600)
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > MAX_VIDEO_BYTES:
                                raise ValueError("Video exceeds the 100 MB download limit.")
                            output.write(chunk)
                    return
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


async def ffmpeg(*args):
    child = await asyncio.create_subprocess_exec(imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, args),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, stderr = await asyncio.wait_for(child.communicate(), 60)
        if child.returncode:
            raise ValueError("Video decoding failed: " + stderr.decode(errors="replace")[-1200:])
    except BaseException:
        if child.returncode is None:
            child.kill()
            await child.communicate()
        raise


async def extract_last_frame(path):
    with tempfile.TemporaryDirectory(prefix="goz-frame-") as directory:
        output = Path(directory)/"last.png"
        await ffmpeg("-protocol_whitelist", "file,pipe", "-threads", "2", "-f", "mov", "-i", path,
                     "-map", "0:v:0", "-an", "-vf",
                     "scale='min(1920,iw)':'min(1920,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                     "-fps_mode", "passthrough", "-threads", "2", "-update", "1", output)
        return verify_image(output.read_bytes(), "actual-last-frame.png")


def media_metadata(path):
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        return next(reader)
    finally:
        reader.close()


async def stitch_videos(clips, destination):
    """Normalize local clips then concatenate strictly by their explicit order.

    Preserve source audio; add silence for clips without audio. Mixed image
    aspect ratios are letterboxed to the first clip's canvas, without cropping.
    Trim/pad each segment to its requested duration (provider output can vary).
    """
    import os
    ordered = sorted(clips, key=lambda clip:clip["order"])
    if not ordered or [c["order"] for c in ordered] != list(range(len(ordered))):
        raise ValueError("Cannot stitch an incomplete or ambiguous clip order.")
    if len({c["clipId"] for c in ordered}) != len(ordered):
        raise ValueError("Cannot stitch duplicate clip IDs.")
    metadata = [await asyncio.to_thread(media_metadata, c["path"]) for c in ordered]
    width, height = metadata[0]["size"]
    factor = min(1, 1920/max(width,height))
    width, height = max(2,int(width*factor)//2*2), max(2,int(height*factor)//2*2)
    temporary = destination.with_suffix(".part.mp4")
    try:
        with tempfile.TemporaryDirectory(prefix="goz-stitch-", dir=destination.parent) as directory:
            directory = Path(directory)
            files = []
            for index, (clip, meta) in enumerate(zip(ordered,metadata)):
                output = directory/f"clip-{index:03}.mp4"
                args = ["-protocol_whitelist", "file,pipe", "-threads", "2", "-i", clip["path"]]
                if not meta.get("audio_codec"):
                    args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
                args += ["-map", "0:v:0", "-map", "0:a:0" if meta.get("audio_codec") else "1:a:0",
                         "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24,tpad=stop_mode=clone:stop_duration={clip['duration']}",
                         "-af", "apad", "-t", clip["duration"], "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                         "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", output]
                await ffmpeg(*args)
                files.append(output.name)
            manifest = directory/"clips.txt"
            manifest.write_text("".join(f"file '{name}'\n" for name in files))
            await ffmpeg("-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "1", "-i", manifest,
                         "-c", "copy", "-movflags", "+faststart", temporary)
            temporary.chmod(0o600)
            os.replace(temporary,destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
