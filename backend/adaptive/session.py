"""One adaptive viewing session: the generate -> watch -> adapt loop.

Scene N+1 is planned when playback of scene N passes ANALYZE_AT of its
duration (so generation overlaps viewing), from the response measured so
far. Saved bundles keep their original prompt and frame pair. Custom
scenes start from the actual last frame of the previous one.
All signals are logged with wall-clock timestamps under
data/adaptive/<session>/.
"""
import asyncio
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from . import director, fusion, profile as profiles, tracks
from ..frames import ffmpeg
from ..clip_settings import generation_options

ANALYZE_AT = float(os.getenv("GOZ_ANALYZE_AT", "0.7"))
MAX_SCENES = int(os.getenv("GOZ_MAX_SCENES", "4"))
TRACKER = os.getenv("GOZ_TRACKER", "fal")
DEMO_COLORS = ["0xe07a5f", "0x3d85c6", "0x81b29a", "0xf2cc8f"]


class AdaptiveSession:
    def __init__(self, engine, gaze, eeg, directory, premise, characters, duration, resolution, opening,
                 opening_video=None, timeline="", clip_bundles=None):
        self.engine, self.gaze, self.eeg = engine, gaze, eeg
        self.id = str(uuid4())
        self.dir = Path(directory)/"adaptive"/self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.names = [c["name"] for c in characters]
        self.story = dict(premise=premise, characters=characters, scenes=[])
        self.duration, self.resolution, self.opening = duration, resolution, opening
        self.opening_video, self.timeline = opening_video, timeline
        self.clip_bundles = deepcopy(clip_bundles or [])
        self.max_scenes = min(MAX_SCENES, len(self.clip_bundles)) if self.clip_bundles else MAX_SCENES
        self.profile = profiles.new_profile(self.names)
        self.clips, self.events = [], []
        self.stage, self.error, self.status = "starting", None, "running"
        self.playing, self.rect = None, None
        self.task = None
        self.started = time.time()

    # -- public state ------------------------------------------------------
    def log(self, kind, **data):
        event = dict(t=time.time(), kind=kind, **data)
        self.events.append(event)
        with (self.dir/"events.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")

    def set_stage(self, stage):
        self.stage = stage
        self.log("stage", stage=stage)

    def public(self):
        clips = [{k: v for k, v in c.items() if k not in ("ticks", "timeline", "image")} for c in self.clips]
        return dict(id=self.id, status=self.status, stage=self.stage, error=self.error, names=self.names,
                    story=self.story, profile=self.profile, clips=clips, playing=self.playing,
                    duration=self.duration, demo=self.engine.adapter.demo, maxScenes=self.max_scenes,
                    events=self.events[-12:])

    # -- live helpers (dashboard + simulator) ------------------------------
    def current_clip(self):
        return self.clips[self.playing] if self.playing is not None and self.playing < len(self.clips) else None

    def live_video_time(self):
        clip = self.current_clip()
        if not clip or not clip["ticks"]:
            return None
        tick = clip["ticks"][-1]
        if not tick["playing"] or time.time() - tick["wall"] > 1.0:
            return None
        return tick["video_t"] + (time.time() - tick["wall"])

    def live_boxes(self):
        clip, t = self.current_clip(), self.live_video_time()
        if clip is None or t is None:
            return {}, None
        return tracks.boxes_at(clip.get("track") or [], t), self.rect

    def live_gaze(self):
        clip, sample = self.current_clip(), self.gaze.latest()
        if not clip or not sample:
            return None
        labelled = fusion.label([sample], clip["ticks"][-3:], clip.get("track") or [])
        return labelled[0] if labelled else None

    def live_target(self):
        g = self.live_gaze()
        return g["target"] if g else None

    # -- browser playback reports -------------------------------------------
    def tick(self, index, video_t, playing, rect, wall):
        if not 0 <= index < len(self.clips):
            raise ValueError("Unknown clip.")
        clip = self.clips[index]
        tick = dict(wall=wall, video_t=video_t, playing=playing, rect=rect)
        clip["ticks"].append(tick)
        self.playing, self.rect = index, rect
        if clip["status"] == "ready" and playing:
            clip.update(status="playing", playStartedAt=wall - video_t)
            self.log("play", clip=index)
        if (playing and video_t >= ANALYZE_AT * clip["duration"] and not clip.get("analysisStarted")
                and self.status == "running"):
            clip["analysisStarted"] = True
            self.task = asyncio.create_task(self.adapt(index))

    def ended(self, index):
        if 0 <= index < len(self.clips):
            self.clips[index]["status"] = "watched"
            if not self.clips[index].get("analysisStarted") and self.status == "running":
                self.clips[index]["analysisStarted"] = True
                self.task = asyncio.create_task(self.adapt(index))

    # -- the loop -------------------------------------------------------------
    async def start(self):
        decision = dict(focus=None, tension="same", dialogue="same", pacing="same", tone=None, event=False,
                        reasons=["Opening scene: no viewer data yet"])
        if self.opening_video:
            self.task = asyncio.create_task(self.predefined_scene(decision))
        else:
            self.task = asyncio.create_task(self.make_scene(decision, self.opening, seeds=None, changes=[]))

    async def predefined_scene(self, decision):
        """Scene 1 is the uploaded episode segment; only the rest is generated."""
        try:
            import imageio_ffmpeg
            path = self.opening_video
            _, seconds = await asyncio.to_thread(imageio_ffmpeg.count_frames_and_secs, str(path))
            seconds = round(seconds, 2)
            beats = director.parse_timeline(self.timeline, self.names, seconds)
            plan = dict(scene_title="Opening segment (predefined)", summary=self.story["premise"],
                        beats=beats or [dict(t0=0.0, t1=seconds, description="Opening", characters=list(self.names),
                                             dialogue=False, speaker=None, tags=[])],
                        video_prompt="(predefined clip, not generated)", change_note="The episode starts as written.")
            clip = dict(index=0, status="tracking", decision=decision, changes=[], plan=plan, writer="predefined",
                        duration=seconds, ticks=[], seeds=None, createdAt=time.time(), path=str(path))
            self.clips.append(clip)
            self.story["scenes"].append(dict(title=plan["scene_title"], summary=plan["summary"]))
            self.set_stage("tracking characters in the opening segment")
            clip["track"] = await self.track(path, None, None, seconds)
            clip.update(url=f"/api/adaptive/clips/{self.id}/0", status="ready",
                        detected=sum(1 for f in clip["track"] if f["boxes"]))
            self.set_stage("scene 1 ready")
        except Exception as error:
            self.fail(error)

    async def track(self, path, seeds, focus, seconds=None):
        if self.engine.adapter.demo:
            return self.demo_track(focus, seconds or self.duration)
        if TRACKER == "people":
            return await tracks.track_clip(path, self.engine.directory, self.names, seeds)
        return await tracks.detect_characters(path, self.story["characters"], self.engine.adapter.key)

    async def adapt(self, index):
        try:
            clip = self.clips[index]
            self.set_stage(f"analyzing scene {index + 1}")
            t0 = clip.get("playStartedAt") or (clip["ticks"][0]["wall"] if clip["ticks"] else time.time())
            t1 = time.time()
            samples = self.gaze.window(t0, t1)
            eeg = self.eeg.window(t0, t1)
            timeline = fusion.label(samples, clip["ticks"], clip.get("track") or [])
            analysis = fusion.analyze(timeline, eeg, clip.get("track") or [], self.names, clip.get("plan"))
            clip.update(analysis=analysis, timeline=timeline)
            self.dump(f"scene{index + 1}_signals.json", dict(gaze=samples, eeg=eeg, ticks=clip["ticks"],
                                                              timeline=timeline, analysis=analysis))
            self.profile, changes = profiles.update(self.profile, analysis)
            self.log("profile", clip=index, changes=changes)
            if len(self.clips) >= self.max_scenes:
                self.set_stage(f"finished: {self.max_scenes}-scene limit reached")
                self.status = "finished"
                return
            self.story["viewer_analysis"] = analysis
            # Only the no-key rehearsal uses a heuristic decision. OpenAI sees
            # the measurements and chooses the adjustment itself.
            decision = profiles.decide(self.profile, analysis) if not os.getenv("OPENAI_API_KEY", "").strip() else {}
            self.set_stage(f"extracting last frame of scene {index + 1}")
            image = None if self.clip_bundles else await self.engine.extractor(clip["path"])
            await self.make_scene(decision, image, seeds=tracks.final_boxes(clip.get("track") or []), changes=changes)
        except Exception as error:
            self.fail(error)

    async def make_scene(self, decision, image, seeds, changes):
        index = len(self.clips)
        try:
            self.engine.generation_guard()
            bundle = self.clip_bundles[index] if self.clip_bundles else None
            duration = bundle["duration"] if bundle else self.duration
            self.story["next_prompt"] = bundle["prompt"] if bundle else self.story["premise"]
            self.set_stage(f"deciding a subtle adjustment for scene {index + 1}")
            plan, writer = await director.write_scene(self.story, self.profile, decision, duration)
            decision = plan["decision"]
            if self.status != "running":
                return
            self.engine.generation_guard()
            clip = dict(index=index, status="generating", decision=decision, changes=changes, plan=plan,
                        writer=writer, duration=duration, ticks=[], seeds=seeds, createdAt=time.time(),
                        bundle=deepcopy(bundle), basePrompt=plan["base_prompt"])
            self.clips.append(clip)
            self.story["scenes"].append(dict(title=plan["scene_title"], summary=plan["summary"]))
            self.log("decision", clip=index, decision=decision, writer=writer, change_note=plan["change_note"])
            self.set_stage(f"generating scene {index + 1}")
            if bundle:
                options = generation_options(self.engine.clips.settings(bundle))
                options["prompt"] = plan["video_prompt"]
                images = {}  # Stored first/end frames win; never replace them with another clip's frame.
            else:
                options = dict(mode="frames", prompt=plan["video_prompt"], duration=duration, resolution=self.resolution)
                images = {"start": [image]}
            job = self.engine.new_job(options, adaptiveSession=self.id, sceneIndex=index,
                                      clipId=bundle["id"] if bundle else None,
                                      sourceClipId=bundle["id"] if bundle else None,
                                      basePrompt=plan["base_prompt"], engagementDecision=decision,
                                      decisionModel=writer, decisionRequestId=plan.get("decisionRequestId"))
            clip["jobId"] = job["id"]
            await self.engine.run_job(job, images)
            if job["status"] != "completed":
                raise ValueError(job.get("error", "Generation did not complete."))
            path = await self.engine.media_path(job)
            if self.engine.adapter.demo:
                await self.demo_render(path, decision.get("focus"), duration)
            self.set_stage(f"tracking characters in scene {index + 1}")
            clip["track"] = await self.track(path, seeds, decision.get("focus"), duration)
            clip.update(path=str(path), url=f"/api/jobs/{job['id']}/video", status="ready",
                        generatedS=round(time.time() - clip["createdAt"], 1),
                        detected=sum(1 for f in clip["track"] if f["boxes"]))
            self.set_stage(f"scene {index + 1} ready")
        except Exception as error:
            self.fail(error)

    def demo_boxes(self, focus):
        """Demo mode only: one box per character (focus = big, centred)."""
        n = len(self.names)
        boxes = {}
        for i, name in enumerate(self.names):
            if name == focus:
                boxes[name] = [0.33, 0.12, 0.67, 0.98]
            else:
                slot = (i + 0.5) / n
                w = 0.18 if focus else 0.24
                boxes[name] = [max(0.02, slot - w / 2), 0.35 if focus else 0.2, min(0.98, slot + w / 2), 0.95]
        return boxes

    def demo_track(self, focus, seconds):
        boxes = self.demo_boxes(focus)
        return [dict(t=round(i / tracks.SAMPLE_FPS, 3), boxes=boxes) for i in range(int(seconds * tracks.SAMPLE_FPS) + 1)]

    async def demo_render(self, path, focus, duration=None):
        filters = ",".join(f"drawbox=x=iw*{b[0]}:y=ih*{b[1]}:w=iw*{b[2] - b[0]}:h=ih*{b[3] - b[1]}:color={DEMO_COLORS[i % 4]}:t=fill"
                           for i, b in enumerate(self.demo_boxes(focus).values()))
        temporary = Path(str(path) + ".demo.mp4")
        await ffmpeg("-f", "lavfi", "-i", "color=c=0x1d2330:s=640x360:r=24", "-t", str(duration or self.duration), "-vf", filters,
                     "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", temporary)
        os.replace(temporary, path)

    def fail(self, error):
        self.error = self.engine.error(error)
        self.status = "failed"
        self.set_stage("stopped")

    def stop(self):
        if self.status == "running":
            self.status = "stopped"
            self.set_stage("stopped by viewer")

    def dump(self, name, data):
        (self.dir/name).write_text(json.dumps(data))
