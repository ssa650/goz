"""Own sensor child processes and gate new generations on calibrated live data."""
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from uuid import uuid4

from .fal_adapter import FalError

ROOT = Path(__file__).resolve().parent.parent


class SensorSetup:
    def __init__(self, sensors, directory, demo=False, spawn=None):
        self.sensors = sensors
        self.directory = Path(directory).resolve()
        self.required = not demo and os.getenv("GOZ_REQUIRE_SENSORS", "1") != "0"
        self.phase = "waiting" if self.required else "demo" if demo else "disabled"
        self.message = "Preparing sensors…" if self.required else "Demo mode: real sensor calibration is not required." if demo else "Sensor checks disabled by server configuration."
        self.error = None
        self.task = None
        self.children = {}
        self.spawn = spawn or asyncio.create_subprocess_exec
        self.gaze_calibrated = False
        self.gaze_report = None
        self.attempt_dir = None
        self.logs = {}
        self.readers = {}
        self.stopping = False

    def snapshot(self):
        eeg, gaze = self.sensors.eeg, self.sensors.gaze
        muse = eeg.status()
        gaze_live = gaze.status()["live"] and any(s.get("valid") and s.get("face", True)
                        for s in gaze.window(time.time() - 2, time.time()))
        latest = eeg.latest()
        bridge = self.children.get("muse")
        muse_live = (muse["live"] and bool(latest and not latest[3]) and not muse["qualityError"]
                     and (bridge is None or bridge.returncode is None))
        stream = self.children.get("gaze")
        gaze_alive = stream is not None and stream.returncode is None
        ready = not self.required or (self.phase == "ready" and self.gaze_calibrated and gaze_live and gaze_alive
                                     and muse_live and muse["calibrated"])
        message = self.message
        if self.required and self.phase == "ready" and not ready:
            message = ("Muse stream lost. Retry sensor setup to collect a fresh baseline." if not muse["calibrated"]
                       else "Adjust Muse contacts and sit still: " + muse["qualityError"] if not muse_live
                       else "Gaze signal lost. Face the camera; retry setup if the camera stopped.")
        return dict(required=self.required, generationReady=bool(ready), phase=self.phase,
                    message=message, error=self.error, canRetry=self.required and (self.phase == "failed" or self.phase == "ready" and not ready),
                    muse=muse, gaze=dict(**gaze.status(), calibrated=self.gaze_calibrated,
                                        valid=gaze_live, report=self.gaze_report))

    def require_ready(self):
        state = self.snapshot()
        if not state["generationReady"]:
            raise FalError("Sensor setup is not ready. " + (state["error"] or state["message"]), 409)

    def set_phase(self, phase, message):
        self.phase, self.message = phase, message

    async def start(self):
        if not self.required or self.task and not self.task.done():
            return
        self.task = asyncio.create_task(self.run())

    async def wait_for(self, predicate, timeout, child=None):
        deadline = time.monotonic() + timeout
        while not predicate():
            if self.sensors.eeg.state.startswith("Muse connection error:"):
                raise ValueError(self.sensors.eeg.state)
            if child and child.returncode is not None:
                raise ValueError("Sensor process stopped. " + self.log_tail(child))
            if time.monotonic() > deadline:
                raise ValueError("Sensor setup timed out. " + self.message)
            await asyncio.sleep(.25)

    def log_tail(self, child):
        name = next((name for name, item in self.children.items() if item is child), "")
        # Children never receive provider credentials; keep errors short and readable.
        return " ".join(self.logs.get(name, [])[-4:])[-700:]

    async def launch(self, name, argv, cwd):
        if name in self.children and self.children[name].returncode is None:
            raise ValueError(f"{name} is already running.")
        env = {k:v for k,v in os.environ.items() if k not in ("FAL_KEY", "FAL_API_KEY", "OPENAI_API_KEY")}
        env["PYTHONPATH"] = str(ROOT)
        child = await self.spawn(*map(str, argv), cwd=str(cwd), env=env,
                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                 start_new_session=True)
        self.children[name] = child
        self.logs[name] = []
        async def read():
            while line := await child.stdout.readline():
                self.logs[name].append(line.decode(errors="replace").strip()[:400])
                self.logs[name] = self.logs[name][-20:]
        self.readers[name] = asyncio.create_task(read())
        return child

    async def run(self):
        try:
            self.error = None
            self.gaze_calibrated = False
            self.gaze_report = None
            if self.sensors.gaze_mode != "gazekit" or self.sensors.eeg_mode != "muse":
                raise ValueError("Live generation requires GOZ_GAZE=gazekit and GOZ_EEG=muse. Use GOZ_DEMO=1 for a rehearsal.")
            if self.sensors.gaze_error:
                raise ValueError(self.sensors.gaze_error)
            self.attempt_dir = self.directory / "calibration" / str(uuid4())
            self.attempt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.sensors.gaze.expected_setup_id = self.attempt_dir.name
            gaze_repo = Path(os.getenv("GOZ_GAZEKIT_DIR") or str(ROOT.parent / "gazekit")).expanduser().resolve()
            # Keep the venv executable path: resolving its symlink loses the venv.
            gaze_python = Path(os.path.abspath(os.path.expanduser(os.getenv("GOZ_GAZEKIT_PYTHON") or sys.executable)))
            if not (gaze_repo / "gazekit" / "calibrate.py").is_file():
                raise ValueError("Gazekit was not found. Set GOZ_GAZEKIT_DIR to its checkout.")
            if not gaze_python.is_file():
                raise ValueError("Gazekit Python was not found. Set GOZ_GAZEKIT_PYTHON to its interpreter.")
            self.set_phase("connecting_muse", "Turn on and wear Muse 2. Connecting over Bluetooth…")
            # Reuse an already-published Muse stream; don't take over its process.
            try:
                await self.wait_for(lambda: bool(self.sensors.eeg.device_id), 4)
            except ValueError:
                argv = [sys.executable, "-u", "-m", "muselsl", "stream", "--backend", "bleak", "--model", "legacy", "--lsltime", "--retries", "2"]
                for option, env_name in (("--address", "GOZ_MUSE_ADDRESS"), ("--name", "GOZ_MUSE_NAME")):
                    if os.getenv(env_name): argv += [option, os.environ[env_name]]
                bridge = await self.launch("muse", argv, ROOT)
                await self.wait_for(lambda: bool(self.sensors.eeg.device_id), 90, bridge)
            self.set_phase("calibrating_muse", "Muse connected. Sit still with eyes open for 60 seconds of clean EEG. Adjust forehead/ear contacts if signal quality is poor.")
            self.sensors.eeg.begin_calibration()
            await self.wait_for(lambda: bool(self.sensors.eeg.calibration), 240, self.children.get("muse"))
            (self.attempt_dir / "muse.json").write_text(json.dumps(self.sensors.eeg.calibration, indent=2))
            (self.attempt_dir / "muse.json").chmod(0o600)
            self.set_phase("calibrating_gaze", "Muse calibrated. Gazekit is opening: follow the targets, then press a key on its results screen.")
            worker = [gaze_python, "-u", "-m", "backend.gaze_worker"]
            camera = os.getenv("GOZ_GAZE_CAMERA", "0")
            model = self.attempt_dir / "gaze_model.pkl"
            report_path = self.attempt_dir / "gaze_ready.json"
            calibration = await self.launch("gaze_calibration", [*worker, "calibrate", "--repo", gaze_repo,
                                                    "--camera", camera, "--model", model, "--report", report_path], ROOT)
            try:
                code = await asyncio.wait_for(calibration.wait(), timeout=900)
            except TimeoutError:
                raise ValueError("Gaze calibration timed out. Follow all targets and dismiss the results screen; then retry.") from None
            await self.readers["gaze_calibration"]
            if code != 0 or not report_path.is_file() or not model.is_file():
                raise ValueError("Gaze calibration did not pass. " + self.log_tail(calibration))
            self.gaze_report = json.loads(report_path.read_text())
            if self.gaze_report.get("verdict") not in ("STABLE", "USABLE"):
                raise ValueError("Gaze calibration was poor. Retry in better lighting while facing the camera.")
            self.gaze_calibrated = True
            self.set_phase("starting_gaze", "Gaze calibrated. Complete Gazekit's quick alignment; waiting for valid gaze samples…")
            # Old samples from an unrelated external producer cannot unlock setup.
            self.sensors.gaze.samples.clear()
            self.sensors.gaze.last_rx = 0
            stream = await self.launch("gaze", [*worker, "stream", "--repo", gaze_repo, "--camera", camera,
                                      "--model", model, "--setup-id", self.attempt_dir.name,
                                      "--port", str(os.getenv("GOZ_GAZE_PORT", "5590"))], ROOT)
            await self.wait_for(lambda: any(s.get("valid") for s in self.sensors.gaze.window(time.time() - 1, time.time())), 90, stream)
            self.set_phase("ready", "Muse and gaze are calibrated and live. You can generate your video.")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = str(error)[:1000]
            self.set_phase("failed", "Sensor setup failed. Correct the issue, then retry setup.")
            await self.stop_children(keep_muse=True)

    async def stop_children(self, keep_muse=False):
        targets = {name: child for name, child in self.children.items()
                   if not (keep_muse and name == "muse" and child.returncode is None)}
        for child in targets.values():
            if child.returncode is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for child in targets.values():
            if child.returncode is None:
                try: await asyncio.wait_for(child.wait(), 3)
                except TimeoutError:
                    try: os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    await child.wait()
        readers = [self.readers[name] for name in targets]
        for reader in readers:
            if not reader.done(): reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        for name in targets:
            self.children.pop(name, None)
            self.readers.pop(name, None)

    async def retry(self):
        if self.task and not self.task.done():
            raise FalError("Sensor setup is already running.", 409)
        bridge = self.children.get("muse")
        if bridge is not None and bridge.returncode is not None:
            self.sensors.eeg.disconnected()
        await self.stop_children(keep_muse=True)
        self.sensors.eeg.reset_calibration()
        self.set_phase("waiting", "Retrying sensor setup…")
        await self.start()

    async def close(self):
        self.stopping = True
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.stop_children()
