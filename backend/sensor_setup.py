"""Own sensor child processes and gate new generations on calibrated live data."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from uuid import uuid4
import shlex

from .fal_adapter import FalError
from .adaptive.eeg_calibration import calibration_status

ROOT = Path(__file__).resolve().parent.parent


class SensorSetup:
    def __init__(self, sensors, directory, demo=False, spawn=None):
        self.sensors = sensors
        data_path = Path(directory).expanduser()
        self.directory = (data_path if data_path.is_absolute() else ROOT / data_path).resolve()
        self.profile_id = os.getenv("GOZ_VIEWER_PROFILE", "default").strip() or "default"
        self.required = not demo and os.getenv("GOZ_REQUIRE_SENSORS", "0") == "1"
        self.enabled = not demo and (self.required or sensors.gaze_mode == "gazekit")
        self.phase = "waiting" if self.enabled else "demo" if demo else "disabled"
        self.message = "Preparing sensors…" if self.enabled else "Demo mode: real sensor calibration is not required." if demo else "Sensors optional: continue with available signals."
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
        self.selected_camera = None
        suffix = "" if self.profile_id == "default" else "-" + hashlib.sha256(self.profile_id.encode()).hexdigest()[:16]
        self.saved_gaze_path = self.directory / "calibration" / ("saved-gaze" + suffix + ".json")
        self.reusing_gaze = False
        self.calibration_state = "saved" if self.saved_gaze_path.is_file() else "required"
        self.calibration_error = None
        self.muse_task = None
        self.manual_muse_state = "disconnected"
        self.manual_muse_error = None
        self.manual_muse_generation = 0

    def saved_gaze(self):
        if not self.saved_gaze_path.exists():
            return None
        try:
            saved = json.loads(self.saved_gaze_path.read_text())
            model = (self.directory / saved["model"]).resolve()
            report = saved["report"]
            if (saved.get("version") not in (1, 2) or not model.is_relative_to(self.directory / "calibration")
                    or not model.is_file() or report.get("verdict") not in ("STABLE", "USABLE")
                    or not isinstance(report.get("camera"), str) or not report["camera"].isdigit()
                    or saved.get("profileId", "default") != self.profile_id
                    or hashlib.sha256(model.read_bytes()).hexdigest() != saved["sha256"]):
                raise ValueError("Invalid saved gaze model")
            return model, report
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise ValueError("Saved eye calibration is unavailable or damaged. Remove eye calibration to train a new one.") from None

    def save_gaze(self, model, report):
        saved = dict(version=2, profileId=self.profile_id, model=str(model.relative_to(self.directory)), report=report,
                     sha256=hashlib.sha256(model.read_bytes()).hexdigest(), savedAt=time.time())
        temporary = self.saved_gaze_path.with_suffix(".tmp")
        self.saved_gaze_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with temporary.open("w") as output:
            temporary.chmod(0o600)
            json.dump(saved, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.saved_gaze_path)
        self.calibration_state = "saved"

    def snapshot(self):
        eeg, gaze = self.sensors.eeg, self.sensors.gaze
        muse = eeg.status()
        quality = deepcopy(muse)
        quality["calibration"] = deepcopy(getattr(eeg,"calibration",None) or {})
        guided = calibration_status(quality)
        gaze_only = bool(getattr(self.sensors,"eeg_gaze_only",False))
        gaze_live = gaze.status()["live"] and any(s.get("valid") and s.get("face", True)
                        for s in gaze.window(time.time() - 2, time.time()))
        latest = eeg.latest()
        bridge = self.children.get("muse")
        muse_live = (muse["live"] and bool(latest and not latest[3]) and not muse["qualityError"]
                     and (bridge is None or bridge.returncode is None))
        stream = self.children.get("gaze")
        gaze_alive = stream is not None and stream.returncode is None
        ready = not self.required or (self.phase == "ready" and self.gaze_calibrated and gaze_live and gaze_alive
                                     and (gaze_only or muse_live and muse["calibrated"]
                                          and (muse.get("source") != "muse" or guided["ready"])))
        message = "Gaze-only selected; EEG pacing is disabled for new runs." if gaze_only else self.message
        if self.required and self.phase == "ready" and not ready:
            message = ("Gaze signal lost. Face the camera; retry gaze setup if needed." if gaze_only
                       else "Muse EEG: " + muse["qualityError"] if muse["qualityError"]
                       else f"Muse baseline incomplete: {muse['cleanSeconds']}/{muse['targetSeconds']} clean seconds." if not muse["calibrated"]
                       else "Muse samples stopped. Retry sensor setup to collect a fresh baseline." if not muse_live
                       else "Gaze signal lost. Face the camera; retry setup if the camera stopped.")
        return dict(required=self.required, enabled=self.enabled, eegMode=self.sensors.eeg_mode, generationReady=bool(ready), phase=self.phase,
                    message=message, error=self.error, canRetry=self.enabled and (self.phase == "failed" or self.phase == "ready" and (not gaze_live or self.required and not ready)),
                    gazeOnly=gaze_only, eegCalibration=dict(guided,gazeOnly=gaze_only),
                    museConnection=dict(state=self.manual_muse_state, error=self.manual_muse_error),
                    camera=dict(source=self.selected_camera),
                    muse=muse, gaze=dict(**gaze.status(), calibrated=self.gaze_calibrated,
                                        valid=gaze_live, report=self.gaze_report,
                                        savedCalibration=self.saved_gaze_path.is_file(), reusingCalibration=self.reusing_gaze,
                                        calibrationState=self.calibration_state, calibrationPath=str(self.saved_gaze_path),
                                        failureReason=self.calibration_error, profileId=self.profile_id,
                                        checkCommand=self.check_command(),
                                        checkGuidance="Stop the backend, then run the check command while following its targets. Add --recenter for a checked alignment correction; restart the backend afterwards. Changing viewer, camera, seating or display requires a fresh check/calibration."))

    def check_command(self):
        try:
            saved = self.saved_gaze()
        except ValueError:
            return None
        if not saved:
            return None
        model, _ = saved
        python = os.getenv("GOZ_GAZEKIT_PYTHON") or sys.executable
        repo = os.getenv("GOZ_GAZEKIT_DIR") or str(ROOT.parent / "gazekit")
        return "cd " + shlex.quote(str(ROOT)) + " && " + shlex.join([python, "-m", "backend.gaze_worker", "check", "--repo", repo, "--model", str(model), "--profile", self.profile_id])

    def require_ready(self):
        state = self.snapshot()
        if not state["generationReady"]:
            raise FalError("Sensor setup is not ready. " + (state["error"] or state["message"]), 409)

    def set_phase(self, phase, message):
        self.phase, self.message = phase, message

    async def start(self):
        if not self.enabled or self.task and not self.task.done():
            return
        self.task = asyncio.create_task(self.run())

    async def wait_for(self, predicate, timeout, child=None):
        deadline = time.monotonic() + timeout
        while not predicate():
            if self.phase in ("connecting_muse", "calibrating_muse") and self.sensors.eeg.state.startswith("Muse connection error:"):
                raise ValueError(self.sensors.eeg.state)
            if child and child.returncode is not None:
                raise ValueError("Sensor process stopped. " + self.log_tail(child))
            if time.monotonic() > deadline:
                detail = ""
                if self.phase in ("connecting_muse", "calibrating_muse"):
                    eeg = self.sensors.eeg.status()
                    detail = (f" EEG: {eeg['qualityError'] or eeg['connectionState']}. "
                              f"Baseline: {eeg['cleanSeconds']}/{eeg['targetSeconds']} clean seconds.")
                raise ValueError("Sensor setup timed out. " + self.message + detail)
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
                value = line.decode(errors="replace").strip()
                if name == "gaze_calibration" and value == "GOZ_CAMERA_WINDOW ready":
                    self.message = "Gazekit’s camera window is open. Select iPhone Camera (Continuity Camera). Keep the phone locked with its rear cameras facing you. Press R to refresh. Muse starts after gaze calibration."
                if name in ("gaze_calibration", "gaze") and value.startswith("GOZ_CAMERA "):
                    try:
                        camera = json.loads(value[len("GOZ_CAMERA "):])["camera"]
                        if isinstance(camera, str) and camera.isdigit():
                            self.selected_camera = camera
                            if name == "gaze_calibration":
                                self.set_phase("calibrating_gaze", "Camera selected. Follow Gazekit's calibration targets and dismiss its results screen. Your eye calibration will be saved automatically.")
                    except (ValueError, KeyError, TypeError):
                        pass
                if name == "gaze_calibration" and value.startswith("GOZ_CALIBRATION_SAVED "):
                    try:
                        report = json.loads(value[len("GOZ_CALIBRATION_SAVED "):])
                        if report.get("verdict") in ("STABLE", "USABLE"):
                            self.gaze_report = report
                            self.save_gaze(self.attempt_dir / "gaze_model.pkl", report)
                            self.message = "Eye calibration saved. Dismiss Gazekit's results screen to start tracking."
                    except (OSError, ValueError, TypeError) as error:
                        self.calibration_error = f"Could not register saved calibration: {error}"
                if name == "gaze_calibration" and value.startswith("GOZ_CALIBRATION_REJECTED "):
                    try:
                        self.gaze_report = json.loads(value[len("GOZ_CALIBRATION_REJECTED "):])
                        self.calibration_state = "invalid"
                        self.calibration_error = f"Fresh validation failed: mean error {self.gaze_report.get('mean_error_px')} screen points. Dismiss the results screen and retry in steadier lighting/seating."
                        self.message = self.calibration_error
                    except (ValueError, TypeError):
                        pass
                self.logs[name].append(value[:400])
                self.logs[name] = self.logs[name][-20:]
        self.readers[name] = asyncio.create_task(read())
        return child

    async def run(self):
        try:
            self.error = None
            self.gaze_calibrated = False
            self.gaze_report = None
            self.reusing_gaze = False
            self.calibration_error = None
            if self.sensors.gaze_mode != "gazekit" or self.required and self.sensors.eeg_mode not in ("muse", "mindmonitor"):
                raise ValueError("Live generation requires GOZ_GAZE=gazekit and GOZ_EEG=mindmonitor (or muse). Use GOZ_DEMO=1 for a rehearsal.")
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
            worker = [gaze_python, "-u", "-m", "backend.gaze_worker"]
            model = self.attempt_dir / "gaze_model.pkl"
            report_path = self.attempt_dir / "gaze_ready.json"
            saved = self.saved_gaze()
            if saved:
                model, self.gaze_report = saved
                self.reusing_gaze = True
            else:
                self.calibration_state = "calibrating"
                self.set_phase("select_camera", "Gazekit is opening automatically. Select your iPhone Camera in its native window, then follow the calibration targets. Keep the iPhone locked, with its rear cameras facing you. Your eye calibration will be saved for future starts.")
                requested_camera = os.getenv("GOZ_GAZE_CAMERA", "select")
                if requested_camera.isdigit():
                    self.selected_camera = requested_camera
                    self.set_phase("calibrating_gaze", "Opening the configured camera. Follow Gazekit's targets, keeping camera and seating fixed; dismiss the results screen when finished.")
                calibration = await self.launch("gaze_calibration", [*worker, "calibrate", "--repo", gaze_repo,
                                                    "--camera", requested_camera, "--model", model, "--report", report_path,
                                                    "--profile", self.profile_id], ROOT)
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
            camera = self.gaze_report.get("camera")
            if not isinstance(camera, str) or not camera.isdigit():
                raise ValueError("Gazekit did not report the selected camera. Retry calibration.")
            self.selected_camera = camera
            if not saved:
                self.save_gaze(model, self.gaze_report)
            self.gaze_calibrated = True
            self.set_phase("starting_gaze", "Loading the saved model in a fresh process and checking viewer/camera/display compatibility…" if saved
                           else "Eye calibration saved. Loading the validated model and waiting for live gaze…")
            # Old samples from an unrelated external producer cannot unlock setup.
            self.sensors.gaze.samples.clear()
            self.sensors.gaze.last_rx = 0
            camera_args = []
            if saved:
                camera_args.append("--reuse-calibration")
            for option, field in (("--camera-device-id", "cameraDeviceId"), ("--camera-name", "cameraName")):
                if self.gaze_report.get(field):
                    camera_args += [option, self.gaze_report[field]]
            stream = await self.launch("gaze", [*worker, "stream", "--repo", gaze_repo, "--camera", camera, *camera_args,
                                      "--model", model, "--setup-id", self.attempt_dir.name,
                                      "--profile", self.profile_id,
                                      "--port", str(os.getenv("GOZ_GAZE_PORT", "5590"))], ROOT)
            await self.wait_for(lambda: any(s.get("valid") for s in self.sensors.gaze.window(time.time() - 1, time.time())), 90, stream)
            self.calibration_state = "loaded"
            if self.sensors.eeg_mode == "mindmonitor":
                self.sensors.start_muse_reader()
            if not self.required:
                self.set_phase("ready", "Gaze is ready. Connect Muse when you want EEG observations; playback and gaze work without it.")
                return
            if self.sensors.eeg_mode == "muse":
                self.set_phase("ready", "Gaze is ready. Connect Muse to enable generation; playback and gaze setup remain independent.")
                return
            if self.sensors.eeg_mode == "mindmonitor":
                status = self.sensors.eeg.status()
                self.set_phase("connecting_muse", f"Connect Muse 2 in Mind Monitor on your phone. On the same Wi-Fi, set OSC destination to {status['oscDestination']}, UDP port {status['oscPort']}, and enable OSC Stream Brainwaves (All Values or Average Only). Waiting for 10 readings with at least one good contact…")
                await self.wait_for(lambda: self.sensors.eeg.status()["calibrated"], 240)
            else:
                self.set_phase("connecting_muse", "Turn on and wear Muse 2. Connecting over Bluetooth…")
                # Reuse an already-published Muse stream; don't take over its process.
                try:
                    await self.wait_for(lambda: bool(self.sensors.eeg.device_id), 4)
                except ValueError:
                    argv = [sys.executable, "-u", "-m", "muselsl", "stream", "--backend", "bleak", "--model", "legacy", "--lsltime", "--retries", "2"]
                    for option, env_name in (("--address", "GOZ_MUSE_ADDRESS"), ("--name", "GOZ_MUSE_NAME")):
                        if os.getenv(env_name): argv += [option, os.environ[env_name]]
                    bridge = self.children.get("muse")
                    if bridge is None or bridge.returncode is not None:
                        bridge = await self.launch("muse", argv, ROOT)
                    await self.wait_for(lambda: bool(self.sensors.eeg.device_id), 90, bridge)
                self.set_phase("calibrating_muse", "Muse LSL outlet found. Waiting for actual samples and 60 seconds of clean EEG. Sit still with eyes open and adjust forehead/ear contacts if signal quality is poor.")
                self.sensors.eeg.begin_calibration()
                await self.wait_for(lambda: self.sensors.eeg.status()["calibrated"], 240, self.children.get("muse"))
            (self.attempt_dir / "muse.json").write_text(json.dumps(self.sensors.eeg.calibration, indent=2))
            (self.attempt_dir / "muse.json").chmod(0o600)
            self.set_phase("ready", "Muse and gaze are ready and live. You can generate your video.")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = str(error)[:1000]
            self.calibration_error = self.error
            self.calibration_state = "invalid" if self.saved_gaze_path.exists() else "required"
            self.set_phase("failed", "Sensor setup failed. Correct the issue, then retry setup.")
            await self.stop_children(keep_muse=True)

    async def connect_muse(self):
        """Start Muse discovery only after an explicit user request."""
        if self.sensors.eeg_mode != "muse":
            raise FalError("Muse controls require GOZ_EEG=muse.", 409)
        if self.manual_muse_state in ("connecting", "connected"):
            return
        self.manual_muse_generation += 1
        generation = self.manual_muse_generation
        self.manual_muse_error = None
        self.manual_muse_state = "connecting"
        self.sensors.start_muse_reader()
        self.muse_task = asyncio.create_task(self._connect_muse(generation))

    async def _connect_muse(self, generation):
        try:
            try:
                await self.wait_for(lambda: bool(self.sensors.eeg.device_id), 4)
            except ValueError:
                argv = [sys.executable, "-u", "-m", "muselsl", "stream", "--backend", "bleak", "--model", "legacy", "--lsltime", "--retries", "2"]
                for option, env_name in (("--address", "GOZ_MUSE_ADDRESS"), ("--name", "GOZ_MUSE_NAME")):
                    if os.getenv(env_name): argv += [option, os.environ[env_name]]
                bridge = self.children.get("muse")
                if bridge is None or bridge.returncode is not None:
                    bridge = await self.launch("muse", argv, ROOT)
                await self.wait_for(lambda: bool(self.sensors.eeg.device_id), 90, bridge)
            if generation != self.manual_muse_generation:
                return
            self.sensors.eeg.begin_calibration()
            await self.wait_for(lambda: self.sensors.eeg.status()["calibrated"], 240, self.children.get("muse"))
            if generation != self.manual_muse_generation:
                return
            if self.attempt_dir:
                (self.attempt_dir / "muse.json").write_text(json.dumps(self.sensors.eeg.calibration, indent=2))
                (self.attempt_dir / "muse.json").chmod(0o600)
            self.set_phase("ready", "Muse and gaze are ready. You can generate your video.")
            self.manual_muse_state = "connected"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if generation != self.manual_muse_generation:
                return
            self.manual_muse_state = "error"
            self.manual_muse_error = str(error)[:500]
            self.sensors.eeg.state = f"Muse connection error: {error}. Gaze-only mode remains available."
            self.sensors.eeg.connection_error = self.sensors.eeg.state

    async def disconnect_muse(self):
        """Stop only the bridge process launched and owned by this setup."""
        self.manual_muse_generation += 1
        if self.muse_task and not self.muse_task.done():
            self.muse_task.cancel()
            await asyncio.gather(self.muse_task, return_exceptions=True)
        bridge = self.children.get("muse")
        if bridge is not None and bridge.returncode is None:
            try: os.killpg(bridge.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: await asyncio.wait_for(bridge.wait(), 3)
            except TimeoutError:
                try: os.killpg(bridge.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                await bridge.wait()
            reader = self.readers.pop("muse", None)
            if reader and not reader.done():
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            self.children.pop("muse", None)
        self.sensors.eeg.disconnected()
        self.manual_muse_state = "disconnected"
        self.manual_muse_error = None

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
        if self.muse_task and not self.muse_task.done():
            self.muse_task.cancel()
            await asyncio.gather(self.muse_task, return_exceptions=True)
        bridge = self.children.get("muse")
        if bridge is not None and bridge.returncode is not None:
            self.sensors.eeg.disconnected()
        await self.stop_children(keep_muse=True)
        self.sensors.eeg.reset_calibration()
        self.selected_camera = None
        self.set_phase("waiting", "Retrying sensor setup…")
        await self.start()

    async def check_gaze(self, recenter=False):
        """Pause the owned camera stream, validate fresh targets, then reload it."""
        if self.task and not self.task.done():
            raise FalError("Sensor setup is already running.", 409)
        try:
            saved = self.saved_gaze()
        except ValueError as error:
            raise FalError(str(error), 409) from None
        if not saved:
            raise FalError("Calibrate this viewer before running the gaze check.", 409)
        async def perform():
            model, _ = saved
            await self.stop_children(keep_muse=True)
            self.sensors.gaze.samples.clear()
            self.sensors.gaze.last_rx = 0
            self.sensors.gaze.expected_setup_id = "checking-" + str(uuid4())
            self.gaze_calibrated = False
            self.calibration_state = "checking"
            self.set_phase("checking_gaze", "Follow the independent gaze-check targets. The previous model/alignment remains saved until this check passes.")
            try:
                argv = [os.getenv("GOZ_GAZEKIT_PYTHON") or sys.executable, "-u", "-m", "backend.gaze_worker", "check", "--repo",
                        os.getenv("GOZ_GAZEKIT_DIR") or str(ROOT.parent / "gazekit"), "--model", model, "--profile", self.profile_id]
                if recenter: argv.append("--recenter")
                child = await self.launch("gaze_check", argv, ROOT)
                code = await asyncio.wait_for(child.wait(), 300)
                await self.readers["gaze_check"]
                if code != 0:
                    raise ValueError(self.log_tail(child))
                self.save_gaze(model, json.loads(model.with_suffix(".report.json").read_text()))
                await self.run()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.error = self.calibration_error = str(error)[:1000]
                self.calibration_state = "invalid"
                self.set_phase("failed", "Gaze check failed; previous model preserved. Recalibrate for the current viewer/camera/seating.")
                await self.stop_children(keep_muse=True)
        self.task = asyncio.create_task(perform())

    async def close(self):
        self.stopping = True
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.muse_task and not self.muse_task.done():
            self.muse_task.cancel()
            await asyncio.gather(self.muse_task, return_exceptions=True)
        await self.stop_children()

    async def remove_gaze_calibration(self):
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.stop_children(keep_muse=True)
        self.saved_gaze_path.unlink(missing_ok=True)
        self.gaze_calibrated = False
        self.reusing_gaze = False
        self.gaze_report = None
        self.calibration_state = "required"
        self.calibration_error = None
        self.sensors.gaze.samples.clear()
        self.sensors.gaze.last_rx = 0
        self.sensors.gaze.expected_setup_id = "removed-" + str(uuid4())
        self.sensors.eeg.reset_calibration()
        self.selected_camera = None
        self.set_phase("waiting", "Eye calibration removed. Starting a fresh calibration…")
        await self.start()
