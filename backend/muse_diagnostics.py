"""Small private Muse lifecycle journal; never record EEG sample arrays."""
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from .provider_errors import safe_text


def producer_observation(text):
    """Only explicit producer statements, never a guessed OS/BLE root cause."""
    value = text.lower().strip()
    if 'no muses found' in value:
        return 'discovery_empty'
    if value == 'disconnected.' or 'auto-disconnect:' in value:
        return 'stream_disconnected'
    if 'failed to connect to muse' in value:
        return 'connection_failed'
    if 'ble connected.' in value:
        return 'ble_connected'
    if value.startswith('streaming eeg'):
        return 'stream_started'
    return None


class MuseDiagnostics:
    MAX_BYTES = 128 * 1024
    MAX_ROW_BYTES = 16 * 1024
    MAX_SUMMARY_ROW_BYTES = 1536
    SUMMARY_EVENTS = frozenset(("connect_requested", "disconnect_requested", "producer_started",
                               "producer_exit", "producer_stop_requested", "sample_progress", "interruption",
                               "reader_state", "baseline_invalidated", "recovery_needed", "recovery_backoff",
                               "recovery_blocked", "recovery_restored", "producer_observation",
                               "supervisor_error", "supervisor_stopped", "external_outlet_reused"))

    def __init__(self, directory):
        self.path = Path(directory) / 'diagnostics' / 'muse.jsonl'
        self.lock = threading.RLock()
        self.events = deque(maxlen=80)
        self.summary_path = self.path.with_name('muse-state.json')
        self.summary = {}
        self.write_error = None
        self.secrets = tuple(value for key, value in os.environ.items()
                             if any(word in key.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))
                             and len(value) >= 4)
        try:
            if self.summary_path.stat().st_size <= 32 * 1024:
                previous = json.loads(self.summary_path.read_text())
                if isinstance(previous, dict):
                    self.summary = {key: self.safe_value(row) for key, row in previous.items()
                                    if key in self.SUMMARY_EVENTS and isinstance(row, dict)}
        except (OSError, ValueError):
            pass

    def safe_text(self, value):
        text = safe_text(str(value), self.secrets, limit=2048)
        text = re.sub(r'(?i)(?:[0-9a-f]{2}:){5}[0-9a-f]{2}', '[device]', text)
        text = re.sub(r'(?i)\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b', '[device]', text)
        # Numerical arrays can contain raw sensor output; omit them entirely.
        text = re.sub(r'\[[^\]]*\]', '[data omitted]', text)
        text = re.sub(r'(?:[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[\s,;]+){3,}[-+]?\d+(?:\.\d+)?',
                      '[numeric data omitted]', text)
        return ''.join(char for char in text if char >= ' ' or char == '\t')[:400]

    def safe_value(self, value, depth=0):
        if isinstance(value, str): return self.safe_text(value)
        if value is None or isinstance(value, bool): return value
        if isinstance(value, (int, float)):
            try: return value if math.isfinite(value) else None
            except OverflowError: return None
        if isinstance(value, dict) and depth < 3:
            return {key: self.safe_value(item, depth + 1) for key, item in list(value.items())[:24]
                    if isinstance(key, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', key)}
        return '[data omitted]'

    @staticmethod
    def bounded_row(row, limit):
        if len(json.dumps(row, allow_nan=False).encode()) <= limit:
            return row
        # Retain lifecycle scalars even for a pathological nested diagnostic.
        keys = ('at', 'timestamp', 'event', 'lastSampleAt', 'code', 'pid', 'reason', 'attempt',
                'generation', 'samplesReceived', 'seconds', 'nextRetryAt', 'ownership', 'state', 'connectIntent')
        compact = {key: row[key][:80] if isinstance(row[key], str) else row[key]
                   for key in keys if key in row and not isinstance(row[key], (dict, list))}
        compact['detailOmitted'] = True
        while len(json.dumps(compact, allow_nan=False).encode()) > limit:
            key = next((key for key in reversed(keys) if key in compact), None)
            if key is None: break
            compact.pop(key)
        return compact

    def record(self, event, **fields):
        with self.lock:
            now = time.time()
            row = dict(at=now, timestamp=datetime.fromtimestamp(now, timezone.utc).isoformat(), event=self.safe_text(event)[:64])
            row.update({key: self.safe_value(value) for key, value in list(fields.items())[:24]
                        if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', key)})
            row = self.bounded_row(row, self.MAX_ROW_BYTES)
            self.events.append(row)
            encoded = (json.dumps(row, allow_nan=False) + '\n').encode()
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if self.path.exists() and self.path.stat().st_size + len(encoded) > self.MAX_BYTES:
                    self.path.replace(self.path.with_suffix('.jsonl.1'))
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, 'ab') as output:
                    os.fchmod(output.fileno(), 0o600)
                    output.write(encoded)
                    output.flush()
                    if event != 'producer_output': os.fsync(output.fileno())
                # Keep critical latest events independently of noisy output
                # rotation. This bounded summary survives a backend restart.
                if event in self.SUMMARY_EVENTS:
                    self.summary[event] = self.bounded_row(row, self.MAX_SUMMARY_ROW_BYTES)
                    temporary = self.summary_path.with_suffix('.tmp')
                    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, 'w') as output:
                        os.fchmod(output.fileno(), 0o600)
                        json.dump(self.summary, output, allow_nan=False)
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.replace(self.summary_path)
                self.write_error = None
            except OSError as error:
                self.write_error = self.safe_text(error)

    def snapshot(self):
        with self.lock:
            return dict(path=str(self.path), summaryPath=str(self.summary_path), writeError=self.write_error,
                        latestEvents=dict(self.summary), events=list(self.events))
