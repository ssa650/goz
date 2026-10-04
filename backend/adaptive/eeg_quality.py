"""Raw EEG validity and causal preprocessing, independent of policy/UI.

MuseLSL decodes 12-bit ADC counts as .48828125 * (count - 2048) µV.
The existing conservative 950 µV hard gate is retained, even after filtering.
SciPy SOS state is carried across chunks, never re-filtered per feature window:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
https://github.com/alexandrebarachant/muse-lsl/blob/master/muselsl/muse.py
These are engineering quality gates, not validated emotion classifiers.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

CHANNELS = ("TP9", "AF7", "AF8", "TP10")
RATE = 256
MIN_CHANNELS = 2
HARD_CLIP_UV = 950.0
ADC_RAIL_UV = 999.5
MAX_INTERVAL_S = 1.5 / RATE
REASONS = {
    "raw_clip": "clipped sensor channel",
    "raw_flat": "flat sensor channel",
    "band_flat": "flat band-limited sensor channel",
    "band_amplitude": "movement or poor contact",
    "line_noise": "50/60 Hz interference",
    "sample_gap": "EEG packets missing or irregular",
    "invalid_input": "Invalid EEG samples",
    "insufficient_channels": "fewer than two clean EEG channels",
    "metadata": "Muse EEG labels, units or identity could not be verified",
}


class CausalEegFilter:
    """Fourth-order Butterworth 1–40 Hz, eight poles; no notch or lookahead."""

    def __init__(self):
        self.sos = butter(4, [1, 40], btype="bandpass", fs=RATE, output="sos")
        self.reset()

    def reset(self):
        self.zi = None

    def process(self, samples):
        x = np.asarray(samples, dtype=float)
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def channel_diagnostics(raw, filtered, stamps, amplitude_uv):
    """Report raw evidence before any cleanup; select channels only afterward."""
    raw, filtered, stamps = map(np.asarray, (raw, filtered, stamps))
    intervals = np.diff(stamps)
    gaps = intervals > MAX_INTERVAL_S
    irregular = (intervals < .5 / RATE) | (intervals <= 0)
    span = float(stamps[-1] - stamps[0])
    timing_invalid = bool(gaps.any() or irregular.any() or not 1.8 <= span <= 2.2)
    gap_count = int(gaps.sum())
    missing = int(np.maximum(np.rint(intervals[gaps] * RATE) - 1, 1).sum())
    gap_seconds = float(np.maximum(intervals[gaps] - 1 / RATE, 0).sum())
    spectrum = np.abs(np.fft.rfft((raw - raw.mean(axis=0)) * np.hanning(len(raw))[:, None], axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(raw), 1 / RATE)
    line_mask = ((frequencies >= 49) & (frequencies <= 51)) | ((frequencies >= 59) & (frequencies <= 61))
    line = spectrum[line_mask].sum(axis=0) / np.maximum(spectrum[frequencies >= 1].sum(axis=0), 1e-9)
    output = {}
    for i, name in enumerate(CHANNELS):
        x, y = raw[:, i], filtered[:, i]
        raw_std, filtered_p2p = float(x.std()), float(np.ptp(y))
        reasons = []
        if np.abs(x).max() >= HARD_CLIP_UV: reasons.append("raw_clip")
        if raw_std < .05: reasons.append("raw_flat")
        if y.std() < .05: reasons.append("band_flat")
        if filtered_p2p > amplitude_uv: reasons.append("band_amplitude")
        if line[i] > .35: reasons.append("line_noise")
        if timing_invalid: reasons.append("sample_gap")
        output[name] = dict(
            # Existing amplitude keys retain their raw meaning.
            peakToPeakUV=round(float(np.ptp(x)), 3), stdUV=round(raw_std, 3),
            peakAbsUV=round(float(np.abs(x).max()), 3),
            rawMinUV=round(float(x.min()), 3), rawMaxUV=round(float(x.max()), 3),
            rawOffsetUV=round(float(x.mean()), 3),
            railFraction=round(float((np.abs(x) >= ADC_RAIL_UV).mean()), 6),
            hardClipFraction=round(float((np.abs(x) >= HARD_CLIP_UV).mean()), 6),
            flat=bool(raw_std < .05),
            unchangedFraction=round(float((np.abs(np.diff(x)) < .01).mean()), 6),
            filteredPeakToPeakUV=round(filtered_p2p, 3), filteredStdUV=round(float(y.std()), 3),
            lineNoiseFraction=round(float(line[i]), 4),
            gapCount=gap_count, irregularIntervals=int(irregular.sum()),
            windowSpanSeconds=round(span, 6), missingSamples=missing, gapSeconds=round(gap_seconds, 6),
            maxIntervalSeconds=round(float(intervals.max()), 6),
            rejectReasons=reasons, usable=not reasons)
    return output


class UniqueCleanTime:
    """Union of accepted intervals, with no credit for an initial warmup window."""

    def __init__(self):
        self.seconds = 0.0
        self.end = None

    def add(self, start, end):
        if self.end is not None:
            self.seconds += max(0.0, end - max(start, self.end))
        self.end = max(end, self.end) if self.end is not None else end
        return self.seconds

    def pause(self):
        """Keep accepted duration; require a fresh full window before more credit."""
        self.end = None


def muse_metadata(info, identity):
    """Validate full inlet XML. Test adapters without XML state their assumption."""
    result = dict(sourceId=identity, sampleRate=RATE, units="microvolts",
                  channelLabels=list(CHANNELS), excludedLabels=["Right AUX"],
                  verification="assumed_muselsl", packetIdentity="LSL source_id and strictly increasing sample timestamp",
                  hardwarePacketCounterAvailable=False, valid=True)
    if not hasattr(info, "as_xml"):
        return result
    try:
        root = ET.fromstring(info.as_xml())
        channels = root.findall("./desc/channels/channel")
        labels = [c.findtext("label", "") for c in channels]
        units = [c.findtext("unit", "") for c in channels]
        identity_ok = root.findtext("source_id") == identity and bool(identity)
        result.update(channelLabels=labels[:4], excludedLabels=labels[4:],
                      declaredUnits=units, verification="verified",
                      valid=bool(identity_ok and labels[:4] == list(CHANNELS) and len(labels) == 5
                          and "aux" in labels[4].lower() and all(
                              unit.lower().replace("μ", "u").replace("µ", "u") in
                              ("uv", "microvolts", "microvolt") for unit in units[:4])))
        if not result["valid"]:
            result["verification"] = "rejected"
    except (ET.ParseError, ValueError, TypeError):
        result.update(valid=False, verification="rejected")
    return result
