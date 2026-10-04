"""Pure guided Muse calibration presentation; existing feed owns accumulation."""
from . import eeg_policy
from .cumulative_eeg_policy import compatibility_key

TARGET_SECONDS = 60.0
INSTRUCTIONS = (
    "Keep your eyes open in the same screen and viewing setup you will use for the clips. "
    "Sit comfortably with a relaxed jaw and breathe naturally. "
    "The timer counts unique clean signal time and pauses when signal is invalid."
)


def calibration_status(quality):
    """Do not create a baseline, advance timers, or reset running hardware."""
    q = quality if isinstance(quality, dict) else {}
    measured = q.get('cleanSeconds')
    clean = max(0., min(TARGET_SECONDS, measured)) if eeg_policy.finite(measured) else 0.
    result = dict(supported=q.get('source') == 'muse', ready=False,
        state='unavailable', cleanSeconds=clean, targetSeconds=TARGET_SECONDS,
        remainingCleanSeconds=TARGET_SECONDS-clean, progress=clean/TARGET_SECONDS,
        instructions=INSTRUCTIONS, protocol='eyes-open-screen-viewing-60-clean-seconds-v1',
        timing='unique clean seconds; invalid signal pauses progress',
        interpretation='Experimental physiological baseline; not a liking, focus or emotion classifier',
        startAction='start_or_recalibrate', resetPolicy='Explicit action for a different wearer or viewing setup; stable setup can reuse a valid baseline across generation runs',
        wearerChangeDetection='Unavailable; the user must identify wearer or setup changes')
    if not result['supported']:
        result.update(state='unsupported', reason='60-second guided calibration supports direct Muse EEG. Continue with gaze-only for this source.')
        return result
    baseline = q.get('calibration') if isinstance(q.get('calibration'), dict) else {}
    if clean >= TARGET_SECONDS and compatibility_key(q) and eeg_policy.quality_failure(q) is None:
        result.update(ready=True, state='ready', reason='A genuine 60-clean-second baseline is ready for this device and channel set.')
    elif q.get('live') is True and not q.get('qualityError') and clean < TARGET_SECONDS:
        result.update(state='collecting', reason='Collecting unique clean EEG time. Keep the same eyes-open viewing setup.')
    else:
        result.update(state='paused', reason=q.get('qualityError') or
            ('Complete a fresh 60-clean-second baseline for the current channels.' if baseline else
             'Waiting for live clean EEG; gaze-only remains available.'))
    return result
