"""Bounded provider diagnostics. Never retain responses, URLs or headers."""
import json
import os
from pathlib import Path
import re
import threading
import time

DETAIL_LIMIT = 400
STAGES = frozenset(('upload', 'submit', 'status', 'result', 'cancel', 'unknown'))
CATEGORIES = frozenset(('account_locked_top_up', 'account_locked', 'authentication', 'permission',
                        'rate_limit', 'invalid_request', 'not_found', 'provider_failure',
                        'transport_error', 'malformed_response', 'unknown'))
RETRYABILITY = frozenset(('same_request_read', 'manual_review', 'none'))


def safe_text(value, secrets=(), limit=DETAIL_LIMIT):
    """Redact before truncating, including credentials not loaded in this process."""
    if not isinstance(value, str):
        return ''
    text = value
    known = tuple(secrets) + tuple(v for k, v in os.environ.items()
                                  if any(s in k.upper() for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))
                                  and len(v) >= 4)
    for secret in sorted((s for s in known if isinstance(s, str) and s), key=len, reverse=True):
        text = text.replace(secret, '[redacted]')
    text = re.sub(r'(?i)\b(?:authorization|proxy-authorization|set-cookie|cookie|x-api-key|x-auth-token)\s*[=:]\s*[^\r\n]+', '[credentials omitted]', text)
    text = re.sub(r'(?i)\b(?:bearer|key|basic)\s+[A-Za-z0-9._~+/=:-]+', '[credentials omitted]', text)
    text = re.sub(r'(?i)[\"\']?(?:api[_-]?key|key|access[_-]?token|refresh[_-]?token|token|password|secret)[\"\']?\s*[=:]\s*(?:\"[^\"]*\"|\'[^\']*\'|[^\s,;}]+)', '[credentials omitted]', text)
    text = re.sub(r'(?i)\b(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>\"\']+', '[url omitted]', text)
    text = re.sub(r'(?i)\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?(?:/[^\s<>\"\']*)?', '[url omitted]', text)
    text = re.sub(r'\b(?:sk-|hf_)[A-Za-z0-9_-]+|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+', '[redacted]', text)
    # Unknown opaque credentials are not useful diagnostic prose.
    text = re.sub(r'[A-Za-z0-9_+/=-]{48,}', '[opaque data omitted]', text)
    text = ''.join(c if c.isprintable() else ' ' for c in text)
    return ' '.join(text.split())[:limit]


def safe_id(value, secrets=()):
    if (not isinstance(value, str) or len(value) > 128 or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or
            safe_text(value, secrets, 128) != value):
        return None
    return value


def selected_detail(body):
    """Read only diagnostic leaves; validation inputs and arbitrary data are excluded."""
    leaves = []
    def visit(value, depth=0):
        if len(leaves) >= 8 or depth > 3:
            return
        if isinstance(value, str):
            leaves.append(value)
        elif isinstance(value, dict):
            for key in ('detail', 'message', 'error', 'code', 'type', 'error_type', 'category', 'msg'):
                if key in value:
                    visit(value[key], depth + 1)
        elif isinstance(value, list):
            for item in value[:4]:
                visit(item, depth + 1)
    visit(body)
    return '; '.join(leaves)


def diagnostic(detail='', *, status=None, request_id=None, stage='unknown', category=None, secrets=()):
    status = status if type(status) is int and 100 <= status <= 599 else None
    stage = stage if isinstance(stage, str) and stage in STAGES else 'unknown'
    detail = safe_text(detail, secrets) or 'Provider did not return a safe diagnostic detail.'
    normalized = detail.upper()
    if not isinstance(category, str) or category not in CATEGORIES:
        category = ('account_locked_top_up' if 'TOP_UP' in normalized and 'LOCK' in normalized else
                    'account_locked' if 'USER IS LOCKED' in normalized else
                    'authentication' if status == 401 else 'permission' if status == 403 else
                    'rate_limit' if status == 429 else 'not_found' if status == 404 else
                    'invalid_request' if status in (400, 422) else
                    'provider_failure' if status and status >= 500 else 'unknown')
    retry = ('same_request_read' if stage in ('status', 'result') and category in
             ('transport_error', 'rate_limit', 'provider_failure') else
             'none' if category == 'invalid_request' else 'manual_review')
    return dict(provider='fal', httpStatus=status, requestId=safe_id(request_id, secrets),
                category=category, detail=detail, stage=stage, retryability=retry, automaticRetry=False)


def safe_diagnostic(value, secrets=(), request_id=None):
    if not isinstance(value, dict):
        return None
    result = diagnostic(value.get('detail', ''), status=value.get('httpStatus'),
                        request_id=request_id or value.get('requestId'), stage=value.get('stage'),
                        category=value.get('category'), secrets=secrets)
    return result


def user_message(info):
    prefix = f"Fal {info['stage']} failed"
    if info.get('httpStatus') is not None:
        prefix += f" (HTTP {info['httpStatus']})"
    message = prefix + ': ' + info['detail']
    if info['category'] in ('account_locked_top_up', 'account_locked'):
        message += ' Review this request and account status in Fal before choosing another generation; the account balance cause is unverified.'
    elif info['retryability'] == 'same_request_read':
        message += ' Inspect the saved request; only reading the same request is eligible to reconnect.'
    else:
        message += ' Inspect the saved request before choosing another generation.'
    if info.get('requestId'):
        message += ' Request: ' + info['requestId'] + '.'
    return message + ' No new generation was automatically submitted.'


class ProviderFailureJournal:
    """One bounded private file and one rotated file; safe fields only."""
    MAX_BYTES = 64 * 1024

    def __init__(self, directory):
        self.path = Path(directory) / 'diagnostics' / 'provider-errors.jsonl'
        self.lock = threading.RLock()
        self.write_error = None

    def record(self, info, *, job_id=None, session_id=None, secrets=()):
        info = safe_diagnostic(info, secrets)
        if info is None:
            return
        row = dict(at=time.time(), event='provider_failed', providerError=info,
                   jobId=safe_id(job_id, secrets), sessionId=safe_id(session_id, secrets))
        encoded = (json.dumps(row, allow_nan=False) + '\n').encode()
        with self.lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if self.path.exists() and self.path.stat().st_size + len(encoded) > self.MAX_BYTES:
                    self.path.replace(self.path.with_suffix('.jsonl.1'))
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, 'ab') as output:
                    os.fchmod(output.fileno(), 0o600)
                    output.write(encoded)
                    output.flush()
                    os.fsync(output.fileno())
                self.write_error = None
            except OSError as error:
                self.write_error = safe_text(str(error), secrets)
