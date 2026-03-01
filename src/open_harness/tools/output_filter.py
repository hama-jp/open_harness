"""Output filter — redact sensitive data before sending tool output to LLM."""

from __future__ import annotations

import re

# (pattern, replacement) pairs applied in order
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Generic key=value secrets
    (re.compile(
        r'(?i)(api[_-]?key|secret|token|password|passwd|credentials?)\s*[=:]\s*\S+'),
     r'\1=***REDACTED***'),
    # AWS credentials
    (re.compile(
        r'(?i)(AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|AWS_SESSION_TOKEN)\s*=\s*\S+'),
     r'\1=***REDACTED***'),
    # GitHub personal access tokens
    (re.compile(r'ghp_[A-Za-z0-9_]{36,}'), '***GITHUB_PAT***'),
    # GitHub OAuth tokens
    (re.compile(r'gho_[A-Za-z0-9_]{36,}'), '***GITHUB_TOKEN***'),
    # OpenAI-style API keys (sk-...)
    (re.compile(r'sk-[A-Za-z0-9]{20,}'), '***API_KEY***'),
    # Bearer tokens in headers
    (re.compile(r'(?i)bearer\s+[A-Za-z0-9._\-]{20,}'), 'Bearer ***REDACTED***'),
    # Generic hex/base64 secrets after common variable names
    (re.compile(
        r'(?i)(PRIVATE_KEY|SECRET_KEY|DATABASE_URL|CONNECTION_STRING)\s*=\s*\S+'),
     r'\1=***REDACTED***'),
]


def redact_secrets(text: str) -> str:
    """Replace known secret patterns in *text* with redacted placeholders.

    Designed to be fast for typical tool outputs (short-circuit on empty text,
    compiled patterns).
    """
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
