"""Tests for Issue 9: LLM Retry/Backoff."""

import json
from unittest.mock import MagicMock, patch

import httpx

from open_harness.llm.client import LLMClient, LLMResponse, _MAX_RETRIES, _BACKOFF_BASE
from open_harness.config import ProviderConfig


def _make_client() -> LLMClient:
    provider = ProviderConfig(base_url="http://localhost:1234/v1")
    return LLMClient(provider, timeout=5)


class TestChatRetry:
    def test_retries_on_429(self):
        client = _make_client()
        call_count = 0

        def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count < 3:
                resp.status_code = 429
                resp.raise_for_status = MagicMock()
            else:
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {},
                }
            return resp

        with patch.object(client.client, "post", side_effect=mock_post), \
             patch("open_harness.llm.client.time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}], "test-model")
        assert result.content == "ok"
        assert call_count == 3

    def test_retries_on_500(self):
        client = _make_client()
        call_count = 0

        def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count < 2:
                resp.status_code = 500
                resp.raise_for_status = MagicMock()
            else:
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {
                    "choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}],
                    "usage": {},
                }
            return resp

        with patch.object(client.client, "post", side_effect=mock_post), \
             patch("open_harness.llm.client.time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}], "test-model")
        assert result.content == "recovered"

    def test_no_retry_on_400(self):
        client = _make_client()

        exc = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=MagicMock(status_code=400)
        )

        with patch.object(client.client, "post", side_effect=exc):
            result = client.chat([{"role": "user", "content": "hi"}], "test-model")
        assert result.finish_reason == "error"

    def test_retries_on_timeout(self):
        client = _make_client()
        call_count = 0

        def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ReadTimeout("timed out")
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }
            return resp

        with patch.object(client.client, "post", side_effect=mock_post), \
             patch("open_harness.llm.client.time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}], "test-model")
        assert result.content == "ok"

    def test_exhausted_retries(self):
        client = _make_client()

        def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 503
            resp.raise_for_status = MagicMock()
            return resp

        with patch.object(client.client, "post", side_effect=mock_post), \
             patch("open_harness.llm.client.time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}], "test-model")
        assert result.finish_reason == "error"
        assert "exhausted" in result.content.lower() or "Error" in result.content


class TestConstants:
    def test_retry_count(self):
        assert _MAX_RETRIES == 3

    def test_backoff_base(self):
        assert _BACKOFF_BASE == 1
