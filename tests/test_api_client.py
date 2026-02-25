"""Tests for lib.api_client — base HTTP client with retry, rate limiting, caching.

RED phase: these tests define the contract for BaseAPIClient before
the module exists. They must FAIL (ImportError) on first run.

The BaseAPIClient should provide:
- GET requests with automatic JSON/text parsing
- Exponential backoff retry on 429 and 5xx errors
- Configurable rate limiting (minimum delay between requests)
- Cache integration via lib.cache.ResponseCache
- Context manager protocol
- Request/cache statistics
- Configurable timeout, headers, User-Agent
"""

import time

import httpx
import pytest

from lib.api_client import BaseAPIClient


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_client(handler, **kwargs):
    """Create a BaseAPIClient with a mock HTTP transport.

    The handler receives an httpx.Request and returns an httpx.Response.
    This avoids real network calls while testing retry/backoff logic.
    """
    transport = httpx.MockTransport(handler)
    return BaseAPIClient(transport=transport, **kwargs)


def json_handler(request):
    """Always returns 200 with a JSON body."""
    return httpx.Response(200, json={"result": "ok"})


def text_handler(request):
    """Always returns 200 with a text body."""
    return httpx.Response(200, text="hello world")


# ── Basic request tests ─────────────────────────────────────────────────────


class TestGetJson:
    """GET requests that return parsed JSON."""

    def test_basic_json_response(self):
        client = make_client(json_handler, rate_limit_delay=0)
        result = client.get_json("https://api.example.com/data")
        assert result == {"result": "ok"}
        client.close()

    def test_json_with_params(self):
        def handler(request):
            assert b"page=2" in bytes(str(request.url), "utf-8")
            return httpx.Response(200, json={"page": 2})

        client = make_client(handler, rate_limit_delay=0)
        result = client.get_json("https://api.example.com/data", params={"page": 2})
        assert result == {"page": 2}
        client.close()

    def test_base_url_prepended(self):
        def handler(request):
            assert "api.example.com/v1/data" in str(request.url)
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler, base_url="https://api.example.com/v1", rate_limit_delay=0)
        result = client.get_json("/data")
        assert result == {"ok": True}
        client.close()

    def test_absolute_url_ignores_base(self):
        def handler(request):
            assert "other.com" in str(request.url)
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler, base_url="https://api.example.com", rate_limit_delay=0)
        result = client.get_json("https://other.com/data")
        assert result == {"ok": True}
        client.close()


class TestGetText:
    """GET requests that return text."""

    def test_basic_text_response(self):
        client = make_client(text_handler, rate_limit_delay=0)
        result = client.get_text("https://api.example.com/page")
        assert result == "hello world"
        client.close()


# ── Retry tests ──────────────────────────────────────────────────────────────


class TestRetry:
    """Retry logic with exponential backoff."""

    def test_retries_on_500(self):
        """Server error (500) should trigger retry."""
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(500)
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler, max_retries=3, rate_limit_delay=0, backoff_base=0.01)
        result = client.get_json("https://api.example.com/data")
        assert result == {"ok": True}
        assert len(attempts) == 3
        client.close()

    def test_retries_on_429(self):
        """Rate limit (429) should trigger retry."""
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) < 2:
                return httpx.Response(429)
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler, max_retries=3, rate_limit_delay=0, backoff_base=0.01)
        result = client.get_json("https://api.example.com/data")
        assert result == {"ok": True}
        assert len(attempts) == 2
        client.close()

    def test_retries_on_502_503_504(self):
        """Gateway errors should trigger retry."""
        for status in [502, 503, 504]:
            attempts = []

            def handler(request, _attempts=attempts):
                _attempts.append(1)
                if len(_attempts) < 2:
                    return httpx.Response(status)
                return httpx.Response(200, json={"ok": True})

            client = make_client(handler, max_retries=3, rate_limit_delay=0, backoff_base=0.01)
            result = client.get_json("https://api.example.com/data")
            assert result == {"ok": True}
            client.close()

    def test_no_retry_on_400(self):
        """Client error (400) should NOT retry — raise immediately."""
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(400, text="Bad Request")

        client = make_client(handler, max_retries=3, rate_limit_delay=0, backoff_base=0.01)
        with pytest.raises(httpx.HTTPStatusError):
            client.get_json("https://api.example.com/data")
        assert len(attempts) == 1  # no retries
        client.close()

    def test_no_retry_on_404(self):
        """Not found (404) should NOT retry."""
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(404, text="Not Found")

        client = make_client(handler, max_retries=3, rate_limit_delay=0, backoff_base=0.01)
        with pytest.raises(httpx.HTTPStatusError):
            client.get_json("https://api.example.com/data")
        assert len(attempts) == 1
        client.close()

    def test_raises_after_max_retries_exhausted(self):
        """After exhausting retries, should raise the error."""
        def handler(request):
            return httpx.Response(500)

        client = make_client(handler, max_retries=2, rate_limit_delay=0, backoff_base=0.01)
        with pytest.raises(httpx.HTTPStatusError):
            client.get_json("https://api.example.com/data")
        client.close()

    def test_custom_retry_status_codes(self):
        """Custom retry_on_status set should be respected."""
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) < 2:
                return httpx.Response(418)  # I'm a teapot
            return httpx.Response(200, json={"ok": True})

        client = make_client(
            handler, max_retries=3, rate_limit_delay=0,
            backoff_base=0.01, retry_on_status={418},
        )
        result = client.get_json("https://api.example.com/data")
        assert result == {"ok": True}
        assert len(attempts) == 2
        client.close()

    def test_backoff_increases_with_attempts(self, monkeypatch):
        """Backoff delay should increase exponentially."""
        delays = []
        original_sleep = time.sleep

        def mock_sleep(seconds):
            delays.append(seconds)

        monkeypatch.setattr(time, "sleep", mock_sleep)

        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) <= 3:
                return httpx.Response(500)
            return httpx.Response(200, json={"ok": True})

        # rate_limit_delay=0 so only backoff sleeps are captured
        client = make_client(handler, max_retries=4, rate_limit_delay=0, backoff_base=1.0)
        client.get_json("https://api.example.com/data")
        # Should have 3 backoff sleeps: 1.0, 2.0, 4.0
        assert len(delays) == 3
        assert delays[0] == pytest.approx(1.0)
        assert delays[1] == pytest.approx(2.0)
        assert delays[2] == pytest.approx(4.0)
        client.close()


# ── Rate limiting tests ──────────────────────────────────────────────────────


class TestRateLimiting:
    """Pre-request rate limiting."""

    def test_enforces_minimum_delay(self, monkeypatch):
        """Requests should be spaced by at least rate_limit_delay."""
        current_time = [100.0]
        sleep_calls = []

        def mock_time():
            return current_time[0]

        def mock_sleep(seconds):
            sleep_calls.append(seconds)
            current_time[0] += seconds

        monkeypatch.setattr(time, "time", mock_time)
        monkeypatch.setattr(time, "sleep", mock_sleep)

        client = make_client(json_handler, rate_limit_delay=1.0)

        # First request — no delay needed
        client.get_json("https://api.example.com/a")
        initial_sleeps = len(sleep_calls)

        # Second request immediately — should sleep
        client.get_json("https://api.example.com/b")
        assert len(sleep_calls) > initial_sleeps
        # The sleep duration should be close to rate_limit_delay
        last_sleep = sleep_calls[-1]
        assert last_sleep > 0
        assert last_sleep <= 1.0
        client.close()

    def test_no_delay_when_enough_time_passed(self, monkeypatch):
        """No sleep when enough time has already elapsed."""
        current_time = [100.0]
        sleep_calls = []

        def mock_time():
            return current_time[0]

        def mock_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(time, "time", mock_time)
        monkeypatch.setattr(time, "sleep", mock_sleep)

        client = make_client(json_handler, rate_limit_delay=1.0)

        client.get_json("https://api.example.com/a")
        # Advance time well past the delay
        current_time[0] += 5.0
        initial_sleeps = len(sleep_calls)
        client.get_json("https://api.example.com/b")
        # No new rate-limit sleep should have been needed
        # (there may be zero sleeps total if first request also didn't need one)
        rate_limit_sleeps = len(sleep_calls) - initial_sleeps
        assert rate_limit_sleeps == 0
        client.close()

    def test_zero_delay_skips_rate_limiting(self):
        """rate_limit_delay=0 should not sleep."""
        client = make_client(json_handler, rate_limit_delay=0)
        # Should not raise or sleep
        client.get_json("https://api.example.com/a")
        client.get_json("https://api.example.com/b")
        client.close()


# ── Cache integration tests ──────────────────────────────────────────────────


class TestCacheIntegration:
    """Cache hit/miss behavior."""

    def test_cache_miss_makes_request(self, tmp_path):
        """On cache miss, request is made and result is cached."""
        from lib.cache import ResponseCache

        db = tmp_path / "cache.db"
        request_count = []

        def handler(request):
            request_count.append(1)
            return httpx.Response(200, json={"data": 42})

        cache = ResponseCache(db_path=db, ttl=3600)
        client = make_client(handler, cache=cache, rate_limit_delay=0)
        result = client.get_json("https://api.example.com/data")
        assert result == {"data": 42}
        assert len(request_count) == 1
        client.close()

    def test_cache_hit_skips_request(self, tmp_path):
        """On cache hit, no HTTP request is made."""
        from lib.cache import ResponseCache

        db = tmp_path / "cache.db"
        request_count = []

        def handler(request):
            request_count.append(1)
            return httpx.Response(200, json={"data": 42})

        cache = ResponseCache(db_path=db, ttl=3600)
        client = make_client(handler, cache=cache, rate_limit_delay=0)

        # First call — cache miss
        client.get_json("https://api.example.com/data")
        assert len(request_count) == 1

        # Second call — cache hit
        result = client.get_json("https://api.example.com/data")
        assert result == {"data": 42}
        assert len(request_count) == 1  # no new request
        client.close()

    def test_use_cache_false_bypasses_cache(self, tmp_path):
        """use_cache=False should always make a request."""
        from lib.cache import ResponseCache

        db = tmp_path / "cache.db"
        request_count = []

        def handler(request):
            request_count.append(1)
            return httpx.Response(200, json={"data": 42})

        cache = ResponseCache(db_path=db, ttl=3600)
        client = make_client(handler, cache=cache, rate_limit_delay=0)

        client.get_json("https://api.example.com/data", use_cache=False)
        client.get_json("https://api.example.com/data", use_cache=False)
        assert len(request_count) == 2
        client.close()

    def test_no_cache_still_works(self):
        """Client without cache should work fine."""
        client = make_client(json_handler, rate_limit_delay=0)
        result = client.get_json("https://api.example.com/data")
        assert result == {"result": "ok"}
        client.close()

    def test_different_params_different_cache_keys(self, tmp_path):
        """Different query params should produce different cache entries."""
        from lib.cache import ResponseCache

        db = tmp_path / "cache.db"
        call_count = []

        def handler(request):
            call_count.append(1)
            page = dict(request.url.params).get("page", "1")
            return httpx.Response(200, json={"page": int(page)})

        cache = ResponseCache(db_path=db, ttl=3600)
        client = make_client(handler, cache=cache, rate_limit_delay=0)

        r1 = client.get_json("https://api.example.com/data", params={"page": 1})
        r2 = client.get_json("https://api.example.com/data", params={"page": 2})
        assert r1 == {"page": 1}
        assert r2 == {"page": 2}
        assert len(call_count) == 2  # two different requests
        client.close()


# ── Context manager tests ────────────────────────────────────────────────────


class TestContextManager:
    """Context manager protocol."""

    def test_works_as_context_manager(self):
        with make_client(json_handler, rate_limit_delay=0) as client:
            result = client.get_json("https://api.example.com/data")
            assert result == {"result": "ok"}

    def test_close_is_idempotent(self):
        client = make_client(json_handler, rate_limit_delay=0)
        client.close()
        client.close()  # should not raise


# ── Stats tests ──────────────────────────────────────────────────────────────


class TestStats:
    """Request and cache statistics."""

    def test_request_count_increments(self):
        client = make_client(json_handler, rate_limit_delay=0)
        assert client.stats["requests"] == 0
        client.get_json("https://api.example.com/a")
        assert client.stats["requests"] == 1
        client.get_json("https://api.example.com/b")
        assert client.stats["requests"] == 2
        client.close()

    def test_cache_hit_count(self, tmp_path):
        from lib.cache import ResponseCache

        db = tmp_path / "cache.db"
        cache = ResponseCache(db_path=db, ttl=3600)
        client = make_client(json_handler, cache=cache, rate_limit_delay=0)

        client.get_json("https://api.example.com/data")
        assert client.stats["cache_hits"] == 0

        client.get_json("https://api.example.com/data")
        assert client.stats["cache_hits"] == 1
        client.close()


# ── Configuration tests ──────────────────────────────────────────────────────


class TestConfiguration:
    """Configurable parameters."""

    def test_custom_user_agent(self):
        def handler(request):
            assert request.headers["user-agent"] == "MyTool/1.0"
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler, user_agent="MyTool/1.0", rate_limit_delay=0)
        client.get_json("https://api.example.com/data")
        client.close()

    def test_custom_headers(self):
        def handler(request):
            assert request.headers["x-custom"] == "value"
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler, headers={"X-Custom": "value"}, rate_limit_delay=0)
        client.get_json("https://api.example.com/data")
        client.close()

    def test_default_user_agent(self):
        def handler(request):
            ua = request.headers.get("user-agent", "")
            assert "ResearchTools" in ua
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler, rate_limit_delay=0)
        client.get_json("https://api.example.com/data")
        client.close()

    def test_default_timeout(self):
        """Default timeout should be reasonable (30-120s)."""
        client = make_client(json_handler, rate_limit_delay=0)
        assert client.timeout >= 30
        client.close()

    def test_default_max_retries(self):
        """Default max retries should be 3."""
        client = make_client(json_handler, rate_limit_delay=0)
        assert client.max_retries == 3
        client.close()
