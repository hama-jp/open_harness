"""Shared test configuration for v2 tests."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
