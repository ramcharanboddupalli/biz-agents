"""Optional, privacy-conscious OpenTelemetry tracing for BizAgents workflows."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from time import perf_counter
from typing import Protocol
from urllib.parse import urlsplit

_LOGGER = logging.getLogger(__name__)
_TRACING_ENABLED = "BIZAGENTS_TRACING_ENABLED"


class Span(Protocol):
    def set_attribute(self, key: str, value: bool | float | int | str) -> None:
        ...


class Tracer(Protocol):
    def start_as_current_span(
        self,
        name: str,
        *,
        record_exception: bool = False,
        set_status_on_exception: bool = False,
    ) -> AbstractContextManager[Span]:
        ...


class SpanHandle:
    """Best-effort attribute recording that never changes workflow behavior."""

    def __init__(self, span: Span | None) -> None:
        self._span = span

    def set_attributes(self, attributes: dict[str, bool | float | int | str]) -> None:
        if self._span is None:
            return
        for key, value in attributes.items():
            try:
                self._span.set_attribute(key, value)
            except Exception as exc:
                _log_tracing_failure("record span metadata", exc)


def configured_tracer() -> Tracer | None:
    """Create a Phoenix tracer only when explicitly enabled by configuration."""
    enabled_value = os.getenv(_TRACING_ENABLED, "false").strip().casefold()
    if enabled_value not in {"1", "true", "yes", "on"}:
        if enabled_value not in {"", "0", "false", "no", "off"}:
            _LOGGER.warning(
                "BizAgents tracing is disabled because %s is not a recognized boolean.",
                _TRACING_ENABLED,
            )
        return None

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    if not endpoint:
        _LOGGER.warning(
            "BizAgents tracing is enabled but PHOENIX_COLLECTOR_ENDPOINT is empty; "
            "tracing is disabled."
        )
        return None
    if not _valid_collector_endpoint(endpoint):
        _LOGGER.warning(
            "BizAgents tracing is disabled because PHOENIX_COLLECTOR_ENDPOINT is invalid."
        )
        return None

    try:
        from phoenix.otel import register

        provider = register(
            endpoint=endpoint,
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "").strip() or "bizagents",
            api_key=os.getenv("PHOENIX_API_KEY") or None,
            batch=True,
            auto_instrument=False,
            verbose=False,
        )
        return provider.get_tracer("bizagents")
    except Exception as exc:
        _log_tracing_failure("initialize Phoenix", exc)
        return None


@contextmanager
def traced_span(tracer: Tracer | None, name: str) -> Generator[SpanHandle, None, None]:
    """Create a span when possible and isolate all tracing failures."""
    if tracer is None:
        yield SpanHandle(None)
        return

    started = perf_counter()
    try:
        span_context = tracer.start_as_current_span(
            name,
            record_exception=False,
            set_status_on_exception=False,
        )
        span = span_context.__enter__()
    except Exception as exc:
        _log_tracing_failure("start span", exc)
        yield SpanHandle(None)
        return

    handle = SpanHandle(span)
    active_exception: BaseException | None = None
    try:
        yield handle
    except BaseException as exc:
        active_exception = exc
        handle.set_attributes(
            {
                "bizagents.error": True,
                "error.type": "unhandled_exception",
            }
        )
        raise
    finally:
        handle.set_attributes(
            {"bizagents.duration_ms": round((perf_counter() - started) * 1000, 3)}
        )
        try:
            if active_exception is None:
                span_context.__exit__(None, None, None)
            else:
                span_context.__exit__(
                    type(active_exception),
                    active_exception,
                    active_exception.__traceback__,
                )
        except Exception as exc:
            _log_tracing_failure("finish span", exc)


def _log_tracing_failure(operation: str, exc: Exception) -> None:
    _LOGGER.warning(
        "Phoenix tracing could not %s (%s); continuing without trace details.",
        operation,
        type(exc).__name__,
    )


def _valid_collector_endpoint(endpoint: str) -> bool:
    if not endpoint or any(character.isspace() for character in endpoint):
        return False
    try:
        parsed = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
        parsed.port
        return (
            parsed.scheme.casefold() in {"", "http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False
