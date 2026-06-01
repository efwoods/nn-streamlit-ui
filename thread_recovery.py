"""Thread refresh after gateway timeout during message send (testable, no Streamlit UI)."""

from __future__ import annotations

import requests

# Sentinel mirrored from studio_chat_app — avoid importing Streamlit app module in tests.
NEW_THREAD = "__new__"

# HTTP statuses where the origin may have finished but the proxy timed out (e.g. CF 524).
GATEWAY_TIMEOUT_STATUS_CODES = frozenset({502, 503, 504, 524})


def convert_lg_messages(lg_messages: list) -> list:
    """Map LangGraph {type:'human'|'ai'} messages to {role:'user'|'assistant'}."""
    result = []
    for msg in lg_messages or []:
        t = msg.get("type", "")
        if t in ("human", "ai"):
            result.append({
                "role": "user" if t == "human" else "assistant",
                "content": msg.get("content", ""),
                "id": msg.get("id"),
                "response_time_ms": None,
            })
    return result


def http_status_from_exception(exc: BaseException) -> int | None:
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code
    return None


def is_recoverable_send_failure(exc: BaseException) -> bool:
    """True when the server may have persisted the turn despite a client/proxy error."""
    status = http_status_from_exception(exc)
    if status is not None and status in GATEWAY_TIMEOUT_STATUS_CODES:
        return True
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(
        exc,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    return False


def resolve_thread_id_after_send_failure(
    pending_thread_id: str | None,
    threads: list,
    partial_result: dict | None,
) -> str | None:
    """Pick the backend thread to reload after a failed send."""
    if partial_result and partial_result.get("thread_id"):
        return partial_result["thread_id"]
    if pending_thread_id and pending_thread_id != NEW_THREAD:
        known = {t["thread_id"] for t in threads}
        if pending_thread_id in known:
            return pending_thread_id
    if threads:
        return threads[0]["thread_id"]
    return None


def apply_recovered_messages(
    thread_messages: dict,
    *,
    pending_thread_id: str | None,
    final_tid: str,
    messages: list,
) -> None:
    """Replace local thread cache with authoritative backend messages."""
    thread_messages[final_tid] = messages
    if (
        pending_thread_id
        and pending_thread_id in thread_messages
        and pending_thread_id != final_tid
    ):
        del thread_messages[pending_thread_id]
