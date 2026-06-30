"""
Studio Assistant Chat UI
-------------------------
A Streamlit chat interface for ``POST /message/{assistant_id}`` with **SSE**
streaming (``stream=true``): assistant tokens are shown as they arrive.
Run with:
    streamlit run studio_chat_app.py
"""
import base64
import json
import io
import uuid
import requests
from datetime import datetime
from urllib.parse import urlencode, urlparse, urlunparse
import streamlit as st
import streamlit.components.v1 as components
import os
from collections.abc import Callable
from dotenv import load_dotenv
from PIL import Image, ImageOps

from thread_recovery import (
    apply_recovered_messages,
    convert_lg_messages,
    is_recoverable_send_failure,
    resolve_thread_id_after_send_failure,
)

load_dotenv()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
AUTO_GREETING = "Hey! Please tell me about yourself and what you can do for me."
# Sentinel for a conversation that hasn't been created on the backend yet
NEW_THREAD = "__new__"

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Studio Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# URL query params: assistant_id, optional api_key, optional thread_id
#   Example:  http://localhost:8501/?assistant_id=abc&thread_id=…
#   Use "Share Avatar" in the sidebar to copy the current link.
# ──────────────────────────────────────────────
assistant_id: str = st.query_params.get("assistant_id", "").strip()
api_key: str= st.query_params.get("api_key", "").strip()

# ──────────────────────────────────────────────
# Session-state bootstrap
# ──────────────────────────────────────────────
DEFAULTS = {
    "base_url": os.getenv("NN_API_BASE_URL", ""),
    "api_key": api_key,
    "user_name": "",
    "user_description": "",
    # Resolved user identity (anonymous or authenticated)
    "user_id": None,
    # Threads from backend (api_key mode only)
    "backend_threads": [],      # list of raw thread dicts from /conversations
    "threads_loaded": False,
    "last_loaded_api_key": None,  # detect api_key changes → reload
    # thread_messages: { thread_id -> [{"role", "content", "response_time_ms"}] }
    "thread_messages": {},
    # Currently open thread id  ("__new__" until backend confirms a thread_id)
    "active_thread_id": None,
    # Local title overrides – written to backend on next message send
    "conversation_titles": {},  # { thread_id -> str }
    # UI
    "show_settings": False,
    "pending_auto_message": None,
    # Increment to reset st.file_uploader after a successful attachment send
    "attachment_uploader_bump": 0,
    # Full data URI or https URL for assistant avatar (from GET /avatar_reference_image)
    "assistant_reference_icon": None,
    "_ref_icon_cache_key": None,
    # Bounded retries so a not-yet-ready reference image is refetched, not pinned to 🤖
    "_ref_icon_attempts": 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────

def _base() -> str:
    return st.session_state.base_url.rstrip("/")

def _headers(json_body: bool = False) -> dict:
    h: dict = {"api-key": st.session_state.api_key}
    if json_body:
        h["Content-Type"] = "multipart/form-data"
    return h

def fetch_user_id() -> str:
    """GET /get_current_user_id — works for anonymous users (no api_key needed)."""
    resp = requests.get(f"{_base()}/get_current_user_id", headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()

def fetch_threads(asst_id: str) -> list:
    """GET /conversations?assistant_id=<id>  →  list of thread dicts sorted newest-first."""
    resp = requests.get(
        f"{_base()}/conversations",
        headers=_headers(),
        params={"assistant_id": asst_id},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

def fetch_thread_messages(thread_id: str, assistant_id: str) -> list:
    """GET /conversations/{thread_id}/messages  →  list of LangGraph message dicts."""
    resp = requests.get(
        f"{_base()}/conversations/{thread_id}/messages",
        headers=_headers(),
        params={"assistant_id":assistant_id},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("messages", [])

def fetch_avatar_reference_icon() -> str | None:
    """GET /avatar_reference_image — returns data:image…;base64,… or an https URL string.

    This endpoint is public per ``assistant_id``; do not send an empty ``api-key`` header
    or anonymous callers may get rejected. Include ``api-key`` only when configured.
    """
    headers: dict[str, str] = {}
    ak = (st.session_state.api_key or "").strip()
    if ak:
        headers["api-key"] = ak
    resp = requests.get(
        f"{_base()}/avatar_reference_image",
        headers=headers,
        params={"assistant_id": assistant_id},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("reference_image_data")
    return data if data else None


def upload_avatar_reference_image(uploaded_file) -> None:
    """POST /update_avatar_identity_with_media with reference_image=true."""
    mime = uploaded_file.type or "image/jpeg"
    body, ct = _chat_image_bytes_and_mime(uploaded_file.getvalue(), mime)
    ct_use = ct if ct.startswith("image/") else mime
    files = [
        (
            "files",
            (uploaded_file.name, body, ct_use),
        )
    ]
    data = {
        "assistant_id": assistant_id,
        "reference_image": "true",
        "reference_audio": "false",
    }
    resp = requests.post(
        f"{_base()}/update_avatar_identity_with_media",
        headers=_headers(),
        files=files,
        data=data,
        timeout=None,
    )
    resp.raise_for_status()


def upload_avatar_reference_image_url(image_url: str) -> None:
    data = {
        "assistant_id": assistant_id,
        "reference_image": "true",
        "reference_audio": "false",
        "url": image_url.strip(),
    }
    resp = requests.post(
        f"{_base()}/update_avatar_identity_with_media",
        headers=_headers(),
        data=data,
        timeout=None,
    )
    resp.raise_for_status()


def _accumulate_sse_message_obj(obj: dict, streamed_parts: list[str], merged: dict) -> bool:
    """Merge one SSE JSON object into streamed_parts / merged. Returns True if a token was appended."""
    hit = False
    if obj.get("type") == "assistant_token":
        t = obj.get("text")
        if t is not None:
            streamed_parts.append(str(t))
            hit = True
    # A human-in-the-loop pause (e.g. correct_identity_fact) arrives as an
    # ``interrupt`` event carrying the approve/edit/reject preview instead of ``done``.
    if obj.get("type") == "interrupt" and obj.get("interrupt") is not None:
        merged["interrupt"] = obj["interrupt"]
    for k in ("content", "thread_id", "total_response_time_ms", "response_metadata"):
        if k in obj and obj[k] is not None:
            merged[k] = obj[k]
    return hit


# Field names of the per-document fact-correction analysis (``ProposedFactEdit``). When the
# backend leaks these structured-output calls into the ``assistant_token`` stream it runs
# several concurrently, so the JSON arrives interleaved — the buffer can even *start* with a
# prose fragment from inside one object's string value. Matching any of these keys anywhere
# in the accumulated text identifies the turn as a correction regardless of interleaving.
_INTERNAL_JSON_MARKERS = (
    '"asserts_inaccurate_fact"',
    '"corrected_text"',
    '"corrected_context"',
)


def _streamed_text_is_internal_json(text: str) -> bool:
    """True when live-streamed text is leaked structured output, not a prose reply.

    During a human-in-the-loop fact correction the backend currently streams the
    per-document analysis (``ProposedFactEdit``: ``asserts_inaccurate_fact`` /
    ``corrected_text`` / ``corrected_context``) through the same ``assistant_token``
    channel as a normal reply, and runs several of those analyses concurrently — so the
    JSON arrives interleaved/garbled. None of that is a user-facing reply, so it should
    never be rendered directly. A real avatar reply is natural language; a leaked chunk
    either begins with ``{`` / ``[`` or carries one of the correction schema keys.
    """
    if (text or "").lstrip()[:1] in ("{", "["):
        return True
    return any(marker in (text or "") for marker in _INTERNAL_JSON_MARKERS)


def _finalize_merged_message(streamed_parts: list[str], merged: dict) -> dict:
    text = "".join(streamed_parts)
    if text and not merged.get("content"):
        merged["content"] = text
    merged.setdefault("content", "")
    return merged


def _parse_message_sse_body(body: str) -> dict:
    """Turn POST /message SSE (data: … lines) into the JSON shape the UI expects."""
    streamed_parts: list[str] = []
    merged: dict = {}
    for raw_line in (body or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].lstrip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        _accumulate_sse_message_obj(obj, streamed_parts, merged)
    return _finalize_merged_message(streamed_parts, merged)


def consume_streaming_message_response(
    resp: requests.Response,
    on_partial_text: Callable[[str], None] | None = None,
) -> dict:
    """Read a POST /message SSE body incrementally; optional callback after each assistant_token.

    Does not close ``resp``; the caller must close the response (e.g. in a ``finally`` block).
    """
    streamed_parts: list[str] = []
    merged: dict = {}
    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].lstrip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        had_token = _accumulate_sse_message_obj(obj, streamed_parts, merged)
        if had_token and on_partial_text is not None:
            on_partial_text("".join(streamed_parts))
    return _finalize_merged_message(streamed_parts, merged)


def api_send_message(
    user_message: str,
    thread_id: str | None = None,
    attachments: list | None = None,
    on_stream_chunk: Callable[[str], None] | None = None,
) -> dict:
    """POST /message/{assistant_id} with ``stream=true`` (SSE) by default.

    Uses HTTP streaming so tokens can be shown as they arrive via ``on_stream_chunk``.
    Returns the same merged dict shape as before: ``content``, ``thread_id``,
    ``total_response_time_ms``, ``response_metadata``.

    With attachments, sends multipart/form-data (same ``files`` field pattern as
    ``update_avatar_identity_with_media``).
    """
    real_thread_id = None if (thread_id is None or thread_id == NEW_THREAD) else thread_id
    title = st.session_state.conversation_titles.get(thread_id) if thread_id else None
    payload = {
        "message": user_message,
        "your_name": st.session_state.user_name or None,
        "your_description": st.session_state.user_description or None,
        "conversation_title": title,
        "thread_id": real_thread_id,
        "stream": True,
        # Browser's IANA timezone (e.g. "America/New_York") so the backend can
        # localize the system clock injected into the prompt to the user's location.
        "user_timezone": getattr(st.context, "timezone", None),
    }
    url = f"{_base()}/message/{assistant_id}"
    if attachments:
        files = [
            (
                "files",
                (
                    uf.name,
                    uf.getvalue(),
                    uf.type or "application/octet-stream",
                ),
            )
            for uf in attachments
        ]
        resp = requests.post(
            url,
            headers=_headers(),
            data={k: v for k, v in payload.items() if v is not None},
            files=files,
            stream=True,
            timeout=None,
        )
    else:
        resp = requests.post(
            url,
            headers=_headers(),
            data=payload,
            stream=True,
            timeout=None,
        )
    resp.raise_for_status()
    ct = (resp.headers.get("Content-Type") or "").lower()
    try:
        if "event-stream" in ct:
            return consume_streaming_message_response(resp, on_stream_chunk)
        if "application/json" in ct:
            return resp.json()
        return _parse_message_sse_body(resp.text or "")
    finally:
        resp.close()


def api_resume_message(
    thread_id: str,
    decision: str,
    *,
    items: list[dict] | None = None,
    on_stream_chunk: Callable[[str], None] | None = None,
) -> dict:
    """POST /message/{assistant_id}/resume — continue a run paused for approval.

    ``decision`` is ``apply`` | ``cancel``. ``items`` carries the owner's per-document
    decisions (one entry per matched document: ``index`` + an ``action`` ∈
    ``accept`` ("Accept Edit") | ``remove`` ("Remove the Document") | ``skip`` ("Leave the
    document unchanged"), plus ``corrected_text`` / ``correction_context`` — the editable
    window applied on ``accept``). Any document the owner leaves alone defaults to ``skip``
    server-side, so it is never silently changed. Returns the same merged dict shape as
    ``api_send_message`` (and may itself carry another ``interrupt``).
    """
    payload = {
        "thread_id": thread_id,
        "decision": decision,
        "your_name": st.session_state.user_name or None,
        "your_description": st.session_state.user_description or None,
        "user_timezone": getattr(st.context, "timezone", None),
    }
    if items is not None:
        payload["items"] = json.dumps(items)

    resp = requests.post(
        f"{_base()}/message/{assistant_id}/resume",
        headers=_headers(),
        data={k: v for k, v in payload.items() if v is not None},
        stream=True,
        timeout=None,
    )
    resp.raise_for_status()
    ct = (resp.headers.get("Content-Type") or "").lower()
    try:
        if "event-stream" in ct:
            return consume_streaming_message_response(resp, on_stream_chunk)
        if "application/json" in ct:
            return resp.json()
        return _parse_message_sse_body(resp.text or "")
    finally:
        resp.close()


class _BytesAttachment:
    """Stand-in for Streamlit UploadedFile when resending attachments after st.rerun()."""

    __slots__ = ("name", "type", "_data")

    def __init__(self, name: str, mime: str, data: bytes):
        self.name = name
        self.type = mime
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _preview_attachment_bar(uploaded_file) -> None:
    """Inline preview for the composer (image thumbnail or file caption)."""
    if uploaded_file.type and uploaded_file.type.startswith("image/"):
        raw = uploaded_file.getvalue()
        pil = _pil_image_exif_normalized(raw)
        st.image(pil if pil is not None else io.BytesIO(raw), width=min(220, 320))
    else:
        size_kb = len(uploaded_file.getbuffer()) / 1024.0
        st.caption(f"📎 **{uploaded_file.name}** — {size_kb:.1f} KB")


def _render_user_chat_content(msg: dict) -> None:
    """User bubble body: optional attachment preview + text (same shape as stored messages)."""
    att = msg.get("attachment_meta")
    text = (msg.get("content") or "").strip()
    if att:
        mime = att.get("mime") or ""
        name = att.get("name", "attachment")
        if mime.startswith("image/") and att.get("bytes"):
            b = att["bytes"]
            pil = _pil_image_exif_normalized(b)
            st.image(pil if pil is not None else io.BytesIO(b), width=min(280, 360))
        elif mime.startswith("image/"):
            st.caption(f"🖼️ {name}")
        else:
            st.caption(f"📎 {name}")
    if text:
        st.markdown(text)
    elif att:
        st.caption("_Attachment only._")

# ──────────────────────────────────────────────
# Conversion / display helpers
# ──────────────────────────────────────────────

def _pil_image_exif_normalized(data: bytes) -> Image.Image | None:
    """Decode image bytes and apply EXIF Orientation so pixels match the camera preview."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return ImageOps.exif_transpose(img)
    except Exception:
        return None


def _chat_image_bytes_and_mime(data: bytes, mime: str | None) -> tuple[bytes, str]:
    """
    Re-encode image with EXIF orientation baked into pixels.

    Many viewers (and model APIs) ignore EXIF; camera JPEGs then look rotated.
    """
    img = _pil_image_exif_normalized(data)
    if img is None:
        return data, (mime or "application/octet-stream")
    mime_l = (mime or "").lower()
    fmt = (img.format or "").upper()
    buf = io.BytesIO()
    try:
        if fmt == "JPEG" or "jpeg" in mime_l or "jpg" in mime_l:
            img.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
            return buf.getvalue(), "image/jpeg"
        if fmt == "WEBP" or "webp" in mime_l:
            img.save(buf, format="WEBP", quality=90)
            return buf.getvalue(), "image/webp"
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"
    except Exception:
        return data, (mime or "application/octet-stream")


def _json_export_sanitize(obj):
    """Deep-copy structures for JSON export; encode binary blobs as base64 text."""
    if isinstance(obj, bytes):
        return base64.standard_b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {k: _json_export_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_export_sanitize(v) for v in obj]
    return obj


def refresh_thread_after_send_failure(
    pending_thread_id: str | None,
    *,
    partial_result: dict | None = None,
) -> tuple[str, list] | None:
    """Re-fetch conversations + messages so UI matches backend after a timeout.

    Returns ``(thread_id, ui_messages)`` or ``None`` if refresh failed.
    """
    if not (st.session_state.api_key or "").strip():
        return None
    try:
        threads = fetch_threads(assistant_id)
        st.session_state.backend_threads = threads
        st.session_state.threads_loaded = True

        tid = resolve_thread_id_after_send_failure(
            pending_thread_id, threads, partial_result
        )
        if not tid:
            return None

        raw = fetch_thread_messages(tid, assistant_id)
        return tid, convert_lg_messages(raw)
    except Exception:
        return None


def apply_recovered_thread_messages(
    pending_thread_id: str | None,
    final_tid: str,
    messages: list,
) -> None:
    """Replace local thread cache with authoritative backend messages."""
    apply_recovered_messages(
        st.session_state.thread_messages,
        pending_thread_id=pending_thread_id,
        final_tid=final_tid,
        messages=messages,
    )
    st.session_state.active_thread_id = final_tid
    _sync_thread_id_query_param(final_tid)


def try_recover_thread_after_send_failure(
    pending_thread_id: str | None,
    exc: BaseException,
    *,
    partial_result: dict | None = None,
) -> bool:
    """On gateway timeout / stream drop, reload thread from backend. Returns True if recovered."""
    if not is_recoverable_send_failure(exc):
        return False
    recovered = refresh_thread_after_send_failure(
        pending_thread_id, partial_result=partial_result
    )
    if not recovered:
        return False
    final_tid, messages = recovered
    apply_recovered_thread_messages(pending_thread_id, final_tid, messages)
    return True


def get_thread_title(thread: dict | None, thread_id: str) -> str:
    """Resolve display title: local override → backend metadata → formatted date."""
    if thread_id == NEW_THREAD:
        return "New conversation"
    if thread_id in st.session_state.conversation_titles:
        return st.session_state.conversation_titles[thread_id]
    if thread:
        title = (
            thread.get("metadata", {})
            .get("thread_metadata", {})
            .get("conversation_title")
        )
        if title and title != thread_id:
            return title
        created = thread.get("created_at", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                return dt.strftime("Conversation %b %d, %H:%M")
            except Exception:
                pass
    return f"Conversation {thread_id[:8]}…"

def get_active_thread_meta() -> dict | None:
    """Return the raw backend thread dict for the active thread (if any)."""
    tid = st.session_state.active_thread_id
    if not tid or tid == NEW_THREAD:
        return None
    for t in st.session_state.backend_threads:
        if t["thread_id"] == tid:
            return t
    return None

def validate_settings() -> list[str]:
    errors = []
    if not st.session_state.base_url.strip():
        errors.append("Base URL is required.")
    if not assistant_id:
        errors.append("No `?assistant_id=` in URL — e.g. `?assistant_id=abc123`")
    return errors


def assistant_chat_avatar() -> str:
    """Small portrait for assistant bubbles: reference image if available, else robot emoji."""
    ref = st.session_state.get("assistant_reference_icon")
    if isinstance(ref, str) and ref.strip():
        s = ref.strip()
        if s.startswith("https://") or s.startswith("http://") or s.startswith("data:image"):
            return s
    return "🤖"


def invalidate_avatar_icon_cache() -> None:
    """Force GET /avatar_reference_image on next render (e.g. after upload)."""
    st.session_state._ref_icon_cache_key = None
    st.session_state._ref_icon_attempts = 0


def _toggle_settings() -> None:
    st.session_state.show_settings = not st.session_state.show_settings


def _app_public_base_url() -> str:
    """Browser-facing origin + path (no query string) for share links."""
    ctx_url = getattr(st.context, "url", None)
    if ctx_url:
        p = urlparse(str(ctx_url))
        path = p.path or "/"
        return urlunparse((p.scheme, p.netloc, path, "", "", ""))
    h = st.context.headers
    host = h.get("Host") or h.get("host") or "localhost:8501"
    scheme = h.get("X-Forwarded-Proto") or h.get("x-forwarded-proto") or "http"
    return f"{scheme}://{host}/"


def _sync_thread_id_query_param(tid: str | None) -> None:
    """Reflect the open conversation in the URL so links are shareable."""
    qp = st.query_params
    want: str | None = None if (tid is None or tid == NEW_THREAD) else tid
    cur = (qp.get("thread_id") or "").strip() or None
    if want == cur:
        return
    if want:
        qp["thread_id"] = want
    elif "thread_id" in qp:
        del qp["thread_id"]


def build_share_url() -> str:
    """Public link to this assistant only — never includes the API key or thread_id.

    The api_key is a secret and must not be leaked in a shared URL; the thread_id is
    omitted so the link opens a fresh chat with the assistant rather than a private
    conversation.
    """
    parts: dict[str, str] = {}
    if assistant_id:
        parts["assistant_id"] = assistant_id
    q = urlencode(parts)
    base = _app_public_base_url().rstrip("/")
    return f"{base}/?{q}" if q else f"{base}/"


# ──────────────────────────────────────────────
# Startup / initialization logic
# (runs every Streamlit render pass)
# ──────────────────────────────────────────────
has_api_key = bool(st.session_state.api_key.strip())
cfg_ok = bool(st.session_state.base_url.strip() and assistant_id)

# ── 1. Resolve user_id (no api_key required) ──
if not st.session_state.user_id and cfg_ok:
    try:
        st.session_state.user_id = fetch_user_id()
    except Exception:
        pass  # will retry next render

# ── 2. Detect api_key change → reset thread state ──
if has_api_key and st.session_state.last_loaded_api_key != st.session_state.api_key:
    st.session_state.threads_loaded = False
    st.session_state.backend_threads = []
    st.session_state.thread_messages = {}
    st.session_state.active_thread_id = None
    st.session_state.pending_auto_message = None
    st.session_state.last_loaded_api_key = st.session_state.api_key
    st.session_state._ref_icon_cache_key = None

# ── 2b. Reference portrait for assistant avatar (works for anonymous + logged-in) ──
# Fetch the assistant's reference image and cache it for the session. Only a *successful*
# (non-empty) result is cached; an empty/failed fetch — e.g. the backend has not finished
# deriving the portrait on a cold open — is retried on later reruns instead of being
# pinned to the 🤖 fallback until a hard browser refresh. A small attempt cap avoids
# re-hitting the endpoint forever for assistants that legitimately have no portrait.
if cfg_ok:
    _rk = f"{assistant_id}:{(st.session_state.api_key or '').strip()}:refimg-v3"
    if st.session_state.get("_ref_icon_cache_key") != _rk:
        # New assistant/api-key → reset cache + retry budget.
        st.session_state._ref_icon_cache_key = _rk
        st.session_state._ref_icon_attempts = 0
        st.session_state.assistant_reference_icon = None
    if (
        not st.session_state.get("assistant_reference_icon")
        and st.session_state.get("_ref_icon_attempts", 0) < 5
    ):
        st.session_state._ref_icon_attempts = st.session_state.get("_ref_icon_attempts", 0) + 1
        try:
            _icon = fetch_avatar_reference_icon()
        except Exception:
            _icon = None
        if _icon:
            st.session_state.assistant_reference_icon = _icon

# ── 3. API-key mode: load threads + restore active thread ──
if has_api_key and cfg_ok:
    if not st.session_state.threads_loaded or st.session_state.last_loaded_api_key != st.session_state.api_key:
        try:
            threads = fetch_threads(assistant_id)
            st.session_state.backend_threads = threads
            st.session_state.threads_loaded = True
            st.session_state.last_loaded_api_key = st.session_state.api_key

            # Smart restore of active thread (`thread_id` query opens a shared conversation)
            qp_tid = (st.query_params.get("thread_id") or "").strip()
            thread_ids = {t["thread_id"] for t in threads}
            if qp_tid and qp_tid in thread_ids:
                st.session_state.active_thread_id = qp_tid
            elif st.session_state.active_thread_id and st.session_state.active_thread_id != NEW_THREAD:
                if st.session_state.active_thread_id not in thread_ids:
                    if threads:
                        st.session_state.active_thread_id = threads[0]["thread_id"]
            elif threads and (
                st.session_state.active_thread_id is None or st.session_state.active_thread_id == NEW_THREAD
            ):
                st.session_state.active_thread_id = threads[0]["thread_id"]
        except Exception as exc:
            st.error(f"❌ Failed to load conversations: {exc}")

    # Load messages for active thread
    tid = st.session_state.active_thread_id
    if tid and tid != NEW_THREAD and tid not in st.session_state.thread_messages:
        try:
            raw = fetch_thread_messages(tid, assistant_id)
            st.session_state.thread_messages[tid] = convert_lg_messages(raw)
        except Exception:
            st.session_state.thread_messages[tid] = []

# ── 4. Anonymous mode fallback (only if no API key)
elif not has_api_key and cfg_ok and st.session_state.active_thread_id is None:
    st.session_state.active_thread_id = NEW_THREAD
    st.session_state.thread_messages[NEW_THREAD] = []
    st.session_state.pending_auto_message = AUTO_GREETING

# ── 5. Anonymous mode: open a fresh conversation window ──
elif not has_api_key and cfg_ok and st.session_state.active_thread_id is None:
    st.session_state.active_thread_id = NEW_THREAD
    st.session_state.thread_messages[NEW_THREAD] = []
    st.session_state.pending_auto_message = AUTO_GREETING

_sync_thread_id_query_param(st.session_state.active_thread_id)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border-radius: 8px;
    margin-bottom: 2px;
    padding: 6px 10px;
    font-size: 0.85rem;
}
.resp-time {
    display: inline-block;
    font-size: 0.7rem;
    color: #aaa;
    padding: 1px 6px;
    border-radius: 20px;
    border: 1px solid #333;
    margin-top: 4px;
}
/* Composer rows: attach column + message shell */
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:first-child {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
/*
 * Nested [text | submit]: shell is position:relative; submit column is position:absolute
 * inside the same rounded bar as the text field (not a sibling strip beside it).
 */
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"],
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:nth-child(2) [class*="stHorizontalBlock"] {
    position: relative !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    border-radius: 12px;
    border: 1px solid rgba(250, 250, 250, 0.14);
    background: rgba(255, 255, 255, 0.06);
    padding: 4px 10px 4px 12px !important;
    box-sizing: border-box !important;
    min-height: 2.65rem !important;
}
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:nth-child(2) div[data-testid="column"]:first-child {
    flex: 1 1 auto !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    padding-right: 3rem !important;
}
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:nth-child(2) div[data-testid="column"]:first-child .stTextInput > div {
    border: none !important;
    background: transparent !important;
}
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:nth-child(2) div[data-testid="column"]:first-child input {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding-left: 2px !important;
}
/* Submit sits inside the bar; zero flex width so text column spans full shell */
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:nth-child(2) div[data-testid="column"]:last-child {
    position: absolute !important;
    right: 6px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: none !important;
    overflow: visible !important;
    flex: 0 0 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    z-index: 3 !important;
}
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:nth-child(2) div[data-testid="column"]:last-child .stButton {
    margin: 0 !important;
}
section[data-testid="stMain"] div[data-testid="stForm"] form div[data-testid="column"]:nth-child(2) div[data-testid="column"]:last-child .stButton > button {
    width: 2.65rem !important;
    min-height: 2.35rem !important;
    height: auto !important;
    padding: 0 !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 1.15rem !important;
    line-height: 1 !important;
}
/* Reserve scroll space above the docked composer */
section[data-testid="stMain"] .block-container {
    padding-bottom: clamp(11rem, 28vh, 22rem) !important;
}
section[data-testid="stMain"] [data-testid="stChatMessage"] {
    scroll-margin-bottom: min(24vh, 15rem);
}
/*
 * Pin only the inner <form>, not the outer stForm wrapper. Styling the wrapper with
 * position:fixed + background can stretch to the full main column and cover the app.
 */
section[data-testid="stMain"] div[data-testid="stForm"] {
    position: static !important;
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
    max-width: none !important;
}
section[data-testid="stMain"] div[data-testid="stForm"] form {
    position: fixed !important;
    z-index: 1001;
    bottom: 0;
    right: 0;
    margin: 0 !important;
    padding: 0.65rem 1.25rem calc(0.85rem + env(safe-area-inset-bottom, 0px)) 1.25rem !important;
    border-top: 1px solid rgba(250, 250, 250, 0.12);
    box-shadow: 0 -6px 28px rgba(0, 0, 0, 0.35);
    background: var(--secondary-background-color, #262730);
    width: calc(100vw - var(--sidebar-width, 21rem)) !important;
    max-width: none !important;
    box-sizing: border-box;
    max-height: min(50vh, 420px);
    overflow-y: auto;
}
body:has([data-testid="stSidebar"][aria-expanded="false"]) section[data-testid="stMain"] div[data-testid="stForm"] form {
    width: calc(100vw - 4.5rem) !important;
}
@media (max-width: 768px) {
    section[data-testid="stMain"] div[data-testid="stForm"] form {
        width: 100vw !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    col_title, col_gear = st.columns([5, 1])
    with col_title:
        st.markdown("### 🤖 Studio Chat")
    with col_gear:
        st.button(
            "⚙️",
            key="gear_btn",
            help="Open settings",
            on_click=_toggle_settings,
        )

    # ── Settings panel ──
    if st.session_state.show_settings:
        st.markdown("---")
        st.markdown("**⚙️ Settings**")
        st.session_state.api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="your-api-key",
            key="input_api_key",
        )
        settings_has_api_key = bool(st.session_state.api_key.strip())
        st.markdown("**🔗 Assistant ID** *(from URL)*")
        if assistant_id:
            st.code(assistant_id, language=None)
        else:
            st.warning("No `?assistant_id=` in URL")
        st.markdown("**👤 Your Identity**")
        st.session_state.user_name = st.text_input(
            "Your Name", value=st.session_state.user_name, placeholder="Your name (optional)", key="input_user_name"
        )
        st.session_state.user_description = st.text_area(
            "Your Description",
            value=st.session_state.user_description,
            placeholder="Brief description about you (optional)",
            key="input_user_desc",
            height=80,
        )

        if settings_has_api_key and cfg_ok:
            st.markdown("---")
            st.markdown("**🖼️ Avatar reference portrait**")
            st.caption(
                "Upload a clear face photo so the assistant icon uses it (stored server-side). "
                "Max file size: 25 MB (see `.streamlit/config.toml`)."
            )
            ref_up = st.file_uploader(
                "Reference image",
                type=["png", "jpg", "jpeg", "webp", "gif"],
                key="reference_image_upload",
                label_visibility="collapsed",
            )
            if ref_up and st.button("Upload reference image", key="btn_upload_ref_img"):
                try:
                    upload_avatar_reference_image(ref_up)
                    invalidate_avatar_icon_cache()
                    st.session_state.assistant_reference_icon = fetch_avatar_reference_icon()
                    st.success("Reference image uploaded.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")
            ref_url = st.text_input(
                "Or image URL",
                placeholder="https://…",
                key="reference_image_url_input",
            )
            if ref_url.strip() and st.button("Use image URL", key="btn_ref_url"):
                try:
                    upload_avatar_reference_image_url(ref_url)
                    invalidate_avatar_icon_cache()
                    st.session_state.assistant_reference_icon = fetch_avatar_reference_icon()
                    st.success("Reference image URL saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"URL upload failed: {exc}")
        elif cfg_ok and not settings_has_api_key:
            st.markdown("---")
            st.caption(
                "🖼️ The assistant’s reference portrait loads in chat when one is configured "
                "for this assistant. **Changing** it (upload or URL) needs an API key above."
            )

        if st.button("✅ Save & Close", key="save_settings"):
            st.session_state.show_settings = False
            st.rerun()

    st.markdown("---")

    # ── New conversation ──
    if st.button("➕  New conversation", key="new_thread_btn", use_container_width=True):
        st.session_state.active_thread_id = NEW_THREAD
        st.session_state.thread_messages[NEW_THREAD] = []
        st.session_state.pending_auto_message = AUTO_GREETING
        st.rerun()

    if cfg_ok and assistant_id:
        share_url = build_share_url()
        share_btn_id = f"studio-share-{uuid.uuid4().hex[:10]}"
        url_literal = json.dumps(share_url)
        components.html(
            f"""
<div style="font-family: 'Source Sans Pro', sans-serif; margin-top: 6px;">
  <button type="button" id="{share_btn_id}"
    style="width:100%; padding:6px 10px; border-radius:8px; cursor:pointer; font-size:0.85rem;
           margin-bottom:2px; text-align:center;">
    🔗 Share Avatar
  </button>
</div>
<script>
(function () {{
  var btn = document.getElementById({json.dumps(share_btn_id)});
  if (!btn) return;
  var url = {url_literal};
  btn.addEventListener("click", function () {{
    navigator.clipboard.writeText(url).then(function () {{
      btn.textContent = "✓ Copied";
      setTimeout(function () {{ btn.textContent = "🔗 Share Avatar"; }}, 2000);
    }}).catch(function () {{
      btn.textContent = "Copy blocked — use browser bar";
      setTimeout(function () {{ btn.textContent = "🔗 Share Avatar"; }}, 2500);
    }});
  }});
}})();
</script>
            """,
            height=52,
        )
        st.caption("Copies a link to this assistant (no API key, opens a fresh chat).")

    st.markdown("---")

    # ── Thread list ──
    if has_api_key and st.session_state.backend_threads:
        st.markdown("**💬 Conversations**")

        # Prepend the __new__ thread if it exists and has messages
        all_sidebar_threads = []
        if NEW_THREAD in st.session_state.thread_messages:
            all_sidebar_threads.append((NEW_THREAD, None))  # (id, meta)
        for t in st.session_state.backend_threads:
            all_sidebar_threads.append((t["thread_id"], t))

        for tid, tmeta in all_sidebar_threads:
            is_active = tid == st.session_state.active_thread_id
            title = get_thread_title(tmeta, tid)
            label = f"{'▶ ' if is_active else ''}{title}"
            # Date caption
            date_str = ""
            if tmeta:
                try:
                    updated = tmeta.get("updated_at", "")
                    if updated:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        date_str = dt.strftime("%b %d, %H:%M")
                except Exception:
                    pass

            btn_col, ref_col = st.columns([5, 1])
            with btn_col:
                if st.button(
                    label,
                    key=f"thread_{tid}",
                    help=date_str or "New conversation",
                    use_container_width=True,
                ):
                    st.session_state.active_thread_id = tid
                    # Always reload messages fresh on thread switch
                    if tid != NEW_THREAD:
                        try:
                            raw = fetch_thread_messages(tid, assistant_id)
                            st.session_state.thread_messages[tid] = convert_lg_messages(raw)
                        except Exception:
                            st.session_state.thread_messages.setdefault(tid, [])
                    st.session_state.pending_auto_message = None
                    st.rerun()
            with ref_col:
                if tid != NEW_THREAD:
                    if st.button("🔄", key=f"ref_{tid}", help="Refresh messages"):
                        try:
                            raw = fetch_thread_messages(tid, assistant_id)
                            st.session_state.thread_messages[tid] = convert_lg_messages(raw)
                        except Exception:
                            pass
                        st.rerun()
    elif not has_api_key:
        st.caption("💡 Add an API key to load your full conversation history.")

    # ── Export ──
    all_exportable = {
        tid: msgs
        for tid, msgs in st.session_state.thread_messages.items()
        if msgs
    }
    if all_exportable:
        st.markdown("---")
        st.download_button(
            "⬇️ Export messages",
            data=json.dumps(_json_export_sanitize(all_exportable), indent=2),
            file_name="studio_threads.json",
            mime="application/json",
            use_container_width=True,
        )

# ──────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────
errors = validate_settings()
tid = st.session_state.active_thread_id

# ── No thread open ──
if tid is None:
    st.markdown("## 👋 Welcome to Studio Chat")
    st.markdown(
        "Configure your settings with **⚙️** in the sidebar, "
        "then click **➕ New conversation** to get started.\n\n"
        "The assistant is determined by the URL: `?assistant_id=<your-id>`"
    )
    if errors:
        with st.expander("⚠️ Configuration needed", expanded=True):
            for e in errors:
                st.warning(e)
    st.stop()

# ── Thread header ──
thread_meta = get_active_thread_meta()
current_title = get_thread_title(thread_meta, tid)
messages: list = st.session_state.thread_messages.get(tid, [])

header_col, rename_col = st.columns([6, 2])
with header_col:
    st.markdown(f"### {current_title}")
    if tid != NEW_THREAD:
        exchange_count = sum(1 for m in messages if m["role"] == "user")
        updated = ""
        if thread_meta:
            try:
                raw_dt = thread_meta.get("updated_at", "")
                if raw_dt:
                    dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                    updated = dt.strftime("%b %d, %H:%M")
            except Exception:
                pass
        st.caption(
            f"Assistant: `{assistant_id}` · "
            + (f"Updated: {updated} · " if updated else "")
            + f"{exchange_count} exchange(s)"
        )
    else:
        st.caption("New conversation — not yet saved")

with rename_col:
    with st.popover("✏️ Rename"):
        new_title = st.text_input("New title", value=current_title, key="rename_input")
        if st.button("Save title", key="do_rename"):
            st.session_state.conversation_titles[tid] = new_title
            st.rerun()

st.divider()

# ── Config error banner ──
if errors:
    for e in errors:
        st.error(f"⚠️ {e}  — open **⚙️ Settings** in the sidebar to fix this.")

# ── Render existing messages ──
for msg in messages:
    role = msg["role"]
    _av = "🧑" if role == "user" else assistant_chat_avatar()
    with st.chat_message(role, avatar=_av):
        if role == "user" and msg.get("attachment_meta"):
            _render_user_chat_content(msg)
        else:
            st.markdown(msg["content"])
        if role == "assistant" and msg.get("response_time_ms"):
            st.markdown(
                f'<span class="resp-time">⏱ {msg["response_time_ms"]} ms</span>',
                unsafe_allow_html=True,
            )

# ── Pending API send (must render above fixed composer — before st.form) ──
_pending_send = st.session_state.get("_studio_pending_send")
if _pending_send and _pending_send.get("tid") == tid:
    _ptid = _pending_send["tid"]
    _text_p = (_pending_send.get("text") or "").strip()
    _raw_atts = _pending_send.get("attachments") or []
    _files_payload = (
        [
            _BytesAttachment(a["name"], a["type"], a["bytes"])
            for a in _raw_atts
        ]
        if _raw_atts
        else None
    )

    with st.chat_message("assistant", avatar=assistant_chat_avatar()):
        stream_slot = st.empty()
        stream_slot.caption("Thinking…")
        try:
            # Latch: once a turn is identified as a leaked fact-correction stream, keep the
            # placeholder for the rest of it. The interleaved JSON can momentarily look like
            # prose at the tail, and we must never flip back to dumping the raw buffer.
            _suppress_stream = {"on": False}

            def _show_chunk(text_so_far: str) -> None:
                # A fact-correction turn streams internal structured-output JSON (often
                # several analyses interleaved) before it resolves into the approve/edit/
                # reject panel below. Don't render that raw — hold a placeholder until the
                # turn is ready to display properly.
                if _suppress_stream["on"] or _streamed_text_is_internal_json(text_so_far):
                    _suppress_stream["on"] = True
                    stream_slot.caption("✏️ Suggesting edits…")
                else:
                    stream_slot.markdown(text_so_far)

            result = api_send_message(
                _text_p,
                thread_id=_ptid,
                attachments=_files_payload,
                on_stream_chunk=_show_chunk,
            )
            if isinstance(result, dict) and result.get("interrupt"):
                # Paused for human approval (e.g. a fact correction). Hand off to the
                # approve/edit/reject panel rendered below; do NOT finalize a reply.
                # Keep the UI thread key as-is; the backend thread_id is carried for
                # the resume call and the thread migrates on final completion.
                st.session_state["_studio_pending_interrupt"] = {
                    "tid": _ptid,
                    "thread_id": result.get("thread_id") or _ptid,
                    "interrupt": result["interrupt"],
                }
                stream_slot.caption("✏️ Awaiting your confirmation on a correction…")
                st.session_state.pop("_studio_pending_send", None)
                result = None

            if result is not None:
                reply = result.get("content", "(empty response)")
                # Final render (covers responses with only a terminal ``content`` field, no tokens)
                stream_slot.markdown(reply)
                resp_time = result.get("total_response_time_ms")
                returned_tid = result.get("thread_id")

                if resp_time:
                    st.markdown(
                        f'<span class="resp-time">⏱ {resp_time} ms</span>',
                        unsafe_allow_html=True,
                    )

                _msgs = st.session_state.thread_messages.get(_ptid, [])
                _msgs.append({
                    "role": "assistant",
                    "content": reply,
                    "response_time_ms": resp_time,
                    "metadata": result.get("response_metadata", {}),
                })

                final_tid = returned_tid or _ptid

                if final_tid:
                    if _ptid == NEW_THREAD or _ptid is None or _ptid != final_tid:
                        st.session_state.thread_messages[final_tid] = _msgs
                        if _ptid in st.session_state.thread_messages and _ptid != final_tid:
                            del st.session_state.thread_messages[_ptid]
                        st.session_state.active_thread_id = final_tid

                        if has_api_key and cfg_ok:
                            try:
                                st.session_state.backend_threads = fetch_threads(assistant_id)
                            except Exception:
                                pass

                    if final_tid not in st.session_state.conversation_titles:
                        title_src = _text_p if _text_p else (
                            _raw_atts[0]["name"] if _raw_atts else ""
                        )
                        short = title_src[:45] + ("…" if len(title_src) > 45 else "")
                        st.session_state.conversation_titles[final_tid] = short

                if not st.session_state.active_thread_id:
                    st.session_state.active_thread_id = final_tid

                if _raw_atts:
                    st.session_state.attachment_uploader_bump += 1

                st.session_state.pop("_studio_pending_send", None)

        except requests.exceptions.ConnectionError as exc:
            if try_recover_thread_after_send_failure(_ptid, exc):
                st.session_state.pop("_studio_pending_send", None)
                st.warning(
                    "⚠️ Connection dropped while waiting for a reply. "
                    "Reloaded this conversation from the server — check for a response below."
                )
                st.rerun()
            else:
                st.session_state.pop("_studio_pending_send", None)
                st.error(
                    f"❌ Could not connect to `{st.session_state.base_url}`. Is the server running?"
                )
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if try_recover_thread_after_send_failure(_ptid, exc):
                st.session_state.pop("_studio_pending_send", None)
                st.warning(
                    f"⚠️ HTTP {status} (gateway timeout). "
                    "Reloaded this conversation from the server — your message may already be there."
                )
                st.rerun()
            else:
                st.session_state.pop("_studio_pending_send", None)
                detail = ""
                try:
                    detail = exc.response.json().get("detail", "")
                except Exception:
                    pass
                st.error(f"❌ HTTP {status}: {detail or str(exc)}")
        except requests.exceptions.Timeout as exc:
            if try_recover_thread_after_send_failure(_ptid, exc):
                st.session_state.pop("_studio_pending_send", None)
                st.warning(
                    "⚠️ Request timed out. "
                    "Reloaded this conversation from the server — your message may already be there."
                )
                st.rerun()
            else:
                st.session_state.pop("_studio_pending_send", None)
                st.error("❌ Request timed out. The server may be overloaded.")
        except Exception as exc:
            if try_recover_thread_after_send_failure(_ptid, exc):
                st.session_state.pop("_studio_pending_send", None)
                st.warning(
                    "⚠️ Send failed but the server may have saved your message. "
                    "Reloaded this conversation from the server."
                )
                st.rerun()
            else:
                st.session_state.pop("_studio_pending_send", None)
                st.error(f"❌ Unexpected error: {exc}")

# ── Pending resume (apply/cancel was chosen → continue the paused run) ──
_pending_resume = st.session_state.get("_studio_pending_resume")
if _pending_resume:
    _rtid = _pending_resume["tid"]
    with st.chat_message("assistant", avatar=assistant_chat_avatar()):
        stream_slot = st.empty()
        stream_slot.caption("Applying…")
        try:
            # Latch: once a turn is identified as a leaked fact-correction stream, keep the
            # placeholder for the rest of it. The interleaved JSON can momentarily look like
            # prose at the tail, and we must never flip back to dumping the raw buffer.
            _suppress_stream = {"on": False}

            def _show_chunk(text_so_far: str) -> None:
                # A fact-correction turn streams internal structured-output JSON (often
                # several analyses interleaved) before it resolves into the approve/edit/
                # reject panel below. Don't render that raw — hold a placeholder until the
                # turn is ready to display properly.
                if _suppress_stream["on"] or _streamed_text_is_internal_json(text_so_far):
                    _suppress_stream["on"] = True
                    stream_slot.caption("✏️ Suggesting edits…")
                else:
                    stream_slot.markdown(text_so_far)

            result = api_resume_message(
                _pending_resume["thread_id"],
                _pending_resume["decision"],
                items=_pending_resume.get("items"),
                on_stream_chunk=_show_chunk,
            )
            if isinstance(result, dict) and result.get("interrupt"):
                # The continuation paused again (another correction) → re-open the panel.
                st.session_state["_studio_pending_interrupt"] = {
                    "tid": _rtid,
                    "thread_id": result.get("thread_id") or _pending_resume["thread_id"],
                    "interrupt": result["interrupt"],
                }
                stream_slot.caption("✏️ Awaiting your confirmation on a correction…")
                st.session_state.pop("_studio_pending_resume", None)
            else:
                reply = result.get("content", "(empty response)")
                stream_slot.markdown(reply)
                resp_time = result.get("total_response_time_ms")
                if resp_time:
                    st.markdown(
                        f'<span class="resp-time">⏱ {resp_time} ms</span>',
                        unsafe_allow_html=True,
                    )
                _msgs = st.session_state.thread_messages.get(_rtid, [])
                _msgs.append({
                    "role": "assistant",
                    "content": reply,
                    "response_time_ms": resp_time,
                    "metadata": result.get("response_metadata", {}),
                })
                st.session_state.thread_messages[_rtid] = _msgs
                st.session_state.pop("_studio_pending_resume", None)
        except Exception as exc:
            # Resume failures get the same backend-reload recovery as a normal send.
            if try_recover_thread_after_send_failure(_rtid, exc):
                st.session_state.pop("_studio_pending_resume", None)
                st.warning(
                    "⚠️ Connection dropped while applying the correction. "
                    "Reloaded this conversation from the server — check below."
                )
                st.rerun()
            else:
                st.session_state.pop("_studio_pending_resume", None)
                st.error(f"❌ Could not apply the correction: {exc}")

# ── Per-document action panel for a paused fact correction (human-in-the-loop) ──
# Each matched document is its own item with THREE explicit choices: "Accept Edit" applies the
# editable window, "Remove the Document" deletes/redacts it, "Leave the document unchanged" is
# the safe default. The radio is PRE-SELECTED to the backend's ``recommended_action`` (a loose
# match recommends "leave unchanged", so a false positive is never pre-armed for a change).
# "Apply my choices" applies each item's action; "Cancel correction" abandons everything.
_ACTION_ORDER = ["accept", "remove", "skip"]
# Fallback labels if the payload omits ``action_labels``; the backend now ships its own.
_ACTION_LABELS = {
    "accept": "✅ Accept Edit",
    "remove": "🗑️ Remove the Document",
    "skip": "🚫 Leave the document unchanged",
}
_pending_interrupt = st.session_state.get("_studio_pending_interrupt")
if _pending_interrupt:
    _intr = _pending_interrupt.get("interrupt") or {}
    _matches = _intr.get("matches") or []
    # Prefer the backend-supplied labels so the panel always matches the server's vocabulary.
    _action_labels = {**_ACTION_LABELS, **(_intr.get("action_labels") or {})}
    with st.chat_message("assistant", avatar=assistant_chat_avatar()):
        st.markdown(
            f"**✏️ I found {len(_matches)} stored item(s) that might match — choose what to do with each.**"
        )
        if _intr.get("inaccurate_information"):
            st.caption(f"You flagged as inaccurate: _{_intr['inaccurate_information']}_")
        st.caption(
            "Each item is pre-selected to my recommendation; you can change any of them. "
            "Anything I recommend leaving unchanged stays exactly as-is unless you pick "
            "another action."
        )

        with st.form("interrupt_corrections"):
            _form_state: list[dict] = []
            for _m in _matches:
                _idx = _m.get("index")
                _ns = "/".join(str(p) for p in (_m.get("namespace") or []))
                _kind = _m.get("kind", "fact")
                _current = _m.get("current_fact_content") or "(unnamed)"
                # Suggested edit is populated by the backend ONLY when the recommendation is to
                # edit (empty for leave-unchanged / remove); it pre-fills the editable window.
                _suggested = _m.get("suggested_edit_fact_content", "")
                _suggested_ctx = _m.get("suggested_edit_fact_context", "")
                _recommended = _m.get("recommended_action") or "skip"
                if _recommended not in _ACTION_ORDER:
                    _recommended = "skip"
                _doc_id = _m.get("document_id") or _m.get("key") or "(no id)"
                with st.container(border=True):
                    _label = "sentence in quote/long text" if _kind == "sentence" else "fact"
                    # Surface the match score so loose/false-positive matches are easy to spot
                    # (a low score usually means the document was swept in by a loose semantic
                    # match and should be left unchanged).
                    _pct = _m.get("match_percent")
                    if not isinstance(_pct, (int, float)):
                        _score = _m.get("score")
                        _pct = round(_score * 100) if isinstance(_score, (int, float)) else None
                    _score_txt = f" · match {_pct}%" if _pct is not None else ""
                    st.caption(f"📄 {_ns} · {_label}{_score_txt}")
                    # The concrete document id distinguishes otherwise-identical matched facts.
                    st.caption(f"document_id: `{_doc_id}`")
                    st.markdown(f"**Current document fact content:** `{_current}`")
                    if _m.get("current_fact_context"):
                        st.caption(f"Current document fact context: {_m['current_fact_context']}")
                    if _recommended == "accept" and _suggested:
                        st.markdown(f"**Suggested edit:** `{_suggested}`")
                    elif _recommended == "remove":
                        st.caption("💡 I recommend removing this document (it only states the inaccurate event).")
                    else:
                        st.caption("💡 I recommend leaving this document unchanged.")
                    # Pre-select the recommended action (highlighted) so the owner can confirm
                    # with one click; the safe "leave unchanged" stays the recommendation for
                    # any loosely-matched document.
                    _action = st.radio(
                        "What should I do with this?",
                        options=_ACTION_ORDER,
                        format_func=lambda a: _action_labels.get(a, a),
                        index=_ACTION_ORDER.index(_recommended),
                        key=f"interrupt_action_{_idx}",
                        horizontal=True,
                    )
                    # The editable window is what "Accept Edit" applies. It is prefilled with the
                    # suggested edit so accepting is one click; the owner may rewrite it (the
                    # backend records an owner-authored edit as ``correction_origin: "user"``).
                    _corrected = st.text_area(
                        "Suggested edit fact content (applied when you choose “Accept Edit”)",
                        value=_suggested,
                        key=f"interrupt_text_{_idx}",
                    )
                    _ctx = st.text_area(
                        "Suggested edit fact context (applied when you choose “Accept Edit”)",
                        value=_suggested_ctx,
                        key=f"interrupt_ctx_{_idx}",
                    )
                    _form_state.append({
                        "index": _idx,
                        "action_key": f"interrupt_action_{_idx}",
                        "text_key": f"interrupt_text_{_idx}",
                        "ctx_key": f"interrupt_ctx_{_idx}",
                    })
            _col_a, _col_r = st.columns(2)
            _apply = _col_a.form_submit_button("✅ Apply my choices", use_container_width=True)
            _cancel = _col_r.form_submit_button("🚫 Cancel correction", use_container_width=True)

        if _apply or _cancel:
            _resume = {
                "tid": _pending_interrupt["tid"],
                "thread_id": _pending_interrupt["thread_id"],
                "decision": "cancel" if _cancel else "apply",
            }
            if _apply:
                _resume["items"] = [
                    {
                        "index": _f["index"],
                        "action": st.session_state.get(_f["action_key"], "skip"),
                        "corrected_text": st.session_state.get(_f["text_key"], ""),
                        "correction_context": st.session_state.get(_f["ctx_key"], ""),
                    }
                    for _f in _form_state
                ]
            st.session_state["_studio_pending_resume"] = _resume
            st.session_state.pop("_studio_pending_interrupt", None)
            st.rerun()

# ── Message composer: one row [attach | type message | send] + preview inside form ──
composer_disabled = bool(errors)
placeholder = "Type your message…" if not errors else "Fix configuration errors before chatting."
_uploader_key = f"msg_attachment_{st.session_state.attachment_uploader_bump}"

attachment = None
msg_field = ""
submitted = False
with st.form("message_bar", clear_on_submit=True):
    bar = st.columns([1.35, 22])
    with bar[0]:
        attachment = st.file_uploader(
            "📎",
            label_visibility="collapsed",
            key=_uploader_key,
            disabled=composer_disabled,
            help="Attach a file",
        )
    with bar[1]:
        inner = st.columns([18, 2], gap="small")
        with inner[0]:
            msg_field = st.text_input(
                "Message",
                placeholder=placeholder,
                label_visibility="collapsed",
                disabled=composer_disabled,
                key="message_bar_text",
            )
        with inner[1]:
            submitted = st.form_submit_button(
                "➤",
                disabled=composer_disabled,
                use_container_width=True,
                help="Send",
            )
    if attachment:
        _preview_attachment_bar(attachment)

text = ""
auto_from_pending = False
if submitted:
    text = (msg_field or "").strip()
elif st.session_state.pending_auto_message:
    text = st.session_state.pending_auto_message.strip()
    st.session_state.pending_auto_message = None
    auto_from_pending = True

should_send = auto_from_pending or (submitted and (bool(text) or bool(attachment)))

# ── Process message (enqueue user turn + rerun so API/chat renders above fixed form) ──
if should_send:
    current_tid = tid

    img_norm: tuple[bytes, str] | None = None
    if attachment and (attachment.type or "").startswith("image/"):
        img_norm = _chat_image_bytes_and_mime(
            attachment.getvalue(), attachment.type or ""
        )

    save_meta = None
    if attachment:
        mime = attachment.type or ""
        save_meta = {
            "name": attachment.name,
            "mime": mime,
        }
        if img_norm is not None:
            save_meta["bytes"], save_meta["mime"] = img_norm

    _msgs = st.session_state.thread_messages.get(current_tid, [])
    _msgs.append({
        "role": "user",
        "content": text,
        "response_time_ms": None,
        "attachment_meta": save_meta,
    })
    st.session_state.thread_messages[current_tid] = _msgs

    _att_payload = None
    if attachment:
        att_mime = attachment.type or "application/octet-stream"
        att_body = attachment.getvalue()
        if img_norm is not None:
            att_body, att_mime = img_norm
        _att_payload = [
            {
                "name": attachment.name,
                "type": att_mime,
                "bytes": att_body,
            }
        ]

    st.session_state["_studio_pending_send"] = {
        "tid": current_tid,
        "text": text,
        "attachments": _att_payload,
    }

    st.rerun()


    