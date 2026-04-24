"""
Studio Assistant Chat UI
-------------------------
A Streamlit chat interface for the /studio/{assistant_id} API endpoint.
Run with:
    streamlit run studio_chat_app.py
"""
import json
import requests
from datetime import datetime
import streamlit as st
import os
from dotenv import load_dotenv

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
# Assistant ID — read from URL query param
#   Usage:  http://localhost:8501/?assistant_id=abc123
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

def api_send_message(user_message: str, thread_id: str | None = None) -> dict:
    """POST /message/{assistant_id}  →  {content, thread_id, total_response_time_ms, …}"""
    real_thread_id = None if (thread_id is None or thread_id == NEW_THREAD) else thread_id
    title = st.session_state.conversation_titles.get(thread_id) if thread_id else None
    payload = {
        "message": user_message,
        "your_name": st.session_state.user_name or None,
        "your_description": st.session_state.user_description or None,
        "conversation_title": title,
        "thread_id": real_thread_id
    }
    resp = requests.post(
        f"{_base()}/message/{assistant_id}",
        headers=_headers(),
        data=payload,
        timeout=None,
    )
    resp.raise_for_status()
    return resp.json()

# ──────────────────────────────────────────────
# Conversion / display helpers
# ──────────────────────────────────────────────

def convert_lg_messages(lg_messages: list) -> list:
    """Map LangGraph {type:'human'|'ai'} messages to {role:'user'|'assistant'}."""
    result = []
    for msg in (lg_messages or []):
        t = msg.get("type", "")
        if t in ("human", "ai"):
            result.append({
                "role": "user" if t == "human" else "assistant",
                "content": msg.get("content", ""),
                "id": msg.get("id"),
                "response_time_ms": None,
            })
    return result

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

# ── 3. API-key mode: load threads + restore active thread ──
if has_api_key and cfg_ok:
    if not st.session_state.threads_loaded or st.session_state.last_loaded_api_key != st.session_state.api_key:
        try:
            threads = fetch_threads(assistant_id)
            st.session_state.backend_threads = threads
            st.session_state.threads_loaded = True
            st.session_state.last_loaded_api_key = st.session_state.api_key

            # Smart restore of active thread
            if st.session_state.active_thread_id and st.session_state.active_thread_id != NEW_THREAD:
                if not any(t["thread_id"] == st.session_state.active_thread_id for t in threads):
                    if threads:
                        st.session_state.active_thread_id = threads[0]["thread_id"]
            elif threads and (st.session_state.active_thread_id is None or st.session_state.active_thread_id == NEW_THREAD):
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
        if st.button("⚙️", key="gear_btn", help="Open settings"):
            st.session_state.show_settings = not st.session_state.show_settings

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
            data=json.dumps(all_exportable, indent=2),
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
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("response_time_ms"):
            st.markdown(
                f'<span class="resp-time">⏱ {msg["response_time_ms"]} ms</span>',
                unsafe_allow_html=True,
            )

# ── Resolve message to send this render (auto-greeting or user input) ──
placeholder = "Type your message…" if not errors else "Fix configuration errors before chatting."
user_input = st.chat_input(placeholder, disabled=bool(errors))

if st.session_state.pending_auto_message and not user_input:
    user_input = st.session_state.pending_auto_message
    st.session_state.pending_auto_message = None

# ── Process message ──
if user_input:
    # Use the already-resolved tid from above (don't re-assign here)
    current_tid = tid  # this 'tid' comes from the line before the header

    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Append user message 
    messages = st.session_state.thread_messages.get(current_tid, [])
    messages.append({"role": "user", "content": user_input, "response_time_ms": None})
    st.session_state.thread_messages[current_tid] = messages

    # Send to API
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                result = api_send_message(user_input, thread_id=current_tid)
                reply = result.get("content", "(empty response)")
                resp_time = result.get("total_response_time_ms")
                returned_tid = result.get("thread_id")

                st.markdown(reply)
                if resp_time:
                    st.markdown(f'<span class="resp-time">⏱ {resp_time} ms</span>', unsafe_allow_html=True)

                # Append assistant reply
                messages.append({
                    "role": "assistant",
                    "content": reply,
                    "response_time_ms": resp_time,
                    "metadata": result.get("response_metadata", {}),
                })

                # === FIXED THREAD PROMOTION ===
                final_tid = returned_tid or current_tid

                if final_tid:
                    if current_tid == NEW_THREAD or current_tid is None or current_tid != final_tid:
                        # Promote / move to real thread
                        st.session_state.thread_messages[final_tid] = messages
                        if current_tid in st.session_state.thread_messages and current_tid != final_tid:
                            del st.session_state.thread_messages[current_tid]
                        st.session_state.active_thread_id = final_tid

                        # Refresh sidebar
                        if has_api_key and cfg_ok:
                            try:
                                st.session_state.backend_threads = fetch_threads(assistant_id)
                            except Exception:
                                pass

                    # Default title
                    if final_tid not in st.session_state.conversation_titles:
                        short = user_input[:45] + ("…" if len(user_input) > 45 else "")
                        st.session_state.conversation_titles[final_tid] = short

                # Safety
                if not st.session_state.active_thread_id:
                    st.session_state.active_thread_id = final_tid

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to `{st.session_state.base_url}`. Is the server running?")
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                detail = ""
                try:
                    detail = exc.response.json().get("detail", "")
                except Exception:
                    pass
                st.error(f"❌ HTTP {status}: {detail or str(exc)}")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The server may be overloaded.")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")