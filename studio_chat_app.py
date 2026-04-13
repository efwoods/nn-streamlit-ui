"""
Studio Assistant Chat UI
-------------------------
A Streamlit chat interface for the /studio/{assistant_id} API endpoint.

Run with:
    streamlit run studio_chat_app.py
"""

import uuid
import json
import requests
from datetime import datetime

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

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

# ──────────────────────────────────────────────
# Session-state bootstrap
# ──────────────────────────────────────────────
DEFAULTS = {
    "base_url": os.getenv('NN_API_BASE_URL'),
    "api_key": "",
    "user_name": "",
    "user_description": "",
    # threads: dict[thread_id -> {title, created_at, messages: list}]
    "threads": {},
    "active_thread_id": None,
    "show_settings": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def new_thread_id() -> str:
    return str(uuid.uuid4())


def create_thread(title: str = "New conversation") -> str:
    tid = new_thread_id()
    st.session_state.threads[tid] = {
        "title": title,
        "created_at": datetime.now().strftime("%b %d, %H:%M"),
        "messages": [],
    }
    st.session_state.active_thread_id = tid
    return tid


def active_thread() -> dict | None:
    tid = st.session_state.active_thread_id
    return st.session_state.threads.get(tid) if tid else None


def send_message(user_message: str) -> dict:
    """POST to /studio/{assistant_id} and return the response dict."""
    url = f"{st.session_state.base_url.rstrip('/')}/studio/{assistant_id}"
    headers = {
        "Content-Type": "application/json",
        "api-key": st.session_state.api_key,
    }
    payload = {
        "message": user_message,
        "your_name": st.session_state.user_name,
        "your_description": st.session_state.user_description,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=None)
    response.raise_for_status()
    return response.json()


def validate_settings() -> list[str]:
    errors = []
    if not st.session_state.base_url.strip():
        errors.append("Base URL is required.")
    if not assistant_id:
        errors.append("No assistant_id in URL — add `?assistant_id=<id>` to the address bar.")
    return errors


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
/* Tighten up sidebar thread buttons */
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
/* Active thread highlight */
div[data-testid="stSidebar"] .stButton > button[aria-pressed="true"],
div[data-testid="stSidebar"] .active-thread button {
    background-color: rgba(255,255,255,0.15) !important;
    border-left: 3px solid #7c5cfc !important;
}
/* Settings popover-style container */
.settings-panel {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #333;
}
/* Metadata chip */
.meta-chip {
    font-size: 0.72rem;
    color: #888;
    margin-top: 2px;
}
/* Response time badge */
.resp-time {
    display:inline-block;
    font-size:0.7rem;
    color:#aaa;
    padding: 1px 6px;
    border-radius:20px;
    border:1px solid #333;
    margin-top: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:

    # ── Header row: title + gear icon ──
    col_title, col_gear = st.columns([5, 1])
    with col_title:
        st.markdown("### 🤖 Studio Chat")
    with col_gear:
        if st.button("⚙️", key="gear_btn", help="Open settings"):
            st.session_state.show_settings = not st.session_state.show_settings

    # ── Settings panel (toggle) ──
    if st.session_state.show_settings:
        with st.container():
            st.markdown("---")
            st.markdown("**⚙️ Settings**")

            st.session_state.api_key = st.text_input(
                "API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="your-api-key",
                key="input_api_key",
            )

            # Assistant ID comes from the URL query param — display only
            st.markdown("**🔗 Assistant ID** *(from URL)*")
            if assistant_id:
                st.code(assistant_id, language=None)
                st.caption(
                    f"Endpoint: `{st.session_state.base_url.rstrip('/')}/studio/{assistant_id}`"
                )
            else:
                st.warning("No `?assistant_id=` in URL — e.g. `?assistant_id=abc123`")

            st.markdown("**👤 Your Identity**")
            st.session_state.user_name = st.text_input(
                "Your Name",
                value=st.session_state.user_name,
                key="input_user_name",
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

    # ── New conversation button ──
    if st.button("➕  New conversation", key="new_thread_btn", use_container_width=True):
        create_thread()
        st.rerun()

    st.markdown("---")

    # ── Thread list ──
    if st.session_state.threads:
        st.markdown("**💬 Conversations**")
        # Show threads newest-first
        sorted_threads = sorted(
            st.session_state.threads.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True,
        )
        for tid, tdata in sorted_threads:
            is_active = tid == st.session_state.active_thread_id
            label = f"{'▶ ' if is_active else ''}{tdata['title']}"
            btn_col, del_col = st.columns([5, 1])
            with btn_col:
                if st.button(
                    label,
                    key=f"thread_{tid}",
                    help=f"Started {tdata['created_at']}",
                    use_container_width=True,
                ):
                    st.session_state.active_thread_id = tid
                    st.rerun()
            with del_col:
                if st.button("🗑", key=f"del_{tid}", help="Delete thread"):
                    del st.session_state.threads[tid]
                    if st.session_state.active_thread_id == tid:
                        remaining = list(st.session_state.threads.keys())
                        st.session_state.active_thread_id = remaining[0] if remaining else None
                    st.rerun()
    else:
        st.caption("No conversations yet — start one above!")

    # ── Export ──
    if st.session_state.threads:
        st.markdown("---")
        export_data = json.dumps(
            {
                tid: {
                    "title": t["title"],
                    "created_at": t["created_at"],
                    "messages": t["messages"],
                }
                for tid, t in st.session_state.threads.items()
            },
            indent=2,
        )
        st.download_button(
            "⬇️ Export all threads",
            data=export_data,
            file_name="studio_threads.json",
            mime="application/json",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────
errors = validate_settings()

# ── No thread selected ──
thread = active_thread()
if thread is None:
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
header_col, rename_col = st.columns([6, 2])
with header_col:
    st.markdown(f"### {thread['title']}")
    st.caption(
        f"Assistant: `{assistant_id or '(not set)'}` · "
        f"Started: {thread['created_at']} · "
        f"{len(thread['messages']) // 2} exchange(s)"
    )

with rename_col:
    with st.popover("✏️ Rename"):
        new_title = st.text_input(
            "New title", value=thread["title"], key="rename_input"
        )
        if st.button("Rename", key="do_rename"):
            st.session_state.threads[st.session_state.active_thread_id]["title"] = new_title
            st.rerun()

st.divider()

# ── Config errors banner ──
if errors:
    for e in errors:
        st.error(f"⚠️ {e}  — open **⚙️ Settings** in the sidebar to fix this.")

# ── Render existing messages ──
for msg in thread["messages"]:
    role = msg["role"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(msg["content"])
        if role == "assistant" and "response_time_ms" in msg:
            st.markdown(
                f'<span class="resp-time">⏱ {msg["response_time_ms"]} ms</span>',
                unsafe_allow_html=True,
            )

# ── Chat input ──
placeholder = (
    "Type your message…"
    if not errors
    else "Fix configuration errors before chatting."
)
user_input = st.chat_input(placeholder, disabled=bool(errors))

if user_input:
    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Store user message
    thread["messages"].append({"role": "user", "content": user_input})

    # Auto-name the thread from the first user message
    if len(thread["messages"]) == 1:
        short_title = user_input[:42] + ("…" if len(user_input) > 42 else "")
        st.session_state.threads[st.session_state.active_thread_id]["title"] = short_title

    # Call the API
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                result = send_message(user_input)
                reply = result.get("content", "(empty response)")
                resp_time = result.get("total_response_time_ms")

                st.markdown(reply)
                if resp_time:
                    st.markdown(
                        f'<span class="resp-time">⏱ {resp_time} ms</span>',
                        unsafe_allow_html=True,
                    )

                # Persist assistant message
                thread["messages"].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "response_time_ms": resp_time,
                        "metadata": result.get("response_metadata", {}),
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Could not connect to `{st.session_state.base_url}`. "
                    "Is the server running?"
                )
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                detail = ""
                try:
                    detail = exc.response.json().get("detail", "")
                except Exception:
                    pass
                st.error(f"❌ HTTP {status}: {detail or str(exc)}")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out (>60 s). The server may be overloaded.")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")