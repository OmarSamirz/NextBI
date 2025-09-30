import streamlit as st

import asyncio
import datetime as dt
from typing import List

from utils import get_ai
from modules.logger import ChatLogger
from ai_modules.base import Message, AI


MAX_MESSAGES: int = 100  # Cap in-memory history length


def init_session_state() -> None:
    """Initialize session state with messages, logger, and AI instance."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # type: List[Message] # type: ignore
    if "logger" not in st.session_state:
        st.session_state["logger"] = ChatLogger()
    if "ai_instance" not in st.session_state:
        try:
            ai_impl = get_ai()
            st.session_state["ai_instance"] = ai_impl
            # Warmup: optional
            getattr(ai_impl, "warmup", lambda: None)()
            st.session_state["logger"].event("ai.init", backend=ai_impl.__class__.__name__)
        except Exception as e:
            st.session_state["logger"].event("ai.init.error", error=str(e))
            st.session_state["ai_instance"] = None

def render_sidebar() -> None:
    """Render the left sidebar with branding and description."""
    from pathlib import Path as _Path

    with st.sidebar:
        # Logo
        _logo_path = _Path(__file__).parent / "assets" / "td_new_trans.png"
        if _logo_path.exists():
            st.image(str(_logo_path))

        # Title
        st.title("NextBI")

        # Description
        st.markdown(
            (
                "<div class=\"sidebar-desc\" style=\"color:#888;\">"
                "<p>NextBI is an intelligent business assistant that replaces traditional dashboards "
                "by allowing executives and business users to get instant insights through conversation. "
                "Instead of navigating reports, users simply ask questions in plain English, "
                "and NextBI generates answers directly from enterprise data—powered by Teradata’s MCP.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

def render_chat(messages: List[Message]) -> None:
    """Render chat messages in Streamlit UI."""
    import re

    def _normalize(text: str) -> str:
        s = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        return re.sub(r"\n{3,}", "\n\n", s)

    for msg in messages:
        role = msg.get("role", "ai")
        content = _normalize(msg.get("content", ""))
        chat_role = "user" if role == "user" else "assistant"
        with st.chat_message(chat_role):
            st.markdown(content)

async def generate_ai_reply(prompt: str) -> str:
    """Generate a reply from the AI backend."""
    logger = st.session_state["logger"]
    backend: AI = st.session_state["ai_instance"]

    logger.event("ai.call.start", count=str(len(st.session_state["messages"])))
    reply = await backend.generate_reply(st.session_state["messages"], context=None)
    logger.event("ai.call.end", chars=str(len(reply or "")))
    return reply


def handle_user_input(prompt: str) -> None:
    """Handle user input and update session state."""
    text = prompt.strip()
    if not text:
        return

    # Add user message
    user_msg = {
        "role": "user",
        "content": text,
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    st.session_state["messages"].append(user_msg)
    st.session_state["logger"].log(user_msg["role"], user_msg["content"])

    if st.session_state.get("ai_instance") is None:
        try:
            st.session_state["ai_instance"] = get_ai()
        except Exception as e:
            st.session_state["logger"].event("ai.init.error", error=str(e))
            st.error(f"Couldn't initialize AI backend: {e}")
            st.rerun()

    try:
        with st.spinner("Thinking..."):
            reply = asyncio.run(generate_ai_reply(text))
    except Exception as e:
        st.error(f"Couldn't get a reply: {e}")
        ai_msg = {
            "role": "ai",
            "content": f"[error] {e}",
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
    else:
        ai_msg = {
            "role": "ai",
            "content": reply if (reply and reply.strip()) else "[empty response]",
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }

    # Append AI message
    st.session_state["messages"].append(ai_msg)
    st.session_state["logger"].log(ai_msg["role"], ai_msg["content"])

    # Enforce max messages
    if len(st.session_state["messages"]) > MAX_MESSAGES:
        st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]

    st.rerun()

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="NextBI", page_icon="💬", layout="wide")
    init_session_state()

    render_sidebar()
    render_chat(st.session_state["messages"])
    prompt = st.chat_input("Type a message and press Enter")
    if prompt:
        handle_user_input(prompt)


if __name__ == "__main__":
    main()
