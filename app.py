import streamlit as st
from main import AppContext
import logging
import re

ANSI_ESCAPE = re.compile(r'\x1b\[([0-9]+)(;[0-9]+)*m')

st.set_page_config(page_title="Chat UI", layout="centered")
st.title("💬 ReAct AI Assistant")
st.info("✏️ Write **exit** to finish the conversation.", icon="ℹ️")

if "app_ctx" not in st.session_state:
    st.session_state["app_ctx"] = AppContext()
if "messages" not in st.session_state:
    st.session_state["messages"] = []
log_placeholder = st.empty()

class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        raw_log_msg = self.format(record)
        log_msg = ANSI_ESCAPE.sub('', raw_log_msg)
        if "log_buffer" not in st.session_state:
            st.session_state["log_buffer"] = []
        if "log_seen" not in st.session_state:
            st.session_state["log_seen"] = set()
        if log_msg not in st.session_state["log_seen"]:
            st.session_state["log_buffer"].append(log_msg)
            st.session_state["log_seen"].add(log_msg)
        logs = "\n".join(st.session_state["log_buffer"][-20:])
        log_placeholder.code(logs, language="text")

logger = logging.getLogger()
if not any(isinstance(h, StreamlitLogHandler) for h in logger.handlers):
    log_handler = StreamlitLogHandler()
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(log_handler)

user_input = st.chat_input("Type your message...")

if user_input:
    if user_input.strip().lower() == "exit":
        st.session_state["app_ctx"].exit_session()
        st.success("✅ Conversation ended. Thank you for chatting!")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        result = st.session_state["app_ctx"].handle_query(user_input)
        assistant_reply = result["final_answer"]
        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})

for msg in st.session_state["messages"]: # Evitar mensajes de logger duplicados
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
