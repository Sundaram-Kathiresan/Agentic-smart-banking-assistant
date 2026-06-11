import requests
import streamlit as st
from datetime import datetime
import json
 
UPLOAD_URL = "http://localhost:8000/api/v1/rag/upload"
QUERY_URL = "http://localhost:8000/api/v1/rag/query/stream"
 
# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Smart Banking Assistant", page_icon="🤖", layout="wide")
 
# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown(
    """
<style>
 
.main {
    background-color: #0E1117;
}
 
.chat-title {
    text-align:center;
    font-size:32px;
    font-weight:bold;
    color:white;
}
 
.sidebar-title {
    font-size:22px;
    font-weight:bold;
}
 
.stChatMessage {
    border-radius: 10px;
    padding: 10px;
}
 
.user-msg {
    background:#1f2937;
    padding:10px;
    border-radius:10px;
}
 
.bot-msg {
    background:#111827;
    padding:10px;
    border-radius:10px;
}
 
</style>
""",
    unsafe_allow_html=True,
)
 
# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
 
if "current_chat" not in st.session_state:
    chat_name = f"Chat {datetime.now().strftime('%H:%M:%S')}"
    st.session_state.current_chat = chat_name
    st.session_state.chats[chat_name] = []
 
# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat = f"Chat {datetime.now().strftime('%H:%M:%S')}"
        st.session_state.chats[new_chat] = []
        st.session_state.current_chat = new_chat
        st.rerun()
    st.divider()
 
    for chat_name in reversed(list(st.session_state.chats.keys())):
        if st.button(chat_name, use_container_width=True, key=chat_name):
            st.session_state.current_chat = chat_name
            st.rerun()
 
# ---------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------
st.markdown(
    "<div class='chat-title'>📄 RAG PDF Assistant</div>", unsafe_allow_html=True
)
 
st.write("")
st.write("")
 
# ---------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------
with st.expander("📂 Upload PDF", expanded=False):
    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"])
    if uploaded_file:
        st.success(f"Selected: {uploaded_file.name}")
        if st.button("Upload PDF"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                with st.spinner("Uploading PDF..."):
                    response = requests.post(UPLOAD_URL, files=files)
                if response.status_code == 200:
                    st.success("PDF uploaded successfully")
                    st.json(response.json())
                else:
                    st.error(response.text)
            except Exception as e:
                st.error(str(e))
 
# ---------------------------------------------------
# CHAT AREA
# ---------------------------------------------------
messages = st.session_state.chats[st.session_state.current_chat]
 
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
query = st.chat_input("Ask anything related to NorthStar bank")
 
if query:
    messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    with st.chat_message("user"):
        st.markdown(query)
    answer = ""
    try:
        response = requests.post(
            QUERY_URL,
            json={
                "question": query,
                # CHANGE: Send session_id so the server knows which chat
                # session this message belongs to. Each chat tab in the sidebar
                # has a unique name (e.g. "Chat 12:00:00"), so each tab gets
                # its own isolated memory on the server side.
                "session_id": st.session_state.current_chat,
            },
            stream=True
        )
 
        # ── Guardrail / server error — show the message, don't stream ────────
        if response.status_code != 200:
            error_message = "An error occurred. Please try again."
            try:
                detail = response.json().get("detail", {})
                if isinstance(detail, dict):
                    # GuardrailViolation shape: {"guardrail": "...", "message": "..."}
                    error_message = detail.get("message", error_message)
                elif isinstance(detail, str):
                    error_message = detail
            except Exception:
                error_message = response.text or error_message
 
            with st.chat_message("assistant"):
                st.warning(error_message)
 
            messages.append({"role": "assistant", "content": error_message})
            st.rerun()
 
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
 
                # print("RAW:", decoded)
 
                if not decoded.startswith("data:"):
                    continue
 
                data = decoded.replace("data:", "").strip()
 
                # print("DATA:", data)
 
                if data == "[DONE]":
                    break
 
                try:
                    payload = json.loads(data)
                    token = payload.get("token", "")
                    answer += token
                    placeholder.markdown(answer + "▌")
                except Exception as e:
                    print("JSON ERROR:", e)
                    print("FAILED DATA:", data)
 
            placeholder.markdown(answer)
 
    except Exception as e:
        answer = str(e)
 
    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
 
    st.rerun()