import streamlit as st
import uuid
import logging
import time
import os
import tempfile
from langchain_core.messages import HumanMessage

from backend import workflow  

logger = logging.getLogger("FRONTEND")
logger.setLevel(logging.INFO)

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ASK PREDUSK")

def reset_chat():
    thread_id = str(uuid.uuid4())
    st.session_state.thread_id = thread_id
    st.session_state.messages = []
    st.session_state.uploaded_document = None
    st.session_state.pasted_text = None
    st.session_state.document_loaded = False

    st.session_state.threads[thread_id] = {
        "messages": [],
        "uploaded_document": None,
        "pasted_text": None,
        "document_loaded": False,
    }

    logger.info("New chat created: %s", thread_id)

if "threads" not in st.session_state:
    st.session_state.threads = {}

if "thread_id" not in st.session_state:
    reset_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

st.sidebar.header("🧠 Chat Controls")

if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.markdown("### 💬 Previous Chats")

for tid in st.session_state.threads:
    label = f"Chat {tid[:8]}"
    if st.sidebar.button(label, key=f"load-{tid}"):
        data = st.session_state.threads[tid]

        st.session_state.thread_id = tid
        st.session_state.messages = data["messages"]
        st.session_state.uploaded_document = data["uploaded_document"]
        st.session_state.pasted_text = data["pasted_text"]
        st.session_state.document_loaded = data["document_loaded"]

        logger.info("Switched to chat: %s", tid)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📄 Knowledge Source")

st.sidebar.info(
    "⚠️ **One document per chat session**\n\n"
    "To upload a new file, click **New Chat**."
)

uploader_disabled = st.session_state.document_loaded

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    disabled=uploader_disabled
)

st.sidebar.markdown("**OR**")

pasted_text_input = st.sidebar.text_area(
    "Paste text here",
    height=180,
    disabled=uploader_disabled
)

confirm_text = st.sidebar.button(
    "✅ Confirm pasted text",
    disabled=uploader_disabled
)

if uploaded_file and not st.session_state.document_loaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        file_path = tmp.name

    st.session_state.uploaded_document = file_path
    st.session_state.pasted_text = None
    st.session_state.document_loaded = True

    st.session_state.threads[st.session_state.thread_id].update({
        "uploaded_document": file_path,
        "pasted_text": None,
        "document_loaded": True,
    })

    logger.info("PDF saved at %s", file_path)
    st.sidebar.success("PDF uploaded and locked for this chat")

if confirm_text and pasted_text_input.strip() and not st.session_state.document_loaded:
    st.session_state.pasted_text = pasted_text_input
    st.session_state.uploaded_document = None
    st.session_state.document_loaded = True

    st.session_state.threads[st.session_state.thread_id].update({
        "uploaded_document": None,
        "pasted_text": pasted_text_input,
        "document_loaded": True,
    })

    logger.info("Pasted text confirmed")
    st.sidebar.success("Text saved and locked for this chat")

for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"],
        avatar="🤖" if msg["role"] == "assistant" else None
    ):
        st.markdown(msg["content"])


query = st.chat_input("Ask a question about the document...")

if query:
    if not st.session_state.document_loaded:
        st.warning("Please upload a PDF or paste text first.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    st.session_state.threads[st.session_state.thread_id]["messages"] = st.session_state.messages

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            logger.info("Invoking backend")

            start = time.time()

            result = workflow.invoke(
                {
                    "query": query,
                    "uploaded_document": st.session_state.uploaded_document,
                    "pasted_text": st.session_state.pasted_text,
                    "messages": [HumanMessage(content=query)]
                },
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }
            )

            elapsed = time.time() - start

            answer_block = result.get("answer", {})
            answer_text = answer_block.get("Answer", "No answer returned.")
            citations = answer_block.get("Citations", [])

            context_docs = answer_block.get("context", [])

            st.markdown("### 🧠 Answer")
            st.markdown(answer_text)

            if citations:
                st.markdown("### 📚 Citations")
                for c in citations:
                    st.markdown(
                        f"- **[{c['id']}]** {c.get('source','unknown')} "
                        f"({c.get('section','unknown')})"
                    )
            if context_docs:
                with st.expander("📄 Retrieved Context Chunks"):
                    for doc in context_docs:
                        st.markdown(f"**Chunk [{doc['id']}]**")
                        st.caption(f"Source: {doc['source']} | Section: {doc['section']}")
                        st.markdown(doc["content"])
                        st.markdown("---")

            input_tokens = len(query) // 4
            output_tokens = len(answer_text) // 4

            st.caption(
                f"⏱ {elapsed:.2f}s | 🔢 Tokens (est.): "
                f"{input_tokens + output_tokens}"
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text}
    )

    st.session_state.threads[st.session_state.thread_id]["messages"] = st.session_state.messages

st.markdown("---")
st.caption("RAG Chatbot • One document per chat • New Chat to change source")
