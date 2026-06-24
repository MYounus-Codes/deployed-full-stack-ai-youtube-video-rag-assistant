import streamlit as st

from ingest import ingest_url
from retriever import get_retriever

from langchain_groq import ChatGroq
from langchain_classic.memory import (
    ConversationBufferMemory,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
)

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 YouTube RAG Assistant")

if "videos" not in st.session_state:
    st.session_state.videos = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = (
        ConversationBufferMemory(
            return_messages=True,
            memory_key="history",
        )
    )

#################################################
# Sidebar
#################################################

with st.sidebar:
    st.header("Index Videos")

    url = st.text_input(
        "YouTube Video or Playlist URL"
    )

    if st.button(
        "Index"
    ):
        with st.spinner(
            "Indexing..."
        ):
            st.session_state.videos = (
                ingest_url(url)
            )

        st.success(
            "Videos indexed!"
        )

#################################################
# Video Selector
#################################################

if st.session_state.videos:

    options = {
        video["title"]: video
        for video
        in st.session_state.videos
    }

    selected_title = st.selectbox(
        "Choose Video",
        list(options.keys()),
    )

    selected_video = options[
        selected_title
    ]

    retriever = get_retriever(
        selected_video["video_id"]
    )

else:
    st.info(
        "Index a video first."
    )
    st.stop()

#################################################
# Chat History
#################################################

for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"]
    ):
        st.markdown(
            msg["content"]
        )

#################################################
# Chat
#################################################

question = st.chat_input(
    "Ask something..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    history = (
        st.session_state.memory
        .load_memory_variables({})
        ["history"]
    )

    prompt = (
        ChatPromptTemplate.from_template(
            """
You are a helpful YouTube AI Assistant.

History:
{history}

Context:
{context}

Question:
{question}

Answer conversationally.

Cite sources like:
[Source 1]
"""
        )
    )

    messages = prompt.invoke(
        {
            "history": history,
            "context": context,
            "question": question,
        }
    )

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.2,
        streaming=True,
    )

    with st.chat_message(
        "assistant"
    ):
        placeholder = st.empty()

        answer = ""

        for chunk in llm.stream(
            messages
        ):
            answer += chunk.content
            placeholder.markdown(
                answer + "▌"
            )

        placeholder.markdown(
            answer
        )

        with st.expander(
            "📚 Sources"
        ):
            for i, doc in enumerate(
                docs
            ):
                timestamp = int(
                    doc.metadata.get(
                        "timestamp",
                        0,
                    )
                )

                m = timestamp // 60
                s = timestamp % 60

                url = (
                    f"{doc.metadata['source']}"
                    f"&t={timestamp}s"
                )

                st.markdown(
                    f"""
**Source {i+1}**

🎬 {doc.metadata['video_title']}

⏱️ {m}:{s:02d}

🔗 {url}
"""
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.session_state.memory.save_context(
        {"input": question},
        {"output": answer},
    )