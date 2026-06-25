import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ingest import ensure_pinecone_index, ingest_url
from retriever import get_retriever

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide",
)

st.title("YouTube RAG Assistant")

INDEX_NAME = "youtube-rag-assistant"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


@st.cache_resource
def get_vector_store(_embeddings, _pc):
    ensure_pinecone_index(_pc)
    index = _pc.Index(INDEX_NAME)
    return PineconeVectorStore(index=index, embedding=_embeddings)


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.2,
        streaming=True,
    )


embeddings = get_embeddings()
pc = get_pinecone_client()
vector_store = get_vector_store(embeddings, pc)
llm = get_llm()

if "videos" not in st.session_state:
    st.session_state.videos = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",
    )

with st.sidebar:
    st.header("Index Videos")

    url = st.text_input("YouTube Video or Playlist URL")

    if st.button("Index"):
        with st.status("Indexing...", expanded=True) as status:
            try:
                st.write("Getting video info...")
                st.write("Downloading transcript...")
                st.write("Creating chunks...")
                st.write("Generating embeddings...")
                st.write("Uploading to Pinecone...")

                st.session_state.videos = ingest_url(url, embeddings, vector_store)

                status.update(label="Complete!", state="complete")
                st.success("Videos indexed!")

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(str(e))

if st.session_state.videos:
    options = {video["title"]: video for video in st.session_state.videos}
    selected_title = st.selectbox("Choose Video", list(options.keys()))
    selected_video = options[selected_title]
    retriever = get_retriever(selected_video["video_id"])
else:
    st.info("Index a video first.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    history = st.session_state.memory.load_memory_variables({})["history"]

    prompt = ChatPromptTemplate.from_template(
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

    messages = prompt.invoke(
        {
            "history": history,
            "context": context,
            "question": question,
        }
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""

        for chunk in llm.stream(messages):
            answer += chunk.content
            placeholder.markdown(answer + "▌")

        placeholder.markdown(answer)

        with st.expander("Sources"):
            for i, doc in enumerate(docs):
                timestamp = int(doc.metadata.get("timestamp", 0))
                m = timestamp // 60
                s = timestamp % 60
                url = f"{doc.metadata['source']}&t={timestamp}s"
                st.markdown(
                    f"""
**Source {i+1}**

Video: {doc.metadata['video_title']}

Timestamp: {m}:{s:02d}

URL: {url}
"""
                )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.memory.save_context({"input": question}, {"output": answer})
