import os
import pickle

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)
from langchain_pinecone import (
    PineconeVectorStore,
)
from pinecone import (
    Pinecone,
    ServerlessSpec,
)

from utils import (
    get_playlist_videos,
    get_video_title,
    get_transcript,
    extract_video_id,
)

load_dotenv()

INDEX_NAME = "youtube-rag-assistant"


def ingest_url(url):
    os.makedirs(
        "data/bm25",
        exist_ok=True,
    )

    embeddings = (
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
    )

    pc = Pinecone(
        api_key=os.getenv(
            "PINECONE_API_KEY"
        )
    )

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    index = pc.Index(INDEX_NAME)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

    videos = get_playlist_videos(url)

    indexed_videos = []

    for video_url in videos:

        title = get_video_title(
            video_url
        )

        video_id = extract_video_id(
            video_url
        )

        transcript = get_transcript(
            video_id
        )

        documents = []

        for seg in transcript:
            documents.append(
                Document(
                    page_content=seg.text,
                    metadata={
                        "video_id": video_id,
                        "video_title": title,
                        "source": video_url,
                        "timestamp": seg.start,
                    },
                )
            )

        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
        )

        chunks = (
            splitter.split_documents(
                documents
            )
        )

        vector_store.add_documents(
            chunks,
            namespace=video_id,
        )

        with open(
            f"data/bm25/{video_id}.pkl",
            "wb",
        ) as f:
            pickle.dump(
                chunks,
                f,
            )

        indexed_videos.append(
            {
                "title": title,
                "video_id": video_id,
                "url": video_url,
            }
        )

    return indexed_videos