import os
import pickle

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import (
    get_playlist_videos,
    get_video_title,
    get_transcript,
    extract_video_id,
)

INDEX_NAME = "youtube-rag-assistant"


def ensure_pinecone_index(pc):
    if INDEX_NAME not in pc.list_indexes().names():
        from pinecone import ServerlessSpec

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )


def ingest_url(url, embeddings, vector_store):
    os.makedirs("data/bm25", exist_ok=True)

    videos = get_playlist_videos(url)
    print(f"Processing {len(videos)} video(s)")

    indexed_videos = []

    for video_url in videos:
        print(f"Processing URL: {video_url}")

        title = get_video_title(video_url)
        print(f"Title: {title}")

        video_id = extract_video_id(video_url)

        print("Extracting transcript...")
        transcript = get_transcript(video_id)

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

        print("Creating chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        print("Creating embeddings...")
        print("Uploading vectors...")
        vector_store.add_documents(chunks, namespace=video_id)

        with open(f"data/bm25/{video_id}.pkl", "wb") as f:
            pickle.dump(chunks, f)

        indexed_videos.append(
            {
                "title": title,
                "video_id": video_id,
                "url": video_url,
            }
        )

        print(f"Finished: {title}")

    return indexed_videos
