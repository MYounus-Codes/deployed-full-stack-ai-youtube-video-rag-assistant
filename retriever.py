import pickle
import os

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_classic.retrievers import (
    EnsembleRetriever,
)
from langchain_community.retrievers import (
    BM25Retriever,
)
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)
from langchain_pinecone import (
    PineconeVectorStore,
)

load_dotenv()

INDEX_NAME = "youtube-rag-assistant"

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

index = pc.Index(INDEX_NAME)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)


def get_retriever(video_id):
    with open(
        f"data/bm25/{video_id}.pkl",
        "rb",
    ) as f:
        docs = pickle.load(f)

    bm25 = (
        BM25Retriever.from_documents(
            docs
        )
    )
    bm25.k = 5

    dense = (
        vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.5,
                "namespace": video_id,
            },
        )
    )

    return EnsembleRetriever(
        retrievers=[
            bm25,
            dense,
        ],
        weights=[0.4, 0.6],
    )