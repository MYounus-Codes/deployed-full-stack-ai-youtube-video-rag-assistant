from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_classic.memory import (
    ConversationBufferMemory
)
from langchain_core.prompts import (
    ChatPromptTemplate
)

from retriever import (
    get_retriever
)

load_dotenv()

video_id = input(
    "Video ID: "
)

retriever = get_retriever(
    video_id
)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history",
)

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.2,
    streaming=True,
)

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful YouTube AI Assistant.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer the question only from the context.

If the answer is not available,
say:
"I couldn't find that information
in the video."

Always cite sources:
[Source 1]
[Source 2]
"""
)

print(
    "\nYouTube RAG Assistant"
)

while True:

    question = input(
        "\nQuestion: "
    )

    if question.lower() == "exit":
        break

    docs = retriever.invoke(
        question
    )

    context = []

    for i, doc in enumerate(docs):

        timestamp = int(
            doc.metadata.get(
                "timestamp",
                0,
            )
        )

        m = timestamp // 60
        s = timestamp % 60

        context.append(
            f"""
[Source {i+1}]
Time: {m}:{s:02d}

{doc.page_content}
"""
        )

    context = "\n".join(
        context
    )

    history = (
        memory.load_memory_variables(
            {}
        )["history"]
    )

    messages = prompt.invoke(
        {
            "history": history,
            "context": context,
            "question": question,
        }
    )

    answer = ""

    print("\nAnswer:\n")

    for chunk in llm.stream(
        messages
    ):
        print(
            chunk.content,
            end="",
            flush=True,
        )
        answer += chunk.content

    print("\n")

    print("Sources:\n")

    shown = set()

    for i, doc in enumerate(docs):

        timestamp = int(
            doc.metadata.get(
                "timestamp",
                0,
            )
        )

        if timestamp in shown:
            continue

        shown.add(
            timestamp
        )

        m = timestamp // 60
        s = timestamp % 60

        url = (
            f"{doc.metadata['source']}"
            f"&t={timestamp}s"
        )

        print(
            f"[Source {i+1}] "
            f"{m}:{s:02d}"
        )

        print(url)
        print(
            f"Video: "
            f"{doc.metadata['video_title']}"
        )
        print()

    memory.save_context(
        {"input": question},
        {"output": answer},
    )