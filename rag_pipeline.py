# rag_pipeline.py

import os
from dotenv import load_dotenv
load_dotenv()

from typing import List
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document, SystemMessage, HumanMessage
from vectordb import faiss_db  # Ensure this is correctly defined and initialized

# -----------------------------------------------------------------------------
# LLM SETUP (GROQ with DeepSeek)
# -----------------------------------------------------------------------------
# You can switch to other Groq-supported models (e.g., mixtral, llama3) if needed
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "deepseek-coder:6.7b")

llm_model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=GROQ_MODEL_NAME,
    temperature=0,
    max_tokens=None,
)

# -----------------------------------------------------------------------------
# EMBEDDINGS (for FAISS embedding model — used in vectordb.py)
# -----------------------------------------------------------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------------------------------------------------------
# RETRIEVAL
# -----------------------------------------------------------------------------
DEFAULT_K = 4

def retrieve_docs(query: str, k: int = DEFAULT_K) -> List[Document]:
    """Retrieve top-k relevant document chunks from FAISS vector store."""
    return faiss_db.similarity_search(query, k=k)

# -----------------------------------------------------------------------------
# CONTEXT BUILDING
# -----------------------------------------------------------------------------
MAX_CONTEXT_CHARS = 12000

def get_context(documents: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join document chunks into a single context string (limit by char count)."""
    if not documents:
        return ""
    
    parts = []
    total_chars = 0

    for i, doc in enumerate(documents, start=1):
        chunk = doc.page_content.strip()
        if not chunk:
            continue

        header = f"--- Document {i} ---\n"
        entry = header + chunk + "\n"

        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                parts.append(entry[:remaining])
            break

        parts.append(entry)
        total_chars += len(entry)

    return "".join(parts)

# -----------------------------------------------------------------------------
# QUERY ANSWERING
# -----------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are 'AI Lawyer', a helpful legal assistant. "
    "Only answer using the context provided from the document. "
    "If the answer is not in the context, say: 'I don't know from the provided document.' "
    "Cite legal articles or clauses if mentioned in the context. Be concise and factual."
)

def answer_query(documents: List[Document], model: ChatGroq, query: str) -> str:
    """Use the Groq-hosted DeepSeek model to answer based on context."""
    context = get_context(documents)
    
    if not context:
        return (
            "I couldn't find relevant information in the uploaded document. "
            "Please upload a PDF that contains material related to your question."
        )

    user_content = (
        f"Question:\n{query}\n\n"
        f"Context (from PDF):\n{context}\n\n"
        "Answer:"
    )

    messages = [
        SystemMessage(content=SYSTEM_MSG),
        HumanMessage(content=user_content)
    ]

    response = model.invoke(messages)
    return response.content


