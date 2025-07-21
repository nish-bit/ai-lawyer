# rag_pipeline.py

import os
from dotenv import load_dotenv
load_dotenv()

from typing import List

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document, SystemMessage, HumanMessage
from vectordb import faiss_db  # must define & persist FAISS DB elsewhere

# -----------------------------------------------------------------------------
# LLM SETUP (GROQ)
# -----------------------------------------------------------------------------
# Use a smaller, broadly available model while debugging.
# You can switch back to "mixtral-8x7b-32768" once everything works.
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")

llm_model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=GROQ_MODEL_NAME,
    temperature=0,           # deterministic for legal answers
    max_tokens=None,         # let API decide; you may cap if needed
)

# -----------------------------------------------------------------------------
# EMBEDDINGS (used in vectordb build step — exposed for completeness)
# -----------------------------------------------------------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------------------------------------------------------
# RETRIEVAL
# -----------------------------------------------------------------------------
DEFAULT_K = 4  # number of chunks to retrieve
def retrieve_docs(query: str, k: int = DEFAULT_K) -> List[Document]:
    """Return top-k most similar chunks from FAISS."""
    return faiss_db.similarity_search(query, k=k)

# -----------------------------------------------------------------------------
# CONTEXT PREP
# -----------------------------------------------------------------------------
MAX_CONTEXT_CHARS = 12000  # ~3K tokens rough safety budget; tune as needed

def get_context(documents: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join docs into a single context string, truncated to max_chars."""
    if not documents:
        return ""
    parts = []
    total = 0
    for i, doc in enumerate(documents, start=1):
        chunk = doc.page_content.strip()
        if not chunk:
            continue
        header = f"--- Document {i} ---\n"
        entry = header + chunk + "\n"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(entry[:remaining])
            break
        parts.append(entry)
        total += len(entry)
    return "".join(parts)

# -----------------------------------------------------------------------------
# ANSWER GENERATION
# -----------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are 'AI Lawyer', a helpful legal explainer. "
    "Answer using ONLY the provided context chunks. "
    "If the answer is not in the context, say: 'I don't know from the provided document.' "
    "Cite article numbers or clauses if present in the text. Respond concisely."
)

def answer_query(documents: List[Document], model: ChatGroq, query: str) -> str:
    """Generate an answer from retrieved docs using Groq chat API."""
    context = get_context(documents)
    if not context:
        # No supporting docs — fail gracefully
        msg = (
            "I couldn't find relevant information in the uploaded document. "
            "Please upload a PDF that contains material related to your question."
        )
        return msg

    user_content = (
        f"Question:\n{query}\n\n"
        f"Context (extracts from uploaded PDF):\n{context}\n\n"
        "Answer:"
    )

    messages = [
        SystemMessage(content=SYSTEM_MSG),
        HumanMessage(content=user_content),
    ]

    # Groq returns an AIMessage; .content holds the text
    response = model.invoke(messages)
    return response.content



