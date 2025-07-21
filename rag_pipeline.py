# rag_pipeline.py

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from vectordb import faiss_db  # Ensure this is correctly defined

# ✅ Setup Groq LLM
llm_model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768"  # or llama3-8b-8192, gemma-7b-it, deepseek-coder
)

# ✅ HuggingFace Embeddings (used in vectordb.py)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Retrieve documents (simple search from FAISS DB)
def retrieve_docs(query: str) -> list[Document]:
    return faiss_db.similarity_search(query)

# ✅ Prepare context from documents
def get_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)

# ✅ Custom prompt
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Question: {question}
Context: {context}

Answer:
"""

# ✅ Use prompt + Groq LLM to answer the query
def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

