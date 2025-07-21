from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Upload & Load raw PDF(s)
pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# ✅ Step 1: Load your PDF into `documents`
loader = PyPDFLoader("universal declaration.pdf")  # Change this to your actual file
documents = loader.load()

# ✅ Step 2: Chunk the loaded documents
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

text_chunks = create_chunks(documents)

# ✅ Optional: print first chunk
print(text_chunks[0].page_content)

# ✅ Step 3: Setup HuggingFace Embeddings Model (Cloud-compatible)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Step 4: Store embeddings in FAISS vector store
FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model())
faiss_db.save_local(FAISS_DB_PATH)


