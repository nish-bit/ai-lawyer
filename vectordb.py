from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#Step 1: Upload & Load raw PDF(s)

pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


#file_path = 'universal declaration.pdf'
#documents = load_pdf(file_path)
#print("PDF pages: ",len(documents))

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Step 1: Load your PDF into `documents`
loader = PyPDFLoader("universal declaration.pdf")  # Change this to your actual PDF file name
documents = loader.load()

# ✅ Step 2: Chunk the loaded documents
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)

# ✅ Optional: print first chunk
print(text_chunks[0].page_content)

#print("Chunks count: ", len(text_chunks))


#Step 3: Setup Embeddings Model (Use DeepSeek R1 with Ollama)
ollama_model_name="deepseek-coder:6.7b"
def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings

#Step 4: Index Documents **Store embeddings in FAISS (vector store)
FAISS_DB_PATH="vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)