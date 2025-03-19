from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



def load_pdfs(pdf_path):
    loader = PyPDFDirectoryLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents successfully.")
    return docs

def split_documents(docs, chunk_size=500, chunk_overlap=0):
    documents = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
    print(f"✅ Successfully split into {len(documents)} text chunks.")
    return documents

def create_vectorstore(documents, model_path):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_path)
    return FAISS.from_documents(documents, embeddings_model)