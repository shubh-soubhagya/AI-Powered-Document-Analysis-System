import os
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv


def load_environment():
    load_dotenv()
    logging.getLogger("langchain").setLevel(logging.ERROR)
    return os.getenv('GROQ_API_KEY')

def load_and_split_pdfs(pdf_directory, chunk_size=500, chunk_overlap=0):
    loader = PyPDFDirectoryLoader(pdf_directory)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents successfully.")

    documents = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)
    print(f"✅ Successfully split into {len(documents)} text chunks.")
    return documents

def create_vector_db(documents, embeddings_path):
    embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_path)
    vectordb = FAISS.from_documents(documents, embeddings_model)
    return vectordb.as_retriever()