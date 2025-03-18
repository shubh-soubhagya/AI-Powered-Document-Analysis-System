import os
import logging
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Load environment variables
load_dotenv()

# Suppress unnecessary logs
logging.getLogger("langchain").setLevel(logging.ERROR)

# Get API Key
groq_api_key = os.getenv('GROQ_API_KEY')


# Define a function to load PDFs dynamically when uploaded
def process_uploaded_pdf(pdf_path):
    """Processes an uploaded PDF and creates a retriever."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("PDF is empty or could not be loaded.")

        # Split text into chunks
        documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)

        # Load Embedding Model
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(documents, embeddings_model)
        retriever = vectordb.as_retriever()

        # Create PDF search tool
        pdf_tool = create_retriever_tool(retriever, "pdf_search", "Search for PDF information only!")
        tools = [pdf_tool]

        # Load LLaMA 3 (via GROQ)
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

        # Create the agent
        prompt = """
        Answer the questions based on the provided PDF context only.
        Provide accurate and detailed responses strictly from the PDF content.
        <context>
        {context}
        <context>
        Questions:{input}
        {agent_scratchpad}
        """
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        return agent_executor

    except Exception as e:
        print(f"❗ Error processing PDF: {e}")
        return None
