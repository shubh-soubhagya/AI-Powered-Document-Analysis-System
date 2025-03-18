import os
import logging
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Load environment variables
load_dotenv()

# Suppress unnecessary logs
logging.getLogger("langchain").setLevel(logging.ERROR)

# Get API Key
groq_api_key = os.getenv('GROQ_API_KEY')

# Ensure the correct PDF directory path
pdf_dir = os.path.abspath("../pdf_app_test")

# Load PDFs
try:
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents successfully.")
except Exception as e:
    print(f"❗ Error loading PDFs: {e}")
    exit(1)

# Split text into chunks
documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)
print(f"✅ Successfully split into {len(documents)} text chunks.")

# Load Embedding Model (Auto-download if not found)
try:
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents, embeddings_model)
    retriever = vectordb.as_retriever()
except Exception as e:
    print(f"❗ Error initializing embeddings: {e}")
    exit(1)

# Create PDF search tool
pdf_tool = create_retriever_tool(retriever, "pdf_search", "Search for PDF information only!")
tools = [pdf_tool]

# Load LLaMA 3 (via GROQ)
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
except Exception as e:
    print(f"❗ Error initializing LLM: {e}")
    exit(1)

# Prompt Setup
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided PDF context only.
Provide accurate and detailed responses strictly from the PDF content.
<context>
{context}
<context>
Questions:{input}
{agent_scratchpad}
"""
)

# Create the agent
try:
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
except Exception as e:
    print(f"❗ Error setting up agent: {e}")
    exit(1)

# Interactive Query Loop
while True:
    query = input("Input your query here (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting... Goodbye!")
        break

    start_time = time.time()
    try:
        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })
        print(f"\n🟩 Final Output:\n{response['output']}")
    except Exception as e:
        print(f"❗ Error processing query: {e}")

    print(f"⏱️ Total Response Time: {time.time() - start_time:.2f} seconds")
