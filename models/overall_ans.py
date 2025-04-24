import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OpenAIEmbeddings
import time
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max=10000)
wiki = WikipediaQueryRun(api_wrapper = wiki_wrapper)

loader = PyPDFDirectoryLoader(r"AI-Powered-Document-Analysis-System\pdf_app_test")
docs = loader.load()

documents = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=25).split_documents(docs)

from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the local model path
embeddings_model = HuggingFaceEmbeddings(model_name=r"C:\Users\hp\Desktop\ps_sol\models\all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(documents, embeddings_model)

retriever = vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
pdf_tool = create_retriever_tool(retriever, "pdf_search",
                     "Search for information for any questions about data, you must use this tool first!")

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=10000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools = [wiki, arxiv, pdf_tool]

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate and a detailed response based on the question
<context>
{context}
<context>
Questions:{input}
{agent_scratchpad}
"""
)

from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

while True:
    query = input("Input your query here: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting... Goodbye!")
        break
    start_overall = time.time()
    start_llm = time.process_time()
    try:
        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })
        response_time_overall = time.time() - start_overall
        response_time_llm = time.process_time() - start_llm
        print(response['output'])
        print(f"Overall Response Time: {response_time_overall:.2f} seconds")
        print(f"LLM Response Time: {response_time_llm:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")