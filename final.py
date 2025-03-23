import os
import logging
import time
import fitz  # PyMuPDF for PDF text extraction
import re
import spacy
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv
from collections import Counter
from plagiarism.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plagiarism.jaccard_similarity import jaccard_similarity
from plagiarism.lcs import lcs
from plagiarism.lsh import lsh_similarity
from plagiarism.n_gram_similarity import n_gram_similarity
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool

# Load Environment Variables
load_dotenv()
logging.getLogger("langchain").setLevel(logging.ERROR)

groq_api_key = os.getenv('GROQ_API_KEY')
PDF_DIRECTORY = r"C:\Users\hp\Desktop\ps_sol\AI-Powered-Document-Analysis-System\pdf_app_test"
EMBEDDING_MODEL_PATH = r"C:\Users\hp\Desktop\ps_sol\models\all-MiniLM-L6-v2"

# PDF Extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Text Preprocessing
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

# Prediction Function
def predict_category(nlp, text):
    doc = nlp(text)
    return max(doc.cats, key=doc.cats.get)

# Plagiarism Worker
def plagiarism_worker(doc1, doc2, name1, name2):
    return {
        "File 1": name1,
        "File 2": name2,
        "Cosine (TF-IDF)": f"{cosine_similarity_tfidf(doc1, doc2) * 100:.2f}%",
        "Cosine (Count)": f"{cosine_similarity_count(doc1, doc2) * 100:.2f}%",
        "Jaccard": f"{jaccard_similarity(doc1, doc2) * 100:.2f}%",
        "LCS": f"{lcs(doc1, doc2) * 100:.2f}%",
        "LSH": f"{lsh_similarity(doc1, doc2) * 100:.2f}%",
        "N-Gram": f"{n_gram_similarity(doc1, doc2) * 100:.2f}%"
    }

# Plagiarism Check
def check_plagiarism(docs):
    results = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            results.append(plagiarism_worker(docs[i]["text"], docs[j]["text"], docs[i]["name"], docs[j]["name"]))
    return results

# Load PDFs
loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
docs = loader.load()
print(f"✅ Loaded {len(docs)} documents successfully.")

documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)
print(f"✅ Successfully split into {len(documents)} text chunks.")

# Embedding Model
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
vectordb = FAISS.from_documents(documents, embeddings_model)
retriever = vectordb.as_retriever()

# Tool Setup
pdf_tool = create_retriever_tool(retriever, "pdf_search", "Search for PDF information only!")
tools = [pdf_tool]

# Load LLaMA 3 (via GROQ)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

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

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# User Selects PDF
pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
print("Available PDFs:")
for idx, pdf in enumerate(pdf_files):
    print(f"{idx + 1}. {pdf}")

choice = int(input("Select a PDF (enter number): ")) - 1
selected_pdf = pdf_files[choice]
selected_text = extract_text_from_pdf(os.path.join(PDF_DIRECTORY, selected_pdf))
preprocessed_text = preprocess_text(selected_text)

# Load NLP Model for Prediction
nlp = spacy.load(r"C:\Users\hp\Desktop\ps_sol\AI-Powered-Document-Analysis-System\model_training\trained_model")
predicted_category = predict_category(nlp, preprocessed_text)
print(f"Predicted Category: {predicted_category}")

# Run Plagiarism Check
# print("\n🔎 Running Plagiarism Check...")
# doc_data = [{"name": pdf, "text": extract_text_from_pdf(os.path.join(PDF_DIRECTORY, pdf))} for pdf in pdf_files]
# plagiarism_results = check_plagiarism(doc_data)

# # Display Plagiarism Report
# if plagiarism_results:
#     print("\n📊 Plagiarism Report:")
#     print(tabulate(plagiarism_results, headers="keys", tablefmt="grid"))
# else:
#     print("No plagiarism detected.")

# # Chatbot Interaction
# while True:
#     query = input("\nInput your query here: ")
#     if query.lower() in ["exit", "quit", "q"]:
#         print("Exiting... Goodbye!")
#         break

#     start_time = time.time()
#     try:
#         response = agent_executor.invoke({
#             "input": query,
#             "context": "",
#             "agent_scratchpad": ""
#         })
#         print(f"\n🟩 Final Output:\n{response['output']}")
#         print(f"⏱️ Total Response Time: {time.time() - start_time:.2f} seconds")
#     except Exception as e:
#         print(f"❗ Error: {e}")
