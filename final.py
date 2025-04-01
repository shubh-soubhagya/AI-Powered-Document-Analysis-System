import os
import fitz  # PyMuPDF for PDF text extraction
import re
import spacy
import logging
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from plagiarism.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plagiarism.jaccard_similarity import jaccard_similarity
from plagiarism.lcs import lcs
from plagiarism.lsh import lsh_similarity
from plagiarism.n_gram_similarity import n_gram_similarity
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

# from sentence_transformers import SentenceTransformer
#
# # Define model name
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
#
# # Load the model
# model = SentenceTransformer(model_name)
#
# # Save the model locally
# model.save("models/all-MiniLM-L6-v2")

# Load Environment Variables
load_dotenv()
logging.getLogger("langchain").setLevel(logging.ERROR)

# Flask Setup
app = Flask(__name__)
CORS(app)

# Paths & API Key
PDF_DIRECTORY = "pdf_app_test"
MODEL_PATH = "models/all-MiniLM-L6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure directories exist
os.makedirs(PDF_DIRECTORY, exist_ok=True)

# Load NLP Model
nlp = spacy.load("model_training/trained_model")

# Function: Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Function: Preprocess text
def preprocess_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

# Function: Predict Category
def predict_category(text):
    doc = nlp(text)
    return max(doc.cats, key=doc.cats.get)

# Function: Check Plagiarism
def check_plagiarism(docs):
    results = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            results.append({
                "File 1": docs[i]["name"],
                "File 2": docs[j]["name"],
                "Cosine (TF-IDF)": f"{cosine_similarity_tfidf(docs[i]['text'], docs[j]['text']) * 100:.2f}%",
                "Jaccard": f"{jaccard_similarity(docs[i]['text'], docs[j]['text']) * 100:.2f}%",
                "LCS": f"{lcs(docs[i]['text'], docs[j]['text']) * 100:.2f}%",
                "LSH": f"{lsh_similarity(docs[i]['text'], docs[j]['text']) * 100:.2f}%",
                "N-Gram": f"{n_gram_similarity(docs[i]['text'], docs[j]['text']) * 100:.2f}%"
            })
    return results

# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Ensure the directory exists before saving the files
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# API Endpoint to upload directory and process PDFs
@app.route("/api/upload-directory", methods=["POST"])
def upload_directory():
    files = request.files.getlist("pdfs")
    doc_data = []

    # Ensure the directory exists
    upload_dir = os.path.join(os.path.dirname(__file__), "temp_pdf_path")

    ensure_directory_exists(upload_dir)

    for file in files:
        pdf_path = os.path.join(upload_dir, file.filename)

        # Save the uploaded file
        try:
            file.save(pdf_path)  # Save the file correctly
            text = extract_text_from_pdf(pdf_path)
            category = predict_category(preprocess_text(text))
            doc_data.append({"name": file.filename, "text": text, "category": category})
        except Exception as e:
            logging.error(f"Failed to save {file.filename}: {e}")
            return jsonify({"error": f"Failed to save file {file.filename}: {str(e)}"}), 500

    plagiarism_results = check_plagiarism(doc_data)

    # Save the results to JSON files
    with open("plagiarism_report.json", "w") as f:
        json.dump(plagiarism_results, f)

    with open("pdf_data.json", "w") as f:
        json.dump(doc_data, f)

    return jsonify({"message": "PDFs Uploaded & Processed!"})

# API Endpoint to get plagiarism report
@app.route("/api/plagiarism-report", methods=["GET"])
def get_plagiarism_report():
    if os.path.exists("plagiarism_report.json"):
        with open("plagiarism_report.json", "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

# API Endpoint to get list of PDFs
@app.route("/api/pdf-list", methods=["GET"])
def get_pdf_list():
    if os.path.exists("pdf_data.json"):
        with open("pdf_data.json", "r") as f:
            pdfs = [item["name"] for item in json.load(f)]
        return jsonify(pdfs)
    return jsonify([])

# API Endpoint for querying PDFs
@app.route("/api/query", methods=["POST"])
def query_pdf():
    data = request.json
    pdf_name = data.get("pdf_name")
    query = data.get("query")

    # Load vector database
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = splitter.split_documents(docs)
    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    vectordb = FAISS.from_documents(documents, embeddings_model)
    retriever = vectordb.as_retriever()

    pdf_tool = create_retriever_tool(retriever, "pdf_search", "Retrieve PDF information.")
    tools = [pdf_tool]

    # Setup AI Model (LLaMA3 via Groq)
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    agent = create_openai_tools_agent(llm, tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # Run AI Query
    try:
        response = agent_executor.invoke({"input": query})
        return jsonify({"response": response["output"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Static Route for serving CSS (and other static files)
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
