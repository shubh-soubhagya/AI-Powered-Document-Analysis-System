# app.py
import os
import logging
import time
import fitz  # PyMuPDF for PDF text extraction
import re
import spacy
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
import tempfile
import shutil
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

# Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'pdf'}
EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', 'models/all-MiniLM-L6-v2')
SPACY_MODEL_PATH = os.getenv('SPACY_MODEL_PATH', 'model_training/trained_model')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Load NLP model once
nlp = None
try:
    nlp = spacy.load(SPACY_MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to load Spacy model: {e}")

# Function: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function: Preprocess text
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

# Function: Predict category using NLP model
def predict_category(nlp, text):
    try:
        doc = nlp(text[:10000])  # Limit text to avoid memory issues
        return max(doc.cats, key=doc.cats.get)
    except Exception as e:
        logging.error(f"Error predicting category: {e}")
        return "unknown"

# Function: Perform plagiarism check
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

def check_plagiarism(docs):
    results = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            results.append(plagiarism_worker(docs[i]["text"], docs[j]["text"], docs[i]["name"], docs[j]["name"]))
    return results

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Clean up any previous files
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    return render_template(r'\templates_shubh\index.html')

@app.route('/upload', methods=['POST'])
def upload_folder():
    if 'folder' not in request.files:
        return redirect(url_for('index'))
    
    files = request.files.getlist('folder')
    
    if not files or all(file.filename == '' for file in files):
        return redirect(url_for('index'))
    
    pdf_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            pdf_files.append(filename)
    
    if not pdf_files:
        return redirect(url_for('index'))
    
    # Process files and generate plagiarism report
    doc_data = []
    for pdf in pdf_files:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf)
        extracted_text = extract_text_from_pdf(pdf_path)
        doc_data.append({"name": pdf, "text": extracted_text})
    
    plagiarism_results = check_plagiarism(doc_data)
    
    # Add "Generate Chatbot" column format for the table
    table_data = []
    for result in plagiarism_results:
        result["Generate Chatbot (File 1)"] = result["File 1"]
        result["Generate Chatbot (File 2)"] = result["File 2"]
        table_data.append(result)
    
    # Store data in session
    session['pdf_files'] = pdf_files
    session['plagiarism_results'] = table_data
    
    # Process and store predicted categories
    categories = {}
    for pdf in pdf_files:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf)
        text = extract_text_from_pdf(pdf_path)
        preprocessed_text = preprocess_text(text)
        if nlp:
            category = predict_category(nlp, preprocessed_text)
            categories[pdf] = category
        else:
            categories[pdf] = "unknown"
    
    session['categories'] = categories
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'pdf_files' not in session or 'plagiarism_results' not in session:
        return redirect(url_for('index'))
    
    return render_template('dashboard.html', 
                          pdf_files=session['pdf_files'],
                          plagiarism_results=session['plagiarism_results'],
                          categories=session['categories'])

@app.route('/pdf/<filename>')
def get_pdf(filename):
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF not found"}), 404
    
    # Return the PDF content
    with open(pdf_path, 'rb') as f:
        pdf_content = f.read()
    
    from flask import send_file
    return send_file(pdf_path, mimetype='application/pdf')

@app.route('/get_text/<filename>')
def get_text(filename):
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF not found"}), 404
    
    text = extract_text_from_pdf(pdf_path)
    return jsonify({"text": text})

@app.route('/chatbot/<filename>', methods=['POST'])
def chatbot_query(filename):
    data = request.get_json()
    query = data.get('query', '')
    
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF not found"}), 404
    
    # Create LangChain components for this specific PDF
    try:
        # Load the single PDF
        loader = PyPDFDirectoryLoader(os.path.dirname(pdf_path))
        docs = loader.load()
        
        # Filter for only the requested PDF
        docs = [doc for doc in docs if filename in doc.metadata.get('source', '')]
        
        documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)
        
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        vectordb = FAISS.from_documents(documents, embeddings_model)
        retriever = vectordb.as_retriever()
        
        # Tool Setup
        pdf_tool = create_retriever_tool(retriever, "pdf_search", f"Search for information in {filename} only!")
        tools = [pdf_tool]
        
        # Load LLM (via GROQ)
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
        
        # Prompt Setup
        prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided PDF context only.
        Provide accurate and detailed responses strictly from the PDF content.
        <context>
        {context}
        </context>
        Question: {input}
        {agent_scratchpad}
        """)
        
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        
        start_time = time.time()
        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })
        
        return jsonify({
            "response": response['output'],
            "time": f"{time.time() - start_time:.2f}"
        })
    
    except Exception as e:
        logging.error(f"Error processing chatbot query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)