import os
import time
import fitz  # PyMuPDF for PDF text extraction
import re
import spacy
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from threading import Thread
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from plagiarism.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plagiarism.jaccard_similarity import jaccard_similarity
from plagiarism.lcs import lcs
from plagiarism.lsh import lsh_similarity
from plagiarism.n_gram_similarity import n_gram_similarity

# Load Environment Variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global variables
UPLOAD_FOLDER = 'uploads'
# EMBEDDING_MODEL_PATH = os.environ.get('EMBEDDING_MODEL_PATH', 'all-MiniLM-L6-v2')
EMBEDDING_MODEL_PATH = r"models\all-MiniLM-L6-v2"
groq_api_key = os.environ.get('GROQ_API_KEY')

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Background processing status
processing_status = {
    'is_processing': False,
    'progress': 0,
    'status_message': '',
    'error': None
}

# Function: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function: Preprocess text
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

# Function: Predict category using NLP model
def predict_category(nlp, text):
    doc = nlp(text)
    return max(doc.cats, key=doc.cats.get)

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

# Process directory in background
def process_directory(directory_path):
    global processing_status
    
    try:
        processing_status = {
            'is_processing': True,
            'progress': 10,
            'status_message': 'Loading PDFs...',
            'error': None
        }
        
        # Find all PDFs in the directory
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            processing_status['error'] = "No PDF files found in the directory."
            processing_status['is_processing'] = False
            return
        
        processing_status['progress'] = 20
        processing_status['status_message'] = 'Extracting text from PDFs...'
        
        # Extract text from PDFs
        doc_data = []
        for pdf in pdf_files:
            pdf_path = os.path.join(directory_path, pdf)
            text = extract_text_from_pdf(pdf_path)
            doc_data.append({"name": pdf, "text": text, "path": pdf_path})
        
        processing_status['progress'] = 50
        processing_status['status_message'] = 'Running plagiarism checks...'
        
        # Run plagiarism check
        plagiarism_results = check_plagiarism(doc_data)
        
        processing_status['progress'] = 70
        processing_status['status_message'] = 'Building vector database...'
        
        # Set up LangChain components
        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)
        
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        vectordb = FAISS.from_documents(documents, embeddings_model)
        
        processing_status['progress'] = 90
        processing_status['status_message'] = 'Categorizing documents...'
        
        # Load NLP model for prediction
        nlp = spacy.load("./model_training/trained_model")
        
        # Predict categories for each document
        for doc in doc_data:
            preprocessed_text = preprocess_text(doc["text"])
            doc["category"] = predict_category(nlp, preprocessed_text)
        
        # Store results in session
        session['doc_data'] = doc_data
        session['plagiarism_results'] = plagiarism_results
        session['directory_path'] = directory_path
        session['vectordb_id'] = str(id(vectordb))  # Store a reference ID
        session['pdfs'] = pdf_files
        
        # Store vectordb in a global variable (for this simple example)
        app.config[session['vectordb_id']] = vectordb
        
        processing_status['progress'] = 100
        processing_status['status_message'] = 'Processing complete!'
        processing_status['is_processing'] = False
        
    except Exception as e:
        processing_status['error'] = str(e)
        processing_status['is_processing'] = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    directory_path = request.form.get('directory_path')
    
    if not directory_path or not os.path.isdir(directory_path):
        return jsonify({"error": "Invalid directory path"}), 400
    
    # Start background processing
    Thread(target=process_directory, args=(directory_path,)).start()
    
    return jsonify({"message": "Processing started"})

@app.route('/status')
def status():
    return jsonify(processing_status)

@app.route('/dashboard')
def dashboard():
    if 'pdfs' not in session:
        return render_template('index1.html', error="No documents processed yet")
    
    return render_template(
        'dashboard.html',
        plagiarism_results=session.get('plagiarism_results', []),
        pdfs=session.get('pdfs', [])
    )

@app.route('/get_pdf_info/<pdf_name>')
def get_pdf_info(pdf_name):
    doc_data = session.get('doc_data', [])
    for doc in doc_data:
        if doc["name"] == pdf_name:
            return jsonify({
                "name": doc["name"],
                "category": doc["category"],
                "text": doc["text"][:1000] + "..." if len(doc["text"]) > 1000 else doc["text"]
            })
    
    return jsonify({"error": "PDF not found"}), 404

@app.route('/chat', methods=['POST'])
def chat():
    pdf_name = request.json.get('pdf_name')
    query = request.json.get('query')
    
    if not pdf_name or not query:
        return jsonify({"error": "Missing pdf_name or query"}), 400
    
    try:
        # Get the vector database
        vectordb_id = session.get('vectordb_id')
        if not vectordb_id or vectordb_id not in app.config:
            return jsonify({"error": "Vector database not found. Please reprocess the directory."}), 404
        
        vectordb = app.config[vectordb_id]
        retriever = vectordb.as_retriever()
        
        # Create tool
        pdf_tool = create_retriever_tool(retriever, "pdf_search", "Search for PDF information only!")
        tools = [pdf_tool]
        
        # Set up LLM
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        
        # Prompt setup
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided PDF context only.
            Provide accurate and detailed responses strictly from the PDF content.
            <context>
            {context}
            </context>
            Question: {input}
            {agent_scratchpad}
            """
        )
        
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        
        start_time = time.time()
        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })
        
        return jsonify({
            "response": response["output"],
            "time_taken": f"{time.time() - start_time:.2f} seconds"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)