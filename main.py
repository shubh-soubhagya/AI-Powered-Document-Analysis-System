import os
import spacy
import re
import threading
import fitz  # PyMuPDF for PDF text extraction
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from collections import Counter
from plagiarism.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plagiarism.jaccard_similarity import jaccard_similarity
from plagiarism.lcs import lcs
from plagiarism.lsh import lsh_similarity
from plagiarism.n_gram_similarity import n_gram_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


MODEL_PATH = r"model_training\trained_model"
PDF_DIRECTORY = r"pdf_app_test"  # Directory containing PDF files

# PDF Extraction Function
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Prediction Function
def predict_category(nlp, text):
    doc = nlp(text)
    return max(doc.cats, key=doc.cats.get)

# Plagiarism Check Function
def plagiarism_worker(doc1, doc2, name1, name2):
    result = [
        f"\n🔎 Comparing: {name1} vs {name2}",
        f"- Cosine Similarity (TF-IDF): {cosine_similarity_tfidf(doc1, doc2) * 100:.2f}%",
        f"- Cosine Similarity (CountVectorizer): {cosine_similarity_count(doc1, doc2) * 100:.2f}%",
        f"- Jaccard Similarity: {jaccard_similarity(doc1, doc2) * 100:.2f}%",
        f"- LCS Similarity: {lcs(doc1, doc2) * 100:.2f}%",
        f"- LSH Similarity: {lsh_similarity(doc1, doc2) * 100:.2f}%",
        f"- N-Gram Similarity: {n_gram_similarity(doc1, doc2) * 100:.2f}%"
    ]
    print("\n".join(result))

# Plagiarism Checker
def check_plagiarism(docs, nlp):
    threads = []
    for i in range(len(docs)):
        category = predict_category(nlp, docs[i]["text"])
        print(f"\n\n 📄 Document: {docs[i]['name']} - Predicted Category: {category}")
        for j in range(i + 1, len(docs)):
            thread = threading.Thread(target=plagiarism_worker, args=(docs[i]["text"], docs[j]["text"], docs[i]["name"], docs[j]["name"]))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('file')
    uploaded_files = []

    for file in files:
        if file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            extracted_text = extract_text_from_pdf(file_path)
            uploaded_files.append({"name": filename, "text": preprocess_text(extracted_text)})

    if uploaded_files:
        results = check_plagiarism(uploaded_files)
        return render_template('result.html', documents=uploaded_files, results=results)

    return jsonify({"error": "No valid PDF files uploaded."})


# Main Function
def main():
    nlp = spacy.load(MODEL_PATH)
    documents = []

    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            documents.append({"name": filename, "text": preprocess_text(extracted_text)})

    if documents:
        check_plagiarism(documents, nlp)
    else:
        print("❗ No PDF documents found in the specified directory.")

if __name__ == "__main__":
    # nlp = spacy.load(MODEL_PATH)
    # app.run(debug=True)
    main()
