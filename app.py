from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from classification_modules.text_extraction import extract_text_from_pdf, split_into_sentences
from collections import Counter
import spacy
import os
from plagiarism.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plagiarism.jaccard_similarity import jaccard_similarity
from plagiarism.lcs import lcs
from plagiarism.lsh import lsh_similarity
from plagiarism.n_gram_similarity import n_gram_similarity
from collections import Counter
import re
import numpy as np

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for frontend-backend communication

MODEL_PATH = "model_training/trained_model"

def predict_category(nlp, sentences):
    """Predicts categories for a list of sentences and returns the majority prediction."""
    predictions = [max(nlp(sentence).cats, key=nlp(sentence).cats.get) for sentence in sentences]
    most_common_label, _ = Counter(predictions).most_common(1)[0]
    return most_common_label, predictions

doc1 = preprocess_text(doc1)
doc2 = preprocess_text(doc2)

print(f"Cosine Similarity (TF-IDF): {cosine_similarity_tfidf(doc1, doc2) * 100:.2f}%")
print(f"Cosine Similarity (CountVectorizer): {cosine_similarity_count(doc1, doc2) * 100:.2f}%")
print(f"Jaccard Similarity: {jaccard_similarity(doc1, doc2) * 100:.2f}%")
print(f"LCS Similarity: {lcs(doc1, doc2) * 100:.2f}%")
print(f"LSH Similarity: {lsh_similarity(doc1, doc2) * 100:.2f}%")
print(f"N-Gram Similarity: {n_gram_similarity(doc1, doc2) * 100:.2f}%")

@app.route("/")
def index():
    """Serve the frontend HTML page."""
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_document():
    """Handle PDF uploads and return NLP analysis results."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    temp_path = "temp.pdf"
    file.save(temp_path)

    try:
        nlp = spacy.load(MODEL_PATH)
        text = extract_text_from_pdf(temp_path)
        sentences = split_into_sentences(text)

        majority_label, all_predictions = predict_category(nlp, sentences)

        return jsonify({
            "majority_label": majority_label,
            "all_predictions": all_predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True)
