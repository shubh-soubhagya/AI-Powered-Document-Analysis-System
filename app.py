from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from classification_modules.text_extraction import extract_text_from_pdf, split_into_sentences
# from classification_modules.plagiarism_checker import check_plagiarism  # Import plagiarism module
from collections import Counter
import spacy
import os
from DBAQ import qna_script  # Import QnA system
import docx

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable CORS for frontend-backend communication

MODEL_PATH = "model_training/trained_model"


def extract_text(file_path):
    """Extracts text from PDF or DOCX."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""


def predict_category(nlp, sentences):
    """Predicts categories for a list of sentences."""
    predictions = [max(nlp(sentence).cats, key=nlp(sentence).cats.get) for sentence in sentences]
    most_common_label, _ = Counter(predictions).most_common(1)[0]
    return most_common_label, predictions


@app.route("/")
def index():
    """Serve the frontend HTML page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_document():
    """Handle document uploads, analyze content, and categorize."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".pdf", ".docx"]:
        return jsonify({"error": "Unsupported file format"}), 400

    temp_path = f"uploads/{file.filename}"
    file.save(temp_path)

    try:
        # Load NLP model
        nlp = spacy.load(MODEL_PATH)

        # Extract text and split into sentences
        text = extract_text(temp_path)
        sentences = split_into_sentences(text)

        # Categorization
        majority_label, all_predictions = predict_category(nlp, sentences)

        # Process PDF for QnA (Initialize QnA system dynamically)
        global agent_executor
        agent_executor = qna_script.process_uploaded_pdf(temp_path)
        if not agent_executor:
            return jsonify({"error": "Failed to initialize QnA system"}), 500

        return jsonify({
            "majority_label": majority_label,
            "all_predictions": all_predictions,
            "message": "Document successfully processed for QnA."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(temp_path)


@app.route("/qna", methods=["POST"])
def qna():
    """Handles user queries based on uploaded document's content."""
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        global agent_executor
        if not agent_executor:
            return jsonify({"error": "No document has been analyzed yet"}), 400

        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })

        return jsonify({"answer": response["output"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
