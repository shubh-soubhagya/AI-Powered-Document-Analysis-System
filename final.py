import os
import spacy
import time
import re
import threading
import fitz  # PyMuPDF for PDF text extraction
from collections import Counter
from plagiarism.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plagiarism.jaccard_similarity import jaccard_similarity
from plagiarism.lcs import lcs
from plagiarism.lsh import lsh_similarity
from plagiarism.n_gram_similarity import n_gram_similarity
from DBAQ.data_utilization import load_pdfs, split_documents, create_vectorstore
from DBAQ.dbqa_script import load_environment, setup_agent

MODEL_PATH = r"AI-Powered-Document-Analysis-System\model_training\trained_model"
PDF_DIRECTORY = r"AI-Powered-Document-Analysis-System\pdf_app_test"

# PDF Extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Prediction Function
def predict_category(nlp, text):
    doc = nlp(text)
    return max(doc.cats, key=doc.cats.get)

# Plagiarism Comparison
def plagiarism_worker(doc1, doc2, name1, name2):
    print(f"\n🔎 Comparing: {name1} vs {name2}")
    print(f"- Cosine Similarity (TF-IDF): {cosine_similarity_tfidf(doc1, doc2) * 100:.2f}%")
    print(f"- Cosine Similarity (CountVectorizer): {cosine_similarity_count(doc1, doc2) * 100:.2f}%")
    print(f"- Jaccard Similarity: {jaccard_similarity(doc1, doc2) * 100:.2f}%")
    print(f"- LCS Similarity: {lcs(doc1, doc2) * 100:.2f}%")
    print(f"- LSH Similarity: {lsh_similarity(doc1, doc2) * 100:.2f}%")
    print(f"- N-Gram Similarity: {n_gram_similarity(doc1, doc2) * 100:.2f}%")

# Plagiarism Check
def check_plagiarism(docs):
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            plagiarism_worker(docs[i]["text"], docs[j]["text"], docs[i]["name"], docs[j]["name"])

# Load chatbot setup for the selected document
def setup_chatbot(selected_doc, groq_api_key):
    vectordb = create_vectorstore([selected_doc], MODEL_PATH)
    return setup_agent(vectordb, groq_api_key)

# Main Interactive Terminal
def main():
    nlp = spacy.load(MODEL_PATH)
    documents = []

    # Load and preprocess PDFs
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            documents.append({"name": filename, "text": preprocess_text(extracted_text)})

    if not documents:
        print("❗ No PDF documents found in the specified directory.")
        return

    print("\n🔎 Displaying Document Similarities:")
    check_plagiarism(documents)

    # Document Selection
    while True:
        print("\n📋 Select a document to analyze:")
        for idx, doc in enumerate(documents):
            print(f"[{idx + 1}] {doc['name']}")

        choice = input("Enter the number of the document to select (or type 'exit' to quit): ")
        if choice.lower() in ["exit", "quit"]:
            print("Exiting... Goodbye!")
            break

        try:
            selected_doc = documents[int(choice) - 1]
        except (ValueError, IndexError):
            print("❗ Invalid choice. Please try again.")
            continue

        # Category Prediction
        category = predict_category(nlp, selected_doc["text"])
        print(f"\n📄 Selected Document: {selected_doc['name']} - Predicted Category: {category}")

        # Setup Chatbot for Selected Document
        groq_api_key = load_environment()
        agent_executor = setup_chatbot(selected_doc, groq_api_key)

        # Chat with Selected Document
        print("\n💬 Chat with the selected document below. Type 'switch' to change documents or 'exit' to quit.")
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                print("Exiting... Goodbye!")
                return
            if query.lower() == "switch":
                break

            start_time = time.time()
            try:
                response = agent_executor.invoke({
                    "input": query,
                    "context": "",
                    "agent_scratchpad": ""
                })
                print(f"\n🟩 Response: {response['output']}")
                print(f"⏱️ Response Time: {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"❗ Error: {e}")

if __name__ == "__main__":
    main()
