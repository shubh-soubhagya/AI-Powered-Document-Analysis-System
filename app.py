from plagiarism.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from classification_modules.text_extraction import extract_text_from_pdf, split_into_sentences
from plagiarism.jaccard_similarity import jaccard_similarity
from plagiarism.lcs import lcs
from plagiarism.lsh import lsh_similarity
from plagiarism.n_gram_similarity import n_gram_similarity
from collections import Counter
import spacy
import fitz  # PyMuPDF


def predict_category(nlp, sentences):
    """Predicts categories for a list of sentences and returns the majority prediction."""
    predictions = []
    for sentence in sentences:
        doc = nlp(sentence)
        predicted_label = max(doc.cats, key=doc.cats.get)
        predictions.append(predicted_label)

    # Determine majority prediction
    most_common_label, _ = Counter(predictions).most_common(1)[0]
    return most_common_label, predictions


def main(pdf_path, model_path):
    """Loads model, processes PDF, and prints the majority category."""
    print("Loading trained NLP model...")
    nlp = spacy.load(model_path)

    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)

    print("Predicting document category...")
    majority_label, all_predictions = predict_category(nlp, sentences)

    print("Majority Category:", majority_label)
    print("All Predictions:", all_predictions)


if __name__ == "__main__":
    pdf_path = r"data/pdf/legalimit.pdf"  # Update with actual PDF path
    model_path = r"model_training/trained_model"  # Update with actual model path
    main(pdf_path, model_path)
