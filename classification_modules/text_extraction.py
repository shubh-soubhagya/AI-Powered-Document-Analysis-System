import spacy
import fitz  # PyMuPDF
from collections import Counter

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def split_into_sentences(text):
    """Splits text into sentences using punctuation as delimiters."""
    import re
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences