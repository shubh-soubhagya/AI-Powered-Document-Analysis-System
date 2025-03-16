import os
import spacy
import re
import threading
from collections import Counter
from plagiarism.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plagiarism.jaccard_similarity import jaccard_similarity
from plagiarism.lcs import lcs
from plagiarism.lsh import lsh_similarity
from plagiarism.n_gram_similarity import n_gram_similarity

MODEL_PATH = "model_training/trained_model"

# Sample Text Data
documents = [
    {"name": "healthcare1", "text": "Healthcare systems rely heavily on preventive care strategies to reduce the risk of chronic diseases. Regular screenings, vaccinations, and lifestyle modifications play a crucial role in improving public health. Early diagnosis through advanced medical technologies enables timely intervention, minimizing complications. Telemedicine has emerged as a powerful tool, offering remote consultations and improving healthcare accessibility. By integrating electronic health records (EHR), medical professionals can efficiently manage patient data, ensuring accurate diagnosis and treatment plans. Continuous advancements in healthcare technology are transforming patient outcomes, driving improved quality of life for individuals worldwide."},
    {"name": "healthcare2", "text": "Preventive healthcare strategies are essential in reducing chronic disease risks and improving public well-being. Regular health check-ups, timely vaccinations, and adopting a healthy lifestyle can significantly reduce medical complications. Advanced diagnostic technologies support early detection, allowing healthcare providers to implement preventive measures. Telemedicine platforms have improved healthcare access, particularly for individuals in remote areas. Electronic health records (EHR) systems ensure efficient data management, assisting doctors in delivering precise treatments. Continuous innovations in medical technology continue to improve healthcare systems, enhancing patient care and improving survival rates across various demographics."},
    {"name": "legal1", "text": "Legal frameworks play a vital role in ensuring contracts are binding, protecting the rights of all involved parties. A well-structured contract outlines the obligations, duties, and rights of each entity. Understanding the essential elements of a contract—such as offer, acceptance, and consideration—is crucial for ensuring compliance with legal standards. Failure to meet these elements may result in disputes, litigation, or contract nullification. In corporate environments, contracts help establish transparent agreements in business dealings, ensuring accountability and fostering trust between stakeholders. Legal professionals must stay updated on contract law developments to protect clients and minimize risks."},
    {"name": "legal2", "text": "In legal practice, contractual obligations are fundamental to ensuring smooth business transactions. Contracts define the expectations of each party, reducing the potential for misunderstandings or disputes. Understanding key elements like mutual agreement, consideration, and capacity ensures a contract is legally binding. Legal frameworks provide clear rules for handling breaches, disputes, or enforcement actions. Ensuring contracts align with current legal standards safeguards businesses from liability risks. Moreover, strong contracts support organizational growth, improving collaboration between vendors, clients, and internal teams. By carefully drafting agreements, businesses can protect intellectual property, financial assets, and brand integrity."},
    {"name": "fin1", "text": "Financial markets are influenced by various economic indicators, such as inflation rates, interest rates, and employment data. Investors analyze these metrics to predict market behavior and make informed decisions. Stock prices, currency values, and commodity trends often fluctuate based on these economic signals. Strategic investment planning requires understanding global financial trends, risk assessments, and portfolio diversification. Financial advisors recommend monitoring fiscal policies and central bank actions, as these significantly impact financial stability. Developing financial literacy enables individuals to manage investments effectively, minimize risks, and maximize returns in dynamic market conditions."},
    {"name": "fin2", "text": "Economic indicators such as GDP growth, inflation rates, and interest rates greatly influence financial markets. Traders monitor these indicators to anticipate changes in stock values, currency exchange rates, and investment trends. Financial planning involves assessing market conditions, identifying profitable opportunities, and mitigating risks. Investors should diversify portfolios to balance potential gains and losses. Understanding global trade policies, consumer behavior, and government regulations helps create effective financial strategies. Building financial awareness empowers individuals and businesses to make informed investment decisions, ensuring long-term financial security and growth."}
]

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

# Main Function
def main():
    nlp = spacy.load(MODEL_PATH)
    check_plagiarism(documents, nlp)

if __name__ == "__main__":
    main()
