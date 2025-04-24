# 📄 AI-Powered-Document-Analysis-System

The **AI-Powered-Document-Analysis-System** is an AI-powered tool designed to scan PDFs for plagiarism, classify them into financial, legal, and healthcare categories, and enable interaction through a RAG-based chatbot. 

Researchers, enterprises, compliance officers, and students can quickly check plagiarism across academic papers, classify documents, verify content originality, and engage with documents using an AI chatbot to enhance understanding. Ideal for improving productivity in research, compliance, and learning.

---

## 🚀 Key Features

1. **Directory-Wide Plagiarism Detection**  
   - Scans all PDFs in a specified directory and detects similarities between them.  
   - Plagiarism checks are performed based on multiple structural and semantic parameters.  
   - Outputs results in a structured, easy-to-read format.

2. **Document Classification**  
   - Automatically classifies PDFs into categories labelled such as:  
     - 📂 **Healthcare : 0**  
     - 📂 **Financial : 1**  
     - 📂 **Legal : 2**

3. **RAG-Based Chatbot for PDF Interaction**  
   - Select a specific PDF and generate a Retrieval-Augmented Generation (RAG) chatbot.  
   - Ask context-aware questions and get intelligent answers based on the document's content.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/shubh-soubhagya/AI-Powered-Document-Analysis-System.git
cd AI-Powered-Document-Analysis-System
```

### 2. Create a `.env` File
In the root directory of the project, create a .env file with the following content:
Groq API key:
```ini
GROQ_API_KEY="YOUR_API"
```
🔑 Get your free API key from: [Groq Console](https://console.groq.com/keys)

### 3. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 4. Add Your PDFs
Add your PDF files to the directory `pdf_app_test/`

### 5. Run the Application
Once everything is set up, you can run the AI-Powered-Document-Analysis-System
```bash
python app.py
```
---

## 🧠 Just Want to Use the RAG-Based Chatbot?

If you only want to interact with a specific PDF using the RAG chatbot, you can run the following script:
``` bash
python DBQA/dbqa_script.py
```

---

## 📬 Feedback and Contributions
Feel free to submit issues, pull requests, or feature suggestions. Contributions are welcome to help improve the system.

