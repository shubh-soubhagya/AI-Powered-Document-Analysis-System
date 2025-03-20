from DBAQ.data_utilization import load_environment, load_and_split_pdfs, create_vector_db
from DBAQ.query_bot import create_agent_executor, run_query_loop

groq_api_key = load_environment()
pdf_directory = r"AI-Powered-Document-Analysis-System\pdf_app_test"
embeddings_path = r"C:\Users\hp\Desktop\ps_sol\models\all-MiniLM-L6-v2"

documents = load_and_split_pdfs(pdf_directory)
retriever = create_vector_db(documents, embeddings_path)
agent_executor = create_agent_executor(retriever, groq_api_key)

run_query_loop(agent_executor)
