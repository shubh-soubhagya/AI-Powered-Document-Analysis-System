import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool

def create_agent_executor(retriever, groq_api_key):
    pdf_tool = create_retriever_tool(retriever, "pdf_search", "Search for PDF information only!")
    tools = [pdf_tool]

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

    prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided PDF context only.
    Provide accurate and detailed responses strictly from the PDF content.
    <context>
    {context}
    <context>
    Questions:{input}
    {agent_scratchpad}
    """
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

def run_query_loop(agent_executor):
    while True:
        query = input("Input your query here: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting... Goodbye!")
            break

        start_time = time.time()
        try:
            response = agent_executor.invoke({
                "input": query,
                "context": "",
                "agent_scratchpad": ""
            })
            print(f"\n🟩 Final Output:\n{response['output']}")
            print(f"⏱️ Total Response Time: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"❗ Error: {e}")
