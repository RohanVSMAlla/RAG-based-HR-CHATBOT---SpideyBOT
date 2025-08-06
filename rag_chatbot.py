import os
from dotenv import load_dotenv

#  LangChain imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_project_id = os.getenv("OPENAI_PROJECT_ID")

# Initialize embedding model (no project param needed)
embedding_model = OpenAIEmbeddings(api_key=openai_api_key)

# Load vectorstore
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# LLM - Pass project ID correctly via headers (latest fix)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=openai_api_key,
    default_headers={"OpenAI-Project": openai_project_id}
)

# Prompt template
prompt_template = """Use the context below to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate.from_template(prompt_template)

# Build QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# CLI chatbot
print("\n🤖 Ask me anything from your PDFs! (type 'exit' to quit)\n")
while True:
    query = input("🧠 You: ")
    if query.lower() == "exit":
        print("Chatbot shutting down. See you next time!\n")
        break
    try:
        result = qa.invoke(query)
        print("\n📄 Answer:", result["result"], "\n")
    except Exception as e:
        print("\n⚠️ Oops! Something went wrong:", str(e), "\n")







