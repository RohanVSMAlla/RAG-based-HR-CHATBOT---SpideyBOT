import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up embeddings and vectorstore
embedding_model = OpenAIEmbeddings(api_key=openai_api_key)
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Set up LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

# Set up QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Streamlit App UI
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("🕷️🤖🦸 Spidey BOT — Your friendly workplace HR-bot")

#  Persistent chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_input = st.chat_input("Spidey is here! Feel free to ask anything you wanna know...")

if user_input:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain.invoke(user_input)

    # Display and save chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", result["result"]))

# Display chat history
for speaker, message in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(message)
