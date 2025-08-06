import os
from dotenv import load_dotenv

# Set path for poppler (used for PDF parsing)
os.environ["PATH"] += os.pathsep + r"C:\poppler\Library\bin"

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # must be sk-proj-... key

# LangChain (updated imports)
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # updated import

# Step 1: Load and split the documents
def load_and_split_docs():
    pdf_paths = [
        "docs/Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf",
        "docs/HR-Guide_-Policy-and-Procedure-Template.pdf"
    ]

    all_chunks = []
    for path in pdf_paths:
        print(f"📄 Loading {path}...")
        loader = UnstructuredPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    return all_chunks

# Step 2: Create and store vector embeddings
def create_vectorstore(chunks):
    print(" Creating embeddings and storing vectors...")

    # Updated to use langchain-openai's OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    print(" Vectorstore created and persisted.")

#  Run the pipeline
if __name__ == "__main__":
    chunks = load_and_split_docs()
    create_vectorstore(chunks)



