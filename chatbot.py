# Set protobuf to use Python implementation
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Imports
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All
from langchain.chains.question_answering import load_qa_chain

# UI Header
st.header("Chatbot Built By Anil 🤖 to Fetch the Information of Constitution of India📄")

# Sidebar file upload
with st.sidebar:
    st.title("Upload PDF")
    file = st.file_uploader("Upload a PDF file", type="pdf")

# Process the uploaded file
if file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    if not text:
        st.warning("⚠️ No readable text found in the uploaded PDF.")
        st.stop()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert chunks to embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Ask questions
    user_question = st.text_input("Ask a question based on the PDF:")

    if user_question:
        matched_docs = vector_store.similarity_search(user_question)

        # GPT4All LLM (Local model path)
        local_model_path = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        if not os.path.exists(local_model_path):
            st.error("❌ GPT4All model file not found. Please download it from gpt4all.io and place it in the 'models' folder.")
            st.stop()

        llm = GPT4All(
            model=local_model_path,
            backend="gptj",
            verbose=False,
            temp=0.1
        )

        # Run chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=matched_docs, question=user_question)

        st.subheader("📌 Answer:")
        st.write(response)
