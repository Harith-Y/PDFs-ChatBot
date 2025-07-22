import streamlit as st
from dotenv import load_dotenv
from pydantic import SecretStr
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from huggingface_hub import InferenceClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.vectorstores import FAISS
from custom_embeddings import HuggingFaceInferenceEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        # Use += to append text from all pages and all documents
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    model_choice = 'qwen'
    embeddings = None
    if model_choice == 'nvidia':
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables.")

        # Instantiate the NVIDIA Embeddings model
        embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=nvidia_api_key,
            truncate="NONE"
        )

    elif model_choice == 'qwen':
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables.")

        client = InferenceClient(
            provider="nebius",
            api_key=hf_token,
        )

        embeddings = HuggingFaceInferenceEmbeddings(
            client=client,
            model_name="Qwen/Qwen3-Embedding-8B",
        )

    elif model_choice == 'gemini':
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=SecretStr(gemini_api_key),
            task_type="RETRIEVAL_DOCUMENT"
        )

    else:
        raise ValueError(f"Invalid model choice: '{model_choice}'. Please choose 'nvidia', 'qwen', or 'gemini'.")

    st.write("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    st.write("Vector store created successfully!")
    return vector_store


def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF-ChatBot",
        page_icon=":books:"
    )

    st.header("Chat with PDFs :books:")
    st.text_input("Type your Query about your documents: ")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get PDF Text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # Get the Text Chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # Create Vector Store
                vector_store = get_vector_store(text_chunks)
                # st.write(vector_store)

            st.success("Done!")

if __name__ == '__main__':
    main()
