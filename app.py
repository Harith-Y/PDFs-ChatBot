import streamlit as st
from dotenv import load_dotenv
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from embedding_providers import get_embedding_model

from langchain.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI


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

# This function is now much cleaner and more readable.
def get_vector_store(chunks):
    # You can easily switch the model provider here
    model_choice = 'qwen'

    st.write(f"Initializing embedding model: '{model_choice}'...")
    embeddings = get_embedding_model(model_choice)

    st.write("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    st.write("Vector store created successfully!")
    return vector_store

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        # I don't know whether this will work, Have to test it.
        llm=ChatOpenAI(
            base_url="https://generativelanguage.googleapis.com",
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.5-pro"
        ),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF-ChatBot",
        page_icon=":books:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

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

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vector_store)

            st.success("Done!")


if __name__ == '__main__':
    main()
