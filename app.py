import streamlit as st
from dotenv import load_dotenv
import os
from pydantic import SecretStr

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from embedding_providers import get_embedding_model

from langchain.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_google_genai import ChatGoogleGenerativeAI


from htmlTempllates import css, bot_template, user_template

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
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",  # Use a valid and current model name
        google_api_key=SecretStr(gemini_api_key),
        temperature=0.3,  # A lower temperature is often better for RAG
        convert_system_message_to_human=True  # Helps with compatibility
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True  # Useful for debugging and showing sources
    )

    return conversation_chain

def handle_user_input(query):
    response = st.session_state.conversation({
        "question": query
    })

    st.write(response)

def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF-ChatBot",
        page_icon=":books:"
    )

    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with PDFs :books:")
    user_query = st.text_input("Type your Query about your documents: ")

    if user_query:
        handle_user_input(user_query)

    st.write(user_template.replace("{{MSG}}", "Hello Bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)
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
