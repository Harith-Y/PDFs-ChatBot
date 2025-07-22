import streamlit as st
import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from embedding_providers import get_embedding_model
from llm_providers import get_llm
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
def get_vector_store(chunks, model_choice: str):
    st.toast(f"Initializing embedding model: '{model_choice}'...")
    embeddings = get_embedding_model(model_choice)

    st.toast("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    st.toast("Vector store created successfully!")
    return vector_store

def get_conversation_chain(vectorstore, llm_choice: str):
    """Creates a conversation chain with a selectable LLM."""
    st.toast(f"Initializing LLM: '{llm_choice}'...")
    llm = get_llm(llm_choice)

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
    """Gets response from the chain and updates session state."""
    if st.session_state.conversation:
        response = st.session_state.conversation({"question": query})
        st.session_state.chat_history = response['chat_history']
    else:
        st.toast("⚠️ Please process your documents first!", icon="⚠️")

def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF-ChatBot",
        page_icon=":books:"
    )

    st.write(css, unsafe_allow_html=True)

    # Session State Initialization
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = 'gemini'
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = 'gemini'

    st.header("Chat with PDFs :books:")

    if st.session_state.conversation:
        # Display the existing chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        # Using a form for the input
        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your Query about your documents:", placeholder="Ask a question...",
                                       label_visibility="collapsed")
            submit_button = st.form_submit_button("Send")

        # Handle the submission outside the form
        if submit_button and user_query:
            handle_user_input(user_query)
            # Rerun the script to immediately display the new message
            st.rerun()

    else:
        # This message persists until documents are processed
        st.info("Please upload and process your documents in the sidebar to begin chatting.")

    with st.sidebar:
        st.subheader("Configuration")
        embedding_options = ('gemini', 'nvidia', 'qwen')
        # Find the index of the currently selected model
        emb_index = embedding_options.index(st.session_state.embedding_model)
        # Use the calculated index to set the default, and update state with the new selection
        st.session_state.embedding_model = st.selectbox(
            "Choose your Embedding Model:",
            embedding_options,
            index=emb_index
        )

        llm_options = ('gemini', 'huggingface')
        llm_index = llm_options.index(st.session_state.llm_model)
        st.session_state.llm_model = st.selectbox(
            'Choose your Language Model:',
            llm_options,
            index=llm_index
        )

        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            if not pdf_docs:
                st.toast("⚠️ Please upload at least one PDF file.", icon="⚠️")
            else:
                with st.spinner("Processing..."):
                    # Get PDF Text
                    raw_text = get_pdf_text(pdf_docs)
                    # st.write(raw_text)

                    # Get the Text Chunks
                    text_chunks = get_text_chunks(raw_text)
                    # st.write(text_chunks)

                    # Create Vector Store
                    vector_store = get_vector_store(text_chunks, st.session_state.embedding_model)
                    # st.write(vector_store)

                    # Create Conversation Chain
                    st.session_state.conversation = get_conversation_chain(vector_store, st.session_state.llm_model)
                    # Clear previous chat history on new document processing
                    st.session_state.chat_history = []

                st.toast("✅ Done! You can now ask questions.", icon="✅")
                st.rerun()


if __name__ == '__main__':
    main()
