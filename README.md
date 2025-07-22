# PDF-ChatBot 📚

An interactive chat application that allows you to have conversations with your PDF documents. Built with Streamlit and LangChain, this tool uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate answers based on the content of your files.


<img width="1919" height="868" alt="image" src="https://github.com/user-attachments/assets/516a9f8b-e603-4c4b-9724-cbeadaf9fb2e" />


---

## ✨ Features

*   **Multi-PDF Upload:** Upload one or more PDF documents at once.
*   **Dynamic Model Selection:** Easily switch between different embedding and language models directly from the UI.
*   **Multi-Provider Support:** Out-of-the-box support for leading AI providers:
    *   **LLMs:** Google Gemini, Cohere, and Hugging Face (Flan-T5).
    *   **Embeddings:** Google Gemini, NVIDIA, and Qwen (via Hugging Face).
*   **Interactive Chat Interface:** A clean and stateful chat window with conversation memory.
*   **Modular & Extensible:** The codebase is structured to make adding new models and providers simple and clean.

## 🛠️ Tech Stack

*   **Framework:** Streamlit
*   **LLM Orchestration:** LangChain
*   **Vector Store:** FAISS (CPU)
*   **PDF Processing:** PyPDF2
*   **Core:** Python 3

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

*   Python 3.9 or higher.
*   API keys from the AI providers you intend to use (Google, Cohere, etc.).

### 2. Installation & Setup

1. **Clone the repository:**
2. **Create and activate a virtual environment (Recommended)**
3. **Install the required packages:** The `requirements.txt` file contains all the necessary dependencies.
4. **Set up your API keys:** Create a file named `.env` in the root of your project directory. Copy the contents of `.env.example` below into your new file and add your secret keys.

**.env.example:**
```
# -----------------------------------------------------------------------------
# API Keys for PDF-ChatBot
#
# Instructions:
# 1. Create a copy of this file and name it ".env".
# 2. Fill in your secret API keys below.
# 3. The .env file is loaded by the application to access these keys.
#    (This file is listed in .gitignore and should never be committed)
# -----------------------------------------------------------------------------

# Google Gemini API Key
# Used for both the Gemini LLM and the Gemini embedding model.
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# Cohere API Key
# Used for the Cohere Command R+ language model.
COHERE_API_KEY="YOUR_COHERE_API_KEY"

# NVIDIA API Key
# Used for the NVIDIA embedding model.
NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"


# --- Hugging Face Tokens ---
# Note: The same Hugging Face User Access Token can be used for both values.
# They are separate because different LangChain components look for different names.

# Used by the custom HuggingFaceInferenceEmbeddings class for the Qwen model.
HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

# Used by the standard LangChain HuggingFaceHub class for the Flan-T5 LLM.
# Your current code checks for HF_TOKEN, but the library's default is HUGGINGFACEHUB_API_TOKEN.
# Providing both ensures maximum compatibility.
HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```
> **Note:** `HF_TOKEN` and `HUGGINGFACEHUB_API_TOKEN` can be the same Hugging Face User Access Token. They are separate variables because different LangChain components look for different names.

### 3. Running the Application

Once the setup is complete, run the following command in your terminal:

Your browser should automatically open a new tab with the PDF-ChatBot application running.

---

## 📖 How to Use

1.  **Configure Models:** Use the dropdown menus in the sidebar to select your desired Embedding and Language Models.
2.  **Upload Documents:** Drag and drop your PDF files into the file uploader in the sidebar.
3.  **Process:** Click the "Process" button. The application will extract the text, create embeddings, and initialize the conversation chain.
4.  **Chat:** Once processing is complete, the chat interface will appear. You can now ask questions about the content of your documents!

## 📂 Project Structure

The project is organized to separate concerns, making it easy to maintain and extend.
Enjoy chatting with your documents!
