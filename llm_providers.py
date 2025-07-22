import os

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import HuggingFaceHub
from langchain.llms import Cohere

# load_dotenv()

def _get_gemini_llm():
    """Initializes and returns the Google Gemini LLM."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=SecretStr(api_key),
        temperature=0.3,
        convert_system_message_to_human=True
    )


def _get_huggingface_llm():
    """Initializes and returns a HuggingFace Hub LLM."""
    if not os.getenv("HF_TOKEN"):
        raise ValueError("HF_TOKEN not found in environment variables.")

    return HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={
            "temperature": 0.3,
            "max_length": 512
        }
    )

def _get_cohere_llm():
    """Initializes and returns the Cohere LLMs."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in environment variables.")

    return Cohere(
        model="command-a-03-2025",
        cohere_api_key=api_key,
        temperature=0.3
    )

llm_providers = {
    'gemini-2.5-pro': _get_gemini_llm,
    'flan-t5-xxl': _get_huggingface_llm,
    'command-a-03-2025': _get_cohere_llm
}

def get_llm(model_choice: str):
    """
    Factory function to select and initialize the chosen LLM.

    Args:
        model_choice: A string identifier for the model ('gemini-2.5-pro', 'flan-t5-xxl', 'command-a-03-2025).

    Returns:
        An initialized LangChain LLM instance.
    """

    if model_choice not in llm_providers:
        raise ValueError(
            f"Invalid LLM choice: '{model_choice}'. "
            f"Please choose from {list(llm_providers.keys())}."
        )

    # Call the appropriate helper function to get the LLM instance
    return llm_providers[model_choice]()
