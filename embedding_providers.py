import os
from pydantic import SecretStr
from huggingface_hub import InferenceClient

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from huggingface_embeddings import HuggingFaceInferenceEmbeddings


def _get_nvidia_embeddings():
    """Initializes and returns the NVIDIA embedding model."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found in environment variables.")
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=api_key,
        truncate="NONE"
    )


def _get_qwen_embeddings():
    """Initializes and returns the Qwen embedding model via HuggingFace."""
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN not found in environment variables.")
    client = InferenceClient(provider="nebius", api_key=api_key)
    return HuggingFaceInferenceEmbeddings(
        client=client,
        model_name="Qwen/Qwen3-Embedding-8B",
    )


def _get_gemini_embeddings():
    """Initializes and returns the Google Gemini embedding model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(api_key),
        task_type="RETRIEVAL_DOCUMENT"
    )

embedding_providers = {
        'nvidia/nv-embedqa-e5-v5': _get_nvidia_embeddings,
        'Qwen/Qwen3-Embedding-8B': _get_qwen_embeddings,
        'gemini/embedding-001': _get_gemini_embeddings,
    }

def get_embedding_model(model_choice: str):
    """
    Factory function to select and initialize the chosen embedding model.

    Args:
        model_choice: A string identifier for the model ('nvidia/nv-embedqa-e5-v5', 'Qwen/Qwen3-Embedding-8B', 'gemini/embedding-001').

    Returns:
        An initialized LangChain embedding model instance.
    """

    if model_choice not in embedding_providers:
        raise ValueError(
            f"Invalid model choice: '{model_choice}'. "
            f"Please choose from {list(embedding_providers.keys())}."
        )

    # Call the appropriate helper function to get the model instance
    return embedding_providers[model_choice]()
