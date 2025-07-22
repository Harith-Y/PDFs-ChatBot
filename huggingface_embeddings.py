from typing import List
from huggingface_hub import InferenceClient
from langchain.embeddings.base import Embeddings
import numpy as np


class HuggingFaceInferenceEmbeddings(Embeddings):
    """
    A custom LangChain Embeddings class that uses the HuggingFace InferenceClient.
    This wrapper is necessary because the InferenceClient does not have the
    `embed_documents` and `embed_query` methods that LangChain expects.
    """

    def __init__(self, client: InferenceClient, model_name: str):
        """
        Initializes the embedding wrapper.

        Args:
            client: An instance of huggingface_hub.InferenceClient.
            model_name: The name of the embedding model to use.
        """
        self.client = client
        self.model_name = model_name

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        The core embedding logic. Iterates through texts and embeds them one by one.
        """
        all_embeddings = []
        for text in texts:
            # Call the feature_extraction endpoint for each text.
            response_vector = self.client.feature_extraction(
                text=text,
                model=self.model_name
            )
            # Flatten the resulting vector to ensure it's a 1D list.
            # This prevents a 3D array when batching, which FAISS cannot handle.
            all_embeddings.append(response_vector.flatten().tolist())
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents for storage."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query for similarity search."""
        # The _embed method returns a list of embeddings. For a single query,
        # we expect a list containing one embedding, so we take the first element.
        return self._embed([text])[0]


