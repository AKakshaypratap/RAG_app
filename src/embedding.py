import os
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings

def get_embedding_model(model_name: str = "models/text-embedding-004"):
    """
    Initializes the Gemini Embedding model.
    Note: 'text-embedding-004' is highly optimized for retrieval tasks.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # Initialize the model
    embed_model = GoogleGenAIEmbedding(
        model_name=model_name,
        api_key=api_key
    )
    
    # Set this as the global embedding model for LlamaIndex
    Settings.embed_model = embed_model
    
    return embed_model

if __name__ == "__main__":
    # Quick test to see if the model produces a vector
    # Make sure to run 'export GOOGLE_API_KEY=your_key' first
    try:
        model = get_embedding_model()
        test_vector = model.get_text_embedding("Hello, Gemini!")
        print(f"Embedding successful! Vector length: {len(test_vector)}")
        print(f"First 5 values: {test_vector[:5]}")
    except Exception as e:
        print(f"Error: {e}")