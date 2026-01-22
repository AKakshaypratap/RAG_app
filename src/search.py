import os
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings

def get_query_engine(index):
    """
    Configures the LLM and creates a query engine from the provided index.
    """
    # 1. Initialize Gemini 1.5 Flash
    # This model is optimized for speed and efficiency in RAG tasks
    llm = Gemini(
        model="models/gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,  # Low temperature for more factual answers
    )
    
    # 2. Set as global LLM for LlamaIndex
    Settings.llm = llm
    
    # 3. Create the Query Engine
    # similarity_top_k=3 means it will grab the top 3 most relevant chunks
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    return query_engine

def perform_search(query_engine, user_query: str):
    """
    Executes the query and returns the response.
    """
    print(f"Searching for: {user_query}...")
    response = query_engine.query(user_query)
    return response

if __name__ == "__main__":
    # This would typically be called from main.py after loading the index
    pass