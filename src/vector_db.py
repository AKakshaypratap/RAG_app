import os
import faiss
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.vector_stores.faiss import FaissVectorStore

# The dimension for text-embedding-004 is 768
EMBEDDING_DIM = 768
PERSIST_DIR = "./storage"

def initialize_vector_store(nodes=None, embed_model=None):
    """
    Initializes a FAISS vector store. 
    If nodes are provided, it creates a new index.
    Otherwise, it tries to load an existing one from PERSIST_DIR.
    """
    
    # 1. Create the underlying FAISS index
    # IndexFlatL2 is a simple, effective index for smaller datasets
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # 2. Wrap it in LlamaIndex's FaissVectorStore
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    if nodes:
        # Create a new index from chunks
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context,
            embed_model=embed_model
        )
        # Save to disk
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Index created and persisted to {PERSIST_DIR}")
        return index
    else:
        # Load existing index
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, 
                persist_dir=PERSIST_DIR
            )
            index = load_index_from_storage(
                storage_context,
                embed_model=embed_model
            )
            print("Index loaded from storage.")
            return index
        except Exception as e:
            print(f"Could not load index: {e}")
            return None

if __name__ == "__main__":
    # Example usage (Mock)
    # index = initialize_vector_store(nodes=my_nodes, embed_model=my_model)
    pass