from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

def create_chunks(documents: List[Document], chunk_size: int = 1024, chunk_overlap: int = 50) -> List[BaseNode]:
    """
    Splits documents into smaller nodes (chunks) for better retrieval.
    
    Args:
        documents: List of loaded Document objects.
        chunk_size: The target number of tokens per chunk.
        chunk_overlap: The number of tokens to overlap between adjacent chunks 
                       (helps maintain context).
    """
    print(f"Chunking {len(documents)} documents...")

    # Initialize the SentenceSplitter
    parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print("Parser", parser)

    # cut into slices based on chunk size & Get nodes from documents
    nodes = parser.get_nodes_from_documents(documents)
    
    print(f"Created {len(nodes)} chunks (nodes).")
    return nodes

if __name__ == "__main__":
    # Mock test
    doc = Document(text="This is a sample document for testing the chunking logic. " * 50)
    chunks = create_chunks([doc], chunk_size=50, chunk_overlap=10)
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i}: {chunk.text[:50]}...")