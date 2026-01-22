import os
from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader, Document

def load_documents(data_path: str = "data") -> List[Document]:
    """
    Loads PDF documents from the specified directory using PyPDF.
    """
    # Ensure the path exists
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"The directory {data_path} does not exist.")

    print(f"Loading documents from: {path.absolute()}")

    # SimpleDirectoryReader automatically detects file extensions.
    # By default, it uses PyPDFReader for .pdf files if pypdf is installed.
    reader = SimpleDirectoryReader(
        input_dir=str(path),
        required_exts=[".pdf"],
        recursive=False
    )

    documents = reader.load_data()
    print(f"Successfully loaded {len(documents)} pages/documents.")
    
    return documents

if __name__ == "__main__":
    # Quick test to verify loading logic
    # Make sure you have at least one PDF in the /data folder
    try:
        docs = load_documents("data")
        if docs:
            print(f"Sample content from first doc: {docs[0].text[:100]}...")
    except Exception as e:
        print(f"Error: {e}")