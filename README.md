<!-- Gemini-Powered RAG with LlamaIndex & FAISS -->
A high-performance Retrieval-Augmented Generation (RAG) pipeline built with Google's Gemini API and LlamaIndex. This application allows you to "chat" with your local PDF documents using state-of-the-art embeddings and lightning-fast vector search.

- Core idea: The system should answer user questions only from the provided documents, not from the model’s general knowledge.

### Tech Stack
- -> LLM: Google Gemini 1.5 Flash
- -> Embeddings: Google `text-embedding-004`
- -> Framework: LlamaIndex
- -> Vector Store: FAISS (Facebook AI Similarity Search)
- -> Frontend: Streamlit
- -> Package Manager: uv

### Prerequisites
- Python 3.10+
- A google gemini api key

### Installation & setup
1) clone the repository:
- git clone https://github.com/AKakshaypratap/RAG_app.git
- cd RAG

2) Create .env file
GOOGLE_API_KEY=your_api_key_here

3) Install dependencies
uv add -r requirements.txt

4) Place your pdf in data folder & run
uv run file_loader.py

5) Run the application
streamlit run app.py

### WORKING STEPS
- Document Ingestion Layer
- Chunking Strategy
- Embedding Generation
- Vector Database
- User Query Processing: Prompt Engineering
- LLM Inference Layer: Streamlit UI interface

### Video Reference:
https://drive.google.com/file/d/16PX-2aJ8PvVz-tYGRv6ZY0BLb8o6ES0k/view?usp=sharing
