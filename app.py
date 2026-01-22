import streamlit as st
import os
from dotenv import load_dotenv

# Import your custom modules
from src.file_loader import load_documents
from src.chunk_file import create_chunks
from src.embedding import get_embedding_model
from src.vector_db import initialize_vector_store
from src.search import get_query_engine, perform_search

# Load environment variables (GOOGLE_API_KEY)
load_dotenv()

st.set_page_config(page_title="Gemini RAG Assistant", layout="wide")
st.title("Gemini PDF Chatbot")

# --- Sidebar: Ingestion ---
with st.sidebar:
    st.header("Setup")
    if st.button("Build/Update Knowledge Base"):
        with st.spinner("Processing documents..."):
            try:
                # 1. Load
                docs = load_documents("data")
                # 2. Chunk
                nodes = create_chunks(docs)
                # 3. Embed
                embed_model = get_embedding_model()
                # 4. Store in FAISS
                index = initialize_vector_store(nodes, embed_model)
                st.success("Index built successfully!")
                st.session_state.index = index
            except Exception as e:
                st.error(f"Error: {e}")

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask something about your PDFs..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if "index" not in st.session_state:
        # Try to load existing index if not built this session
        embed_model = get_embedding_model()
        st.session_state.index = initialize_vector_store(embed_model=embed_model)

    if st.session_state.index:
        query_engine = get_query_engine(st.session_state.index)
        response = perform_search(query_engine, prompt)
        
        # Add assistant response to history
        with st.chat_message("assistant"):
            st.markdown(str(response))
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
    else:
        st.warning("Please build the index first using the sidebar!")