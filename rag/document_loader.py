"""
Step 1 of RAG: Load and parse PDF documents.
This module reads PDF files and extracts text from each page.
"""

from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os


def load_pdf(uploaded_file):
    """
    Takes an uploaded PDF file and extracts text from it.
    
    Args:
        uploaded_file: A file uploaded via Streamlit's file_uploader
    
    Returns:
        List of Document objects (one per page)
    """
    # Save the uploaded file temporarily so we can process it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    try:
        # Use LangChain's PDF loader to extract text
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        return documents
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)
