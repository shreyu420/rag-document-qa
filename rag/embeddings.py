"""
Step 3 of RAG: Create embeddings and store them in a vector database.

What are embeddings?
- They convert text into numbers (vectors) that capture meaning.
- Similar texts have similar vectors.
- This allows us to "search by meaning" instead of exact keywords.

What is a vector database?
- A database optimized for storing and searching vectors.
- ChromaDB is a simple, local vector database (no setup needed).
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def create_vector_store(chunks):
    """
    Creates embeddings for each chunk and stores them in ChromaDB.
    
    Args:
        chunks: List of Document chunks from the text splitter
    
    Returns:
        A Chroma vector store that can be searched
    """
    # Initialize the embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Create the vector store from our chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="document_qa"
    )

    print(f"✅ Created vector store with {len(chunks)} embeddings")
    return vector_store
