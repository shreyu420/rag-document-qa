"""
Step 2 of RAG: Split documents into smaller chunks.

Why? Because:
- LLMs have token limits (can't send entire documents)
- Smaller chunks = more precise retrieval
- Better embeddings for search
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into smaller, overlapping chunks.
    
    Args:
        documents: List of Document objects from the loader
        chunk_size: Maximum characters per chunk (default 1000)
        chunk_overlap: Characters to overlap between chunks (default 200)
                      Overlap ensures we don't lose context at boundaries
    
    Returns:
        List of smaller Document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # These separators try to split at natural boundaries
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"📄 Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks
