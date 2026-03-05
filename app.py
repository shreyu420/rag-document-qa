"""
Main Streamlit Application - RAG Document Q&A System
"""

import streamlit as st
from dotenv import load_dotenv
from rag.document_loader import load_pdf
from rag.text_splitter import split_documents
from rag.embeddings import create_vector_store
from rag.qa_chain import create_qa_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="📄 Document Q&A with AI",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 RAG Document Q&A System")
st.markdown("""
Upload a PDF document and ask questions about it!  
Powered by **OpenAI GPT** + **RAG (Retrieval-Augmented Generation)**
""")

st.divider()

with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to ask questions about"
    )

    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. 📤 **Upload** a PDF document
    2. 🔍 The system **chunks & indexes** it
    3. ❓ **Ask** any question about the document
    4. 🤖 AI **retrieves** relevant sections & **generates** an answer
    """)

if uploaded_file:
    if "qa_chain" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("🔄 Processing document... This may take a moment."):
            st.info("📄 Loading PDF...")
            documents = load_pdf(uploaded_file)

            st.info("✂️ Splitting into chunks...")
            chunks = split_documents(documents)

            st.info("🧠 Creating embeddings...")
            vector_store = create_vector_store(chunks)

            st.info("⚡ Setting up QA system...")
            qa_chain = create_qa_chain(vector_store)

            st.session_state.qa_chain = qa_chain
            st.session_state.file_name = uploaded_file.name
            st.session_state.num_chunks = len(chunks)
            st.session_state.num_pages = len(documents)

        st.success(f"✅ Ready! Processed {st.session_state.num_pages} pages into {st.session_state.num_chunks} chunks.")

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        page = source['page']
                        page_display = page + 1 if isinstance(page, int) else page
                        st.markdown(f"**Source {i}** (Page {page_display}):")
                        st.caption(source["content_preview"])

    if question := st.chat_input("Ask a question about your document..."):
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                result = ask_question(st.session_state.qa_chain, question)

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(result["sources"], 1):
                        page = source['page']
                        page_display = page + 1 if isinstance(page, int) else page
                        st.markdown(f"**Source {i}** (Page {page_display}):")
                        st.caption(source["content_preview"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })

else:
    st.info("👈 Please upload a PDF document from the sidebar to get started!")

    st.markdown("### 💡 Example Questions You Can Ask:")
    st.markdown("""
    - *"What is the main topic of this document?"*
    - *"Summarize the key findings."*
    - *"What does section 3 talk about?"*
    - *"List all the recommendations mentioned."*
    - *"Explain the methodology used."*
    """)
