# 📄 RAG Document Q&A System

An AI-powered Document Q&A System using **RAG (Retrieval-Augmented Generation)**. Upload any PDF and ask questions — the AI retrieves relevant sections and generates accurate answers with citations.

---

## ✨ Features

- 📤 **PDF Upload** — Upload any PDF document directly in the browser
- 🔍 **Semantic Search** — Finds relevant sections using vector similarity
- 🤖 **AI-Powered Answers** — Uses GPT-3.5-turbo for accurate, context-based responses
- 📚 **Source Citations** — Every answer shows which pages it came from
- 💬 **Chat History** — Maintains conversation context within a session
- ⚡ **Fast & Local Vector DB** — ChromaDB stores embeddings locally, no extra setup

---

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → Vector DB (ChromaDB)
                                                              ↓
User Question → Semantic Search → Relevant Chunks → LLM (GPT) → Answer
```

---

## 🗂️ Project Structure

```
rag-document-qa/
├── .env.example              # Example API keys file
├── .gitignore                # Files to ignore in git
├── requirements.txt          # Python dependencies
├── app.py                    # Main Streamlit web app
├── rag/
│   ├── __init__.py           # Makes it a Python package
│   ├── document_loader.py    # Load & parse PDFs
│   ├── text_splitter.py      # Split text into chunks
│   ├── embeddings.py         # Create embeddings & vector store
│   └── qa_chain.py           # Question-answering logic
└── README.md
```

---

## 🛠️ Tech Stack

| Component        | Technology                        |
|------------------|-----------------------------------|
| Web UI           | Streamlit                         |
| LLM              | OpenAI GPT-3.5-turbo              |
| Embeddings       | OpenAI text-embedding-3-small     |
| Vector Database  | ChromaDB                          |
| PDF Parsing      | LangChain + PyPDF                 |
| Orchestration    | LangChain                         |

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/shreyu420/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your OpenAI API key

```bash
cp .env.example .env
```

Edit `.env` and replace `your-api-key-here` with your actual [OpenAI API key](https://platform.openai.com/api-keys):

```
OPENAI_API_KEY=sk-...
```

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 💡 How RAG Works

**RAG (Retrieval-Augmented Generation)** combines search and generation:

1. **Indexing** (done once per document):
   - The PDF is parsed into pages
   - Pages are split into overlapping chunks (~1000 characters)
   - Each chunk is converted to a vector embedding (a list of numbers capturing meaning)
   - Embeddings are stored in ChromaDB

2. **Retrieval** (done per question):
   - Your question is also converted to an embedding
   - The 4 most similar chunks are retrieved from ChromaDB

3. **Generation**:
   - The retrieved chunks + your question are sent to GPT-3.5-turbo
   - GPT generates an answer grounded in the document content

This approach prevents hallucination — the LLM can only answer based on what's in your document.

---

## 📖 Usage

1. Open the app and use the sidebar to upload a PDF
2. Wait for the document to be processed (chunked & indexed)
3. Type your question in the chat box
4. View the AI's answer along with source citations

---

## 📝 License

MIT License — feel free to use, modify, and distribute.
