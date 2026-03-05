"""
Step 4 of RAG: Question-Answering chain.

This is where the magic happens:
1. User asks a question
2. We search the vector store for relevant chunks
3. We send those chunks + the question to the LLM
4. The LLM generates an answer based on the provided context
"""

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.

Rules:
1. Only answer based on the provided context below.
2. If the answer is not in the context, say "I couldn't find this information in the uploaded document."
3. Be concise but thorough.
4. If relevant, mention which part of the document the answer comes from.

Context from the document:
{context}

Question: {question}

Helpful Answer:"""


def create_qa_chain(vector_store):
    """
    Creates a question-answering chain.
    """
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 4}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


def ask_question(qa_chain, question):
    """
    Ask a question and get an answer with source references.
    """
    result = qa_chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        content = doc.page_content
        preview = content[:150] + "..." if len(content) > 150 else content
        sources.append({
            "page": doc.metadata.get("page", "Unknown"),
            "content_preview": preview
        })

    return {
        "answer": result["result"],
        "sources": sources
    }
