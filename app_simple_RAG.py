import sys
import re
import logging
from typing import Tuple, Optional, Any

# ...existing installation-specific imports...
import ollama
import gradio as gr

# Document processing and retrieval
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_pdf(pdf_bytes: Optional[bytes]) -> Tuple[Any, Any, Any]:
    if pdf_bytes is None:
        return None, None, None
    # ...existing PDF loading code...
    loader = PyMuPDFLoader(pdf_bytes)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()
    logging.info("PDF processed: %d chunks generated", len(chunks))
    return text_splitter, vectorstore, retriever

def combine_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question: str, context: str) -> str:
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    response_content = response['message']['content']
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return final_answer

def rag_chain(question: str, text_splitter: Any, vectorstore: Any, retriever: Any) -> str:
    retrieved_docs = retriever.invoke(question)  # retrieve relevant chunks
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)

def ask_question(pdf_bytes: Optional[bytes], question: str) -> Optional[str]:
    text_splitter, vectorstore, retriever = process_pdf(pdf_bytes)
    if text_splitter is None:
        return "No PDF provided or error in processing."
    result = rag_chain(question, text_splitter, vectorstore, retriever)
    return result

# Simple test function for key functionality
def run_tests():
    from collections import namedtuple
    Doc = namedtuple("Doc", ["page_content"])
    # Test combine_docs()
    docs = [Doc("Hello"), Doc("World")]
    combined = combine_docs(docs)
    assert combined == "Hello\n\nWorld", "combine_docs test failed"
    logging.info("combine_docs test passed")
    print("All tests passed.")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        # Define a Gradio interface
        interface = gr.Interface(
            fn=ask_question,
            inputs=[
                gr.File(label="Upload PDF (optional)"),
                gr.Textbox(label="Ask a question")
            ],
            outputs="text",
            title="Ask questions about your PDF",
            description="Use DeepSeek-R1 1.5B to answer your questions about the uploaded PDF document."
        )
        interface.launch()
