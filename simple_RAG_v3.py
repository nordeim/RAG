import os
import sys
import re
import logging
import shutil  # Added for cross-drive file moving
import mimetypes  # Ensure mimetypes is imported
import fitz  # Added for PDF conversion
from typing import List, Tuple, Optional, Any

# Updated imports
import ollama
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader  # Unused now; can remove if desired
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Updated import for Chroma
try:
    from langchain_ollama import OllamaEmbeddings  # Updated import for embeddings
except ModuleNotFoundError:
    print("Module 'langchain_ollama' not found. Please install it with 'pip install langchain-ollama'")
    sys.exit(1)
from langchain.schema import Document  # For creating document objects

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create RAG source folder if missing
RAG_SOURCE_FOLDER = "RAG_source"
os.makedirs(RAG_SOURCE_FOLDER, exist_ok=True)

def detect_file_type(file_path: str) -> Optional[str]:
    """Detect the file type using mimetypes."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()
        if ext_lower in ['.txt', '.log', '.csv']:
            mime_type = 'text/plain'
        elif ext_lower in ['.epub']:
            mime_type = 'application/epub+zip'
        elif ext_lower in ['.pdf']:
            mime_type = 'application/pdf'
        elif ext_lower in ['.docx', '.doc']:
            mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif ext_lower in ['.pptx', '.ppt']:
            mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        elif ext_lower in ['.xlsx', '.xls']:
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif ext_lower in ['.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp']:
            mime_type = f'image/{ext_lower[1:]}'
    return mime_type

def convert_to_text(input_file: str, output_file: str) -> Optional[str]:
    """Convert various document types to plain text."""
    file_type = detect_file_type(input_file)
    if file_type == "application/pdf":
        # Use PyMuPDF for PDF conversion
        try:
            text = ""
            with fitz.open(input_file) as doc:
                for page in doc:
                    text += page.get_text()
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            return text
        except Exception as e:
            return f"Error converting PDF: {e}"
    elif file_type and (file_type.startswith("text/") or file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        # Add similar logic for other file types
        pass  # Placeholder for additional file type handling
    else:
        return f"Unsupported file type: {file_type}"

def combine_docs(docs: List[Document]) -> str:
    """Combine document contents using newline separator."""
    return "\n\n".join(doc.page_content for doc in docs)

def process_documents(file_paths: List[str]) -> Tuple[Any, Any, Any]:
    """Process multiple documents and store them in the vector database."""
    all_chunks = []
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for file_path in file_paths:
        # Save the original file in the RAG source folder
        file_name = os.path.basename(file_path)
        saved_path = os.path.join(RAG_SOURCE_FOLDER, file_name)
        # Use shutil.move for cross-drive file moving
        shutil.move(file_path, saved_path)

        # Convert the document to text
        temp_text_file = f"temp_{file_name}.txt"
        text_content = convert_to_text(saved_path, temp_text_file)

        # Instead of using PyMuPDFLoader, read the converted text file directly
        try:
            with open(temp_text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            doc = Document(page_content=content)
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        except Exception as e:
            logging.error("Error reading converted file %s: %s", temp_text_file, e)

    # Store all chunks in the vector database
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()
    logging.info("Processed %d documents: %d chunks generated", len(file_paths), len(all_chunks))
    return text_splitter, vectorstore, retriever

def export_processed_data(vectorstore: Any, output_file: str):
    """Export processed data from the vector database into a text file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in vectorstore.get_all_documents():
                f.write(f"{doc.page_content}\n")
        logging.info("Exported processed data to %s", output_file)
    except Exception as e:
        logging.error("Error exporting processed data: %s", e)

def rag_chain(question: str, text_splitter: Any, vectorstore: Any, retriever: Any, system_prompt: str = "") -> str:
    """Generate a response using the RAG pipeline."""
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    formatted_prompt = f"System: {system_prompt}\n\nQuestion: {question}\n\nContext: {formatted_content}"
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    return response['message']['content']

def ask_question(file_paths: List[str], question: str, system_prompt: str = "") -> Optional[str]:
    """Handle the end-to-end process of uploading files and answering questions."""
    text_splitter, vectorstore, retriever = process_documents(file_paths)
    result = rag_chain(question, text_splitter, vectorstore, retriever, system_prompt)
    return result

def run_tests():
    """Run basic tests for key functionality."""
    from collections import namedtuple
    Doc = namedtuple("Doc", ["page_content"])
    docs = [Doc("Hello"), Doc("World")]
    combined = combine_docs([Document(page_content=d.page_content) for d in docs])
    assert combined == "Hello\n\nWorld", "combine_docs test failed"
    logging.info("combine_docs test passed")
    print("All tests passed.")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        interface = gr.Interface(
            fn=ask_question,
            inputs=[
                gr.File(label="Upload Files (optional)", file_count="multiple"),
                gr.Textbox(label="Ask a question"),
                gr.Textbox(label="System Prompt (optional)")
            ],
            outputs="text",
            title="Ask Questions About Your Documents",
            description="Use DeepSeek-R1 1.5B to answer your questions about the uploaded documents."
        )
        interface.launch()

