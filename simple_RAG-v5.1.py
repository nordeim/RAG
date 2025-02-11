import os
import sys
import re
import logging
import shutil
import mimetypes
import fitz
from typing import List, Tuple, Optional, Any, Dict
import ollama
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
try:
    from langchain_ollama import OllamaEmbeddings
except ModuleNotFoundError:
    print("Module 'langchain_ollama' not found. Please install with 'pip install langchain-ollama'")
    sys.exit(1)
from langchain.schema import Document
import pytesseract
from PIL import Image
import aspose.words as aw
from pptx import Presentation
import pandas as pd
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI  # Added OpenAI integration

# Configure logging
logging.basicConfig(level=logging.INFO)
RAG_SOURCE_FOLDER = "RAG_source"
os.makedirs(RAG_SOURCE_FOLDER, exist_ok=True)

# Global session state
SESSION_STATE = {
    "vectorstore": None,
    "retriever": None,
    "text_splitter": None,
    "processed_files": set()
}

# ---------------------- Core Functions (Improved) ----------------------
def detect_file_type(file_path: str) -> Optional[str]:
    """Improved MIME type detection with fallback"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        ext = os.path.splitext(file_path)[1].lower()
        type_map = {
            '.epub': 'application/epub+zip',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.txt': 'text/plain',
            '.csv': 'text/plain',
            '.jpg': 'image/jpeg',
            '.png': 'image/png'
        }
        mime_type = type_map.get(ext, 'application/octet-stream')
    return mime_type

def convert_to_text(input_file: str) -> Optional[str]:
    """Improved conversion with error handling"""
    converters = {
        'application/pdf': lambda f: "".join(page.get_text() for page in fitz.open(f)),
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': convert_pptx_to_text,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': convert_docx_to_text,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': convert_xlsx_to_text,
        'application/epub+zip': convert_epub_to_text,
        'text/plain': convert_text_file_to_text
    }
    
    mime_type = detect_file_type(input_file)
    if mime_type.startswith('image/'):
        return convert_image_to_text(input_file)
    
    converter = converters.get(mime_type)
    if not converter:
        return f"Unsupported file type: {mime_type}"
    
    try:
        return converter(input_file)
    except Exception as e:
        return f"Conversion error: {str(e)}"

# ---------------------- New Features Implementation ----------------------
def process_documents(file_paths: List[str], export_filename: str = None) -> bool:
    """Enhanced processing with export capability"""
    all_chunks = []
    all_texts = []
    
    # Check if we need to reprocess files
    new_files = [f for f in file_paths if f not in SESSION_STATE["processed_files"]]
    if not new_files:
        return True

    try:
        embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        for file_path in new_files:
            # Copy instead of move to handle temp files
            file_name = os.path.basename(file_path)
            saved_path = os.path.join(RAG_SOURCE_FOLDER, file_name)
            shutil.copy(file_path, saved_path)

            # Convert and process
            text_content = convert_to_text(saved_path)
            if text_content.startswith("Error"):
                logging.error(text_content)
                continue

            # Store for export
            all_texts.append(f"\n\n=== FILE: {file_name} ===\n{text_content}")
            
            # Split into chunks
            doc = Document(page_content=text_content)
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
            SESSION_STATE["processed_files"].add(file_path)

        # Handle text export
        if export_filename:
            export_path = os.path.join(RAG_SOURCE_FOLDER, export_filename)
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_texts))
            logging.info(f"Exported processed text to {export_path}")

        # Update vector store
        if all_chunks:
            vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            SESSION_STATE.update({
                "vectorstore": vectorstore,
                "retriever": vectorstore.as_retriever(),
                "text_splitter": text_splitter
            })
        return True
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        return False

def generate_response(
    question: str,
    system_prompt: str,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float
) -> str:
    """Unified response generator for both providers"""
    if not SESSION_STATE["vectorstore"]:
        return "Error: Please process documents first!"
    
    retrieved_docs = SESSION_STATE["retriever"].invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    full_prompt = f"System: {system_prompt}\nQuestion: {question}\nContext:\n{context}"
    
    if provider == "Ollama":
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={'temperature': temperature}
        )
        return response['message']['content']
    else:  # OpenAI-compatible
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"API Error: {str(e)}"

def clear_session():
    """Reset the application state"""
    SESSION_STATE.update({
        "vectorstore": None,
        "retriever": None,
        "text_splitter": None,
        "processed_files": set()
    })
    return [None]*4  # Clear file inputs

# ---------------------- Gradio Interface ----------------------
with gr.Blocks(title="Enhanced RAG Assistant") as interface:
    gr.Markdown("## 📚 Intelligent Document Analysis with RAG")
    
    with gr.Row():
        with gr.Column(scale=1):
            provider = gr.Radio(
                choices=["Ollama", "OpenAI"],
                label="LLM Provider",
                value="Ollama"
            )
            base_url = gr.Textbox(
                label="API URL",
                value="http://localhost:11434/v1",
                visible=False
            )
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Enter API key",
                visible=False
            )
            model = gr.Textbox(
                label="Model",
                value="deepseek-r1:1.5b",
                interactive=True
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.1
            )
            export_check = gr.Checkbox(
                label="Export Converted Text",
                info="Save processed text for future use"
            )
            export_name = gr.Textbox(
                label="Export Filename",
                placeholder="processed_text.txt",
                visible=False
            )
            
            # Dynamic visibility for OpenAI fields
            def toggle_provider(choice):
                return [
                    gr.Textbox(visible=choice == "OpenAI"),
                    gr.Textbox(visible=choice == "OpenAI"),
                    gr.Textbox(value="deepseek-ai/deepseek-r1" if choice == "OpenAI" else "deepseek-r1:1.5b")
                ]
            
            provider.change(
                toggle_provider,
                inputs=provider,
                outputs=[base_url, api_key, model]
            )
            export_check.change(
                lambda x: gr.Textbox(visible=x),
                inputs=export_check,
                outputs=export_name
            )

        with gr.Column(scale=2):
            file_input = gr.Files(label="Upload Documents", file_count="multiple")
            system_prompt = gr.Textbox(
                label="System Prompt",
                placeholder="You are a helpful assistant...",
                lines=3
            )
            question = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the documents...",
                lines=3
            )
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear Session")
            
            output = gr.Textbox(label="AI Response", interactive=False)

    # Event handling
    submit_btn.click(
        fn=process_documents,
        inputs=[file_input, export_name],
        outputs=[]
    ).success(
        fn=generate_response,
        inputs=[
            question,
            system_prompt,
            provider,
            base_url,
            api_key,
            model,
            temperature
        ],
        outputs=output
    )
    
    clear_btn.click(
        fn=clear_session,
        inputs=[],
        outputs=[file_input, question, system_prompt, export_name]
    )

if __name__ == '__main__':
    interface.launch()
