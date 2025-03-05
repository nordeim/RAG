```python
# File simple_RAG-v14.py
import os
import sys
import re
import logging
import shutil
import mimetypes
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict, Union
from datetime import datetime

# Core dependencies
import yaml
import requests
import numpy as np
from tqdm import tqdm

# UI
import gradio as gr

# Document processing
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None

try:
    import aspose.words as aw
except ImportError:
    aw = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError:
    epub = None

# LLM and vectorstore
try:
    import ollama
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings
    from langchain.schema import Document
except ImportError:
    print("LangChain dependencies not found. Install with 'pip install langchain langchain-community langchain-ollama'")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not found. OpenAI integration will not be available.")

# Enhanced RAG dependencies
try:
    from transformers import pipeline
    from sentence_transformers import CrossEncoder
except ImportError:
    print("Some enhancement dependencies missing. Install with 'pip install sentence-transformers transformers'")

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"rag_assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RAG_Assistant")

# Application constants
APP_NAME = "Enhanced RAG Assistant"
APP_VERSION = "2.0.0"
DATA_DIR = "rag_data"
RAG_SOURCE_FOLDER = os.path.join(DATA_DIR, "sources")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db")
CONFIG_FILE = os.path.join(DATA_DIR, "config.yaml")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SESSION_STATE_FILE = os.path.join(DATA_DIR, "session_state.json")

# Create necessary directories
for directory in [DATA_DIR, RAG_SOURCE_FOLDER, VECTOR_DB_PATH, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama2",
        "embedding_model": "nomic-embed-text"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-3.5-turbo"
    },
    "ui": {
        "theme": "default",
        "default_provider": "Ollama",
        "max_history": 20
    },
    "rag": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "k_retrieval": 5,
        "system_prompt": "You are a helpful assistant that provides accurate information based strictly on the provided context. If unsure, respond with 'I don't have enough information to answer this.'"
    },
    "generation": {
        "temperature": 0.7,
        "max_tokens": 1024
    },
    "enhancements": {
        "content_moderation_model": "unitary/toxic-bert",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    },
    "security": {
        "max_file_size_mb": 100,
        "allowed_extensions": [".pdf", ".docx", ".txt", ".pptx", ".xlsx", ".csv", ".epub", ".md", ".jpg", ".png", ".jpeg"]
    }
}

# Global session state
class SessionState:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.text_splitter = None
        self.processed_files = {}  # Changed from set to dict
        self.config = self.load_config()
        self.query_history = []
        self.response_cache = {}
        self.current_status = "Ready"
        self.last_error = None
        self.embedding_dim = None  # Add new field
        self.sanitized_queries = set()
        self.content_moderator = None
        self.hybrid_retriever = None
        self.meta_cache = {}
    
    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {CONFIG_FILE}")
                
                # Merge with defaults to ensure all keys exist
                merged_config = DEFAULT_CONFIG.copy()
                for section, values in config.items():
                    if section in merged_config:
                        merged_config[section].update(values)
                    else:
                        merged_config[section] = values
                
                return merged_config
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return DEFAULT_CONFIG.copy()
        else:
            # Save default config
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict) -> None:
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {CONFIG_FILE}")
            self.config = config
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def update_status(self, status: str) -> None:
        """Update current status"""
        self.current_status = status
        logger.info(f"Status: {status}")
    
    def reset(self) -> None:
        """Reset session state"""
        self.vectorstore = None
        self.retriever = None
        self.processed_files = {}
        self.query_history = []
        self.response_cache = {}
        self.current_status = "Ready"
        self.last_error = None
        self.embedding_dim = None
        self.sanitized_queries = set()
        self.content_moderator = None
        logger.info("Session state reset")
    
    def add_to_history(self, query: str, response: str) -> None:
        """Add query and response to history"""
        max_history = self.config["ui"]["max_history"]
        self.query_history.append({"query": query, "response": response, "timestamp": datetime.now().isoformat()})
        if len(self.query_history) > max_history:
            self.query_history = self.query_history[-max_history:]
    
    def save_state(self) -> None:
        """Save session state to file for persistence"""
        try:
            state_dict = {
                "processed_files": self.processed_files,
                "embedding_dim": self.embedding_dim
            }
            with open(SESSION_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state_dict, f)
            logger.info("Session state saved")
        except Exception as e:
            logger.error(f"Error saving session state: {str(e)}")

# Initialize session state
SESSION = SessionState()

# ---------------------- Utility Functions ----------------------
def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        base_url = SESSION.config["ollama"]["base_url"]
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        else:
            logger.error(f"Failed to get Ollama models: {response.text}")
            return ["llama2"]  # Default fallback
    except Exception as e:
        logger.error(f"Error getting Ollama models: {str(e)}")
        return ["llama2"]  # Default fallback

def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file"""
    import hashlib
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def detect_file_type(file_path: str) -> str:
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
            '.csv': 'text/csv',
            '.md': 'text/markdown',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = type_map.get(ext, 'application/octet-stream')
    return mime_type

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove special characters"""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def validate_file(file_path: str) -> Tuple[bool, str]:
    """Validate file before processing"""
    # Check existence and type
    if not os.path.exists(file_path):
        return False, "File does not exist"
    if not os.path.isfile(file_path):
        return False, "Not a regular file"
    
    # Check size
    max_size = SESSION.config["security"]["max_file_size_mb"] * 1024 * 1024
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        return False, f"File too large ({format_file_size(file_size)})"
    
    # Check extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SESSION.config["security"]["allowed_extensions"]:
        return False, f"Unsupported file extension: {ext}"
    
    return True, "File is valid"

# ---------------------- Document Processing ----------------------
def convert_pdf_to_text(file_path: str) -> str:
    """Extract text from a PDF file with metadata."""
    try:
        doc = fitz.open(file_path)
        metadata = {}
        
        # Extract metadata
        if hasattr(doc, "metadata"):
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata = {
                    "title": pdf_metadata.get("title", ""),
                    "author": pdf_metadata.get("author", ""),
                    "subject": pdf_metadata.get("subject", ""),
                    "keywords": pdf_metadata.get("keywords", ""),
                    "page_count": len(doc)
                }
        
        # Store metadata
        file_hash = calculate_md5(file_path)
        SESSION.meta_cache[file_hash] = metadata
        
        # Extract text with page numbers
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_parts.append(f"Page {page_num + 1}:\n{text}")
        
        # Try OCR if no text found
        if not text_parts and pytesseract is not None:
            return extract_text_from_pdf_with_ocr(file_path, doc)
            
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error converting PDF to text: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_pdf_with_ocr(file_path: str, doc) -> str:
    """Extract text from PDF using OCR when text extraction fails"""
    try:
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            if text.strip():
                text_parts.append(f"Page {page_num + 1}:\n{text}")
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting text from PDF with OCR: {str(e)}")
        return f"Error extracting text from PDF with OCR: {str(e)}"

def convert_pptx_to_text(file_path: str) -> str:
    """Extract text from a PPTX file."""
    try:
        prs = Presentation(file_path)
        text = []
        for slide_num, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)
            if slide_text:
                text.append(f"Slide {slide_num + 1}: " + "\n".join(slide_text))
        return "\n\n".join(text)
    except Exception as e:
        logger.error(f"Error converting PPTX to text: {str(e)}")
        return f"Error extracting text from PPTX: {str(e)}"

def convert_docx_to_text(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        if 'aspose.words' in sys.modules:
            doc = aw.Document(file_path)
            return doc.get_text()
        else:
            # Fallback using python-docx if available
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                return "Aspose.Words and python-docx not available for DOCX conversion."
    except Exception as e:
        logger.error(f"Error converting DOCX to text: {str(e)}")
        return f"Error extracting text from DOCX: {str(e)}"

def convert_xlsx_to_text(file_path: str) -> str:
    """Extract text from an Excel file using pandas."""
    try:
        text_parts = []
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text_parts.append(f"Sheet: {sheet_name}\n{df.to_csv(index=False)}")
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error converting XLSX to text: {str(e)}")
        return f"Error extracting text from XLSX: {str(e)}"

def convert_csv_to_text(file_path: str) -> str:
    """Extract text from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Error converting CSV to text: {str(e)}")
        return f"Error extracting text from CSV: {str(e)}"

def convert_epub_to_text(file_path: str) -> str:
    """Extract text from an EPUB file."""
    try:
        book = epub.read_epub(file_path)
        texts = []
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                texts.append(soup.get_text())
        return "\n\n".join(texts)
    except Exception as e:
        logger.error(f"Error converting EPUB to text: {str(e)}")
        return f"Error extracting text from EPUB: {str(e)}"

def convert_text_file_to_text(file_path: str) -> str:
    """Read plain text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return "Could not decode file with any encoding."
    except Exception as e:
        logger.error(f"Error reading text file: {str(e)}")
        return f"Error reading text file: {str(e)}"

def convert_image_to_text(file_path: str) -> str:
    """Extract text from an image using pytesseract."""
    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return f"Error extracting text from image: {str(e)}"

def convert_to_text(input_file: str) -> str:
    """Convert file to text based on its MIME type."""
    SESSION.update_status(f"Converting {os.path.basename(input_file)} to text...")
    
    converters = {
        'application/pdf': convert_pdf_to_text,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': convert_pptx_to_text,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': convert_docx_to_text,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': convert_xlsx_to_text,
        'application/epub+zip': convert_epub_to_text,
        'text/plain': convert_text_file_to_text,
        'text/csv': convert_csv_to_text,
        'text/markdown': convert_text_file_to_text
    }
    
    mime_type = detect_file_type(input_file)
    
    # Handle image files
    if mime_type and mime_type.startswith('image/'):
        return convert_image_to_text(input_file)
    
    converter = converters.get(mime_type)
    if not converter:
        logger.warning(f"Unsupported file type: {mime_type} for {input_file}")
        return f"Unsupported file type: {mime_type}"
    
    try:
        text = converter(input_file)
        logger.info(f"Successfully converted {input_file}")
        return text
    except Exception as e:
        logger.error(f"Conversion error for {input_file}: {str(e)}")
        return f"Conversion error: {str(e)}"

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from file"""
    try:
        stat = os.stat(file_path)
        metadata = {
            "filename": os.path.basename(file_path),
            "file_type": detect_file_type(file_path),
            "file_size": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "md5": calculate_md5(file_path)
        }
        
        # Extract additional metadata based on file type
        if metadata["file_type"] == "application/pdf":
            try:
                doc = fitz.open(file_path)
                metadata.update({
                    "page_count": len(doc),
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "keywords": doc.metadata.get("keywords", "")
                })
            except Exception as e:
                logger.error(f"Error extracting PDF metadata: {str(e)}")
        
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {
            "filename": os.path.basename(file_path),
            "error": str(e),
            "file_type": detect_file_type(file_path)
        }

def get_collection_name(embedding_model: str) -> str:
    """Generate a unique collection name based on embedding model"""
    sanitized_model = re.sub(r'[^a-zA-Z0-9-]', '_', embedding_model)
    return f"rag-collection-{sanitized_model}"

def validate_embedding_dimensions(embeddings) -> int:
    """Validate embedding dimensions and return the dimension size"""
    try:
        # Get dimension from test embedding
        test_dim = len(embeddings.embed_query("test"))
        logger.info(f"Detected embedding dimension: {test_dim}")
        return test_dim
    except Exception as e:
        logger.error(f"Error validating embedding dimensions: {str(e)}")
        raise ValueError(f"Failed to validate embedding dimensions: {str(e)}")

def process_documents(file_paths: List[str], export_filename: str = None) -> Dict[str, Any]:
    """Process documents and build vector store"""
    if not file_paths:
        return {"status": "error", "message": "No files provided"}
    
    SESSION.update_status("Processing documents...")
    result = {
        "status": "success",
        "processed_files": [],
        "skipped_files": [],
        "errors": []
    }
    
    all_chunks = []
    all_texts = []
    
    # Check for new or modified files
    new_files = []
    for fp in file_paths:
        file_hash = calculate_md5(fp)
        if fp not in SESSION.processed_files or SESSION.processed_files[fp] != file_hash:
            new_files.append(fp)
            SESSION.processed_files[fp] = file_hash
    
    if not new_files:
        SESSION.update_status("No new files to process")
        return {"status": "info", "message": "No new files to process"}
    
    # Initialize text splitter
    chunk_size = SESSION.config["rag"]["chunk_size"]
    chunk_overlap = SESSION.config["rag"]["chunk_overlap"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    SESSION.text_splitter = text_splitter
    
    try:
        # Initialize embeddings
        ollama_base_url = SESSION.config["ollama"]["base_url"]
        ollama_model = SESSION.config["ollama"]["embedding_model"]
        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_model
        )
        
        # Validate embedding dimensions
        embedding_dim = validate_embedding_dimensions(embeddings)
        collection_name = get_collection_name(ollama_model)
        
        # Process each file
        for file_path in tqdm(new_files, desc="Processing files"):
            try:
                # Copy file to source folder
                file_name = os.path.basename(file_path)
                sanitized_name = sanitize_filename(file_name)
                saved_path = os.path.join(RAG_SOURCE_FOLDER, sanitized_name)
                shutil.copy2(file_path, saved_path)
                
                # Extract metadata
                metadata = extract_metadata(saved_path)
                
                # Convert to text
                text_content = convert_to_text(saved_path)
                if text_content.startswith("Error") or text_content.startswith("Unsupported"):
                    result["errors"].append({"file": file_name, "error": text_content})
                    continue
                
                # Store for export
                all_texts.append(f"\n\n=== FILE: {file_name} ===\n{text_content}")
                
                # Split into chunks with metadata
                chunks = text_splitter.create_documents(
                    texts=[text_content],
                    metadatas=[{
                        "source": file_name,
                        "file_type": metadata["file_type"],
                        "chunk": i
                    } for i in range(1)]  # This will be expanded by the splitter
                )
                
                # Update chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk"] = i + 1
                    chunk.metadata["total_chunks"] = len(chunks)
                
                all_chunks.extend(chunks)
                SESSION.processed_files[file_path] = file_hash
                result["processed_files"].append(file_name)
                
            except Exception as e:
                error_msg = f"Error processing {file_name}: {str(e)}"
                logger.error(error_msg)
                result["errors"].append({"file": file_name, "error": str(e)})
        
        # Export combined text if requested
        if export_filename and all_texts:
            export_path = os.path.join(RAG_SOURCE_FOLDER, sanitize_filename(export_filename))
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_texts))
            logger.info(f"Exported processed text to {export_path}")
        
        # Update vector store if we have new chunks
        if all_chunks:
            SESSION.update_status(f"Building vector store with {len(all_chunks)} chunks...")
            
            # Check if we should update existing store or create new one
            if SESSION.vectorstore is None:
                # Creating new vector store
                vectorstore = Chroma.from_documents(
                    documents=all_chunks,
                    embedding=embeddings,
                    persist_directory=VECTOR_DB_PATH,
                    collection_name=collection_name
                )
                SESSION.vectorstore = vectorstore
                SESSION.embedding_dim = embedding_dim
            else:
                # Validate existing vector store
                if SESSION.embedding_dim != embedding_dim:
                    error_msg = (
                        f"Embedding dimension mismatch. Current: {embedding_dim}, "
                        f"Expected: {SESSION.embedding_dim}. Please reset the session "
                        "or use a compatible embedding model."
                    )
                    logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
                
                # Add new documents to existing store
                SESSION.vectorstore.add_documents(all_chunks)
            
            # Update retriever with new configuration
            k_retrieval = SESSION.config["rag"]["k_retrieval"]
            SESSION.retriever = SESSION.vectorstore.as_retriever(
                search_kwargs={"k": k_retrieval}
            )
            
            SESSION.update_status(f"Vector store updated with {len(all_chunks)} chunks from {len(result['processed_files'])} files")
            result["chunk_count"] = len(all_chunks)
        
    except Exception as e:
        error_msg = str(e)
        if "dimension" in error_msg.lower():
            error_msg = (
                f"Embedding dimension error: {error_msg}\n\n"
                "Suggested fixes:\n"
                "1. Click 'Clear All' to reset the session\n"
                "2. Ensure consistent embedding model settings\n"
                "3. Check Ollama server status"
            )
        logger.error(f"Document processing failed: {error_msg}")
        SESSION.last_error = error_msg
        return {"status": "error", "message": error_msg}
    
    return result

# ---------------------- LLM Integration ----------------------
def sanitize_input(query: str) -> str:
    """Enhanced input sanitization with content moderation"""
    sanitized = query.strip().replace("\n", " ")
    
    try:
        # Check for repeated queries
        if sanitized in SESSION.sanitized_queries:
            raise ValueError("Duplicate query detected")
        
        # Initialize content moderator if needed
        if SESSION.content_moderator is None:
            try:
                SESSION.content_moderator = pipeline(
                    "text-classification",
                    model=SESSION.config["enhancements"]["content_moderation_model"]
                )
            except Exception as e:
                logger.error(f"Failed to initialize content moderator: {e}")
                # Continue without moderation if it fails
                return sanitized
        
        # Content moderation
        if SESSION.content_moderator:
            result = SESSION.content_moderator(sanitized)[0]
            if result["label"] == "toxic" and result["score"] > 0.7:
                raise PermissionError("Query violates content policy")
        
        SESSION.sanitized_queries.add(sanitized)
        return sanitized
        
    except (ValueError, PermissionError) as e:
        logger.warning(f"Input validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in input sanitization: {e}")
        raise

def generate_response(
    question: str,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a response using the configured LLM"""
    
    try:
        question = sanitize_input(question)
    except (ValueError, PermissionError) as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Sanitization Error: {str(e)}"}

    if SESSION.vectorstore is None or SESSION.retriever is None:
        return {
            "status": "error", 
            "message": "Please process documents first before asking questions"
        }
    
    # Use defaults from configuration if not specified
    if system_prompt is None:
        system_prompt = SESSION.config["rag"]["system_prompt"]
    
    if provider is None:
        provider = SESSION.config["ui"]["default_provider"]
    
    # Check cache for identical question
    cache_key = f"{question}|{system_prompt}|{provider}"
    if cache_key in SESSION.response_cache:
        logger.info(f"Returning cached response for: {question}")
        return SESSION.response_cache[cache_key]
    
    try:
        SESSION.update_status(f"Retrieving context for: {question}")
        
        # Implement hybrid retrieval if BM25 is available
        try:
            from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
            
            # Get all documents from vectorstore
            all_docs = SESSION.vectorstore.get()
            if all_docs and "documents" in all_docs:
                # Extract document texts properly
                doc_texts = []
                doc_metadata = []
                
                for doc, metadata in zip(all_docs["documents"], all_docs["metadatas"]):
                    if isinstance(doc, (str, bytes)):
                        doc_texts.append(str(doc))
                    elif hasattr(doc, "page_content"):
                        doc_texts.append(doc.page_content)
                    doc_metadata.append(metadata)
                
                if doc_texts:
                    # Create BM25 retriever
                    bm25_retriever = BM25Retriever.from_texts(
                        texts=doc_texts,
                        metadatas=doc_metadata
                    )
                    bm25_retriever.k = SESSION.config["rag"]["k_retrieval"]
                    
                    # Create ensemble retriever
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[SESSION.retriever, bm25_retriever],
                        weights=[0.7, 0.3]
                    )
                    retrieved_docs = ensemble_retriever.get_relevant_documents(question)
                else:
                    retrieved_docs = SESSION.retriever.invoke(question)
            else:
                retrieved_docs = SESSION.retriever.invoke(question)
                
        except (ImportError, Exception) as e:
            logger.warning(f"Falling back to standard retrieval due to error: {str(e)}")
            retrieved_docs = SESSION.retriever.invoke(question)
        
        if not retrieved_docs:
            context = "No relevant information found in the documents."
        else:
            # Format context with sources
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "Unknown")
                chunk = doc.metadata.get("chunk", "Unknown")
                total = doc.metadata.get("total_chunks", "Unknown")
                context_parts.append(f"[Document {i+1}: {source} (Chunk {chunk}/{total})]:\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
        
        SESSION.update_status(f"Generating response using {provider}...")
        
        # Generate response using selected provider
        if provider == "Ollama":
            response = generate_ollama_response(question, context, system_prompt)
        else:  # OpenAI
            response = generate_openai_response(question, context, system_prompt)
        
        # Format the final result
        result = {
            "status": "success",
            "question": question,
            "response": response["content"],
            "model": response["model"],
            "context": context,
            "sources": [doc.metadata.get("source", "Unknown") for doc in retrieved_docs],
            "source_chunks": [f"{doc.metadata.get('source', 'Unknown')}:{doc.metadata.get('chunk', 'Unknown')}" for doc in retrieved_docs],
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the result
        SESSION.response_cache[cache_key] = result
        
        # Add to history
        SESSION.add_to_history(question, response["content"])
        
        SESSION.update_status("Ready")
        return result
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        SESSION.last_error = error_msg
        return {
            "status": "error",
            "message": error_msg
        }

def generate_ollama_response(question: str, context: str, system_prompt: str) -> Dict[str, str]:
    """Generate response using Ollama API"""
    ollama_base_url = SESSION.config["ollama"]["base_url"]
    ollama_model = SESSION.config["ollama"]["model"]
    temperature = SESSION.config["generation"]["temperature"]
    max_tokens = SESSION.config["generation"]["max_tokens"]
    
    # Format the prompt
    full_prompt = f"Question: {question}\n\nContext:\n{context}\n\nPlease answer the question based on the provided context."
    
    # OpenAI-compatible API endpoint
    endpoint = "/api/chat"
    url = f"{ollama_base_url}{endpoint}"
    
    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return {
            "content": result["message"]["content"],
            "model": ollama_model
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API error: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response: {e.response.text}")
        raise Exception(f"Ollama API error: {str(e)}")

def generate_openai_response(question: str, context: str, system_prompt: str) -> Dict[str, str]:
    """Generate response using OpenAI API"""
    openai_base_url = SESSION.config["openai"]["base_url"]
    openai_api_key = SESSION.config["openai"]["api_key"]
    openai_model = SESSION.config["openai"]["model"]
    temperature = SESSION.config["generation"]["temperature"]
    max_tokens = SESSION.config["generation"]["max_tokens"]
    
    if not openai_api_key:
        raise Exception("OpenAI API key not configured. Please update your configuration.")
    
    # Format the prompt
    full_prompt = f"Question: {question}\n\nContext:\n{context}\n\nPlease answer the question based only on the provided context."
    
    try:
        client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)
        completion = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "content": completion.choices[0].message.content,
            "model": openai_model
        }
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise Exception(f"OpenAI API error: {str(e)}")

# ---------------------- UI Functions ----------------------
def clear_session():
    """Reset the application state"""
    # Remove vector store files if they exist
    if os.path.exists(VECTOR_DB_PATH):
        try:
            shutil.rmtree(VECTOR_DB_PATH)
            logger.info("Removed existing vector store")
        except Exception as e:
            logger.error(f"Error removing vector store: {str(e)}")
    
    SESSION.reset()
    return {
        upload_files: gr.update(value=None),
        upload_status: "Session reset. All data has been cleared."
    }

def update_config(provider, ollama_url, ollama_model, openai_url, openai_key, openai_model, 
                 system_prompt, temperature, max_tokens, chunk_size, chunk_overlap, k_retrieval):
    """Update configuration settings"""
    try:
        # Update configuration
        config = SESSION.config.copy()
        
        # Update provider settings
        config["ui"]["default_provider"] = provider
        
        # Update Ollama settings
        config["ollama"]["base_url"] = ollama_url
        config["ollama"]["model"] = ollama_model
        
        # Update OpenAI settings
        config["openai"]["base_url"] = openai_url
        config["openai"]["api_key"] = openai_key
        config["openai"]["model"] = openai_model
        
        # Update RAG settings
        config["rag"]["system_prompt"] = system_prompt
        config["rag"]["chunk_size"] = int(chunk_size)
        config["rag"]["chunk_overlap"] = int(chunk_overlap)
        config["rag"]["k_retrieval"] = int(k_retrieval)
        
        # Update generation settings
        config["generation"]["temperature"] = float(temperature)
        config["generation"]["max_tokens"] = int(max_tokens)
        
        # Save configuration
        SESSION.save_config(config)
        
        # Check if we need to rebuild the retriever with new params
        if SESSION.vectorstore:
            k_retrieval = int(k_retrieval)
            SESSION.retriever = SESSION.vectorstore.as_retriever(
                search_kwargs={"k": k_retrieval}
            )
        
        return "Configuration updated successfully"
    except Exception as e:
        error_msg = f"Error updating configuration: {str(e)}"
        logger.error(error_msg)
        return error_msg

def ask_question(question):
    """Handle question input and generate response"""
    result = generate_response(question)
    if result["status"] == "error":
        logger.error(result['message'])
        return f"Error: {result['message']}"
    return result["response"]

def update_provider_visibility(provider):
    """Update UI visibility based on selected provider"""
    if provider == "Ollama":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def load_query_history():
    """Load query history for display"""
    if not SESSION.query_history:
        return "No query history yet."
    
    history_text = []
    for i, item in enumerate(SESSION.query_history):
        history_text.append(f"Q{i+1}: {item['query']}")
        history_text.append(f"A{i+1}: {item['response']}\n")
    
    return "\n".join(history_text)

def get_file_info():
    """Get information about processed files"""
    if not SESSION.processed_files:
        return "No files processed yet."
    
    info = []
    info.append(f"**Processed Files:** {len(SESSION.processed_files)}")
    
    if SESSION.vectorstore:
        try:
            collection = SESSION.vectorstore._collection
            info.append(f"**Total Chunks:** {collection.count()}")
        except:
            pass
    
    # List files
    info.append("\n**Files:**")
    for i, file_path in enumerate(SESSION.processed_files, 1):
        file_name = os.path.basename(file_path)
        info.append(f"{i}. {file_name}")
    
    return "\n".join(info)

# Add this function before the Gradio UI section
def process_uploaded_files(files):
    """Process uploaded files and update vector store"""
    if not files:
        return "No files selected"
    
    try:
        # Convert to list of file paths if single file
        if isinstance(files, str):
            files = [files]
        
        # Process the documents
        result = process_documents(files)
        
        if result["status"] == "error":
            return f"Error: {result['message']}"
        elif result["status"] == "info":
            return result["message"]
        
        # Format success message
        processed = len(result["processed_files"])
        skipped = len(result["skipped_files"])
        errors = len(result["errors"])
        chunks = result.get("chunk_count", 0)
        
        message_parts = []
        if processed:
            message_parts.append(f"Processed {processed} files")
        if skipped:
            message_parts.append(f"Skipped {skipped} files")
        if errors:
            message_parts.append(f"Failed {errors} files")
        if chunks:
            message_parts.append(f"Created {chunks} chunks")
        
        if result["errors"]:
            message_parts.append("\n\nErrors:")
            for error in result["errors"]:
                message_parts.append(f"- {error['file']}: {error['error']}")
        
        return "\n".join(message_parts)
        
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return f"Error processing files: {str(e)}"

# ---------------------- Gradio UI ----------------------
with gr.Blocks(title=APP_NAME) as interface:
    gr.Markdown(f"# {APP_NAME} v{APP_VERSION}")
    gr.Markdown("An advanced Retrieval Augmented Generation system for document analysis and Q&A")
    
    with gr.Tabs() as tabs:
        # Main Tab - Chat Interface
        with gr.TabItem("Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### Upload Documents")
                        upload_files = gr.File(
                            file_count="multiple",
                            label="Upload files",
                            type="filepath"  # Changed from "file" to "filepath"
                        )
                        with gr.Row():
                            process_btn = gr.Button("Process Documents", variant="primary")
                            clear_btn = gr.Button("Clear All", variant="secondary")
                        upload_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Group():
                        gr.Markdown("### Ask Questions")
                        question_input = gr.Textbox(
                            label="Enter your question",
                            placeholder="What information are you looking for?",
                            lines=2
                        )
                        ask_btn = gr.Button("Ask", variant="primary")
                        answer_output = gr.Markdown(label="Answer")
                
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Document Analysis")
                        doc_info = gr.Markdown("Upload documents to see information")
                        refresh_info_btn = gr.Button("Refresh Info")
                    
                    with gr.Accordion("Query History", open=False):
                        history_display = gr.Markdown("No query history yet.")
                        refresh_history_btn = gr.Button("Refresh History")
                    
                    with gr.Accordion("Status", open=False):
                        status_display = gr.Markdown("System ready.")
        
        # Settings Tab
        with gr.TabItem("Settings"):
            with gr.Group():
                gr.Markdown("### LLM Provider Settings")
                provider_select = gr.Radio(
                    choices=["Ollama", "OpenAI"],
                    label="Select Provider",
                    value=SESSION.config["ui"]["default_provider"]
                )
                
                with gr.Group(visible=(SESSION.config["ui"]["default_provider"] == "Ollama")) as ollama_group:
                    gr.Markdown("#### Ollama Settings")
                    ollama_url = gr.Textbox(
                        label="Ollama API URL",
                        value=SESSION.config["ollama"]["base_url"],
                        placeholder="http://localhost:11434"
                    )
                    # Get available models
                    available_models = get_available_ollama_models()
                    ollama_model = gr.Dropdown(
                        label="Ollama Model",
                        choices=available_models,
                        value=SESSION.config["ollama"]["model"],
                        allow_custom_value=True
                    )
                
                with gr.Group(visible=(SESSION.config["ui"]["default_provider"] == "OpenAI")) as openai_group:
                    gr.Markdown("#### OpenAI Settings")
                    openai_url = gr.Textbox(
                        label="OpenAI API URL",
                        value=SESSION.config["openai"]["base_url"],
                        placeholder="https://api.openai.com/v1"
                    )
                    openai_key = gr.Textbox(
                        label="OpenAI API Key",
                        value=SESSION.config["openai"]["api_key"],
                        placeholder="sk-...",
                        type="password"
                    )
                    openai_model = gr.Textbox(
                        label="OpenAI Model",
                        value=SESSION.config["openai"]["model"],
                        placeholder="gpt-3.5-turbo"
                    )
            
            with gr.Group():
                gr.Markdown("### RAG Settings")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=SESSION.config["rag"]["system_prompt"],
                    lines=3
                )
                with gr.Row():
                    chunk_size = gr.Number(
                        label="Chunk Size",
                        value=SESSION.config["rag"]["chunk_size"],
                        precision=0
                    )
                    chunk_overlap = gr.Number(
                        label="Chunk Overlap",
                        value=SESSION.config["rag"]["chunk_overlap"],
                        precision=0
                    )
                    k_retrieval = gr.Number(
                        label="Number of Retrieved Chunks",
                        value=SESSION.config["rag"]["k_retrieval"],
                        precision=0
                    )
            
            with gr.Group():
                gr.Markdown("### Generation Settings")
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        value=SESSION.config["generation"]["temperature"],
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1
                    )
                    max_tokens = gr.Number(
                        label="Max Tokens",
                        value=SESSION.config["generation"]["max_tokens"],
                        precision=0
                    )
            
            save_config_btn = gr.Button("Save Settings", variant="primary")
            settings_status = gr.Markdown()
        
        # Help Tab
        with gr.TabItem("Help"):
            gr.Markdown("""
            # Enhanced RAG Assistant Help
            
            ## Overview
            
            This application provides a Retrieval Augmented Generation (RAG) system that allows you to:
            
            1. Upload documents in various formats (PDF, DOCX, PPTX, TXT, CSV, EPUB, images)
            2. Process these documents to extract their textual content
            3. Ask questions about the content of these documents
            4. Get AI-generated answers that reference the specific parts of your documents
            
            ## Using the Application
            
            ### 1. Upload and Process Documents
            
            - Click the **Upload Files** button to select files from your computer
            - Click **Process Documents** to extract text and build the knowledge base
            - The status area will show progress and any errors
            
            ### 2. Ask Questions
            
            - Type your question in the input box
            - Click **Ask** to get an answer
            - The system will:
              - Find the most relevant parts of your documents
              - Generate an answer based on this information
              - Show the answer with references to the source documents
            
            ### 3. Configure Settings
            
            In the Settings tab, you can customize:
            
            - **LLM Provider**: Choose between Ollama (local) or OpenAI
            - **RAG Settings**: Adjust how documents are processed and retrieved
            - **Generation Settings**: Control the output generation
            
            ## File Formats
            
            The system supports:
            
            - PDF documents
            - Microsoft Word (DOCX)
            - Microsoft PowerPoint (PPTX)
            - Microsoft Excel (XLSX)
            - Text files (TXT, MD)
            - EPUB e-books
            - CSV data files
            - Images (JPG, PNG) via OCR
            
            ## Troubleshooting
            
            - If document processing fails, check file formats and try again
            - For large documents, processing may take some time
            - If you get unclear answers, try reformulating your question
            - Check the logs for detailed error information
            
            ## Privacy
            
            All processing happens locally or via your configured API endpoints. Your documents are not shared with any third party unless you're using remote API services like OpenAI.
            """)
    
    # Event handlers
    process_btn.click(
        fn=process_uploaded_files,
        inputs=[upload_files],
        outputs=[upload_status]
    )
    
    clear_btn.click(
        fn=clear_session,
        inputs=None,
        outputs=[upload_files, upload_status]
    )
    
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output]
    )
    
    provider_select.change(
        fn=update_provider_visibility,
        inputs=[provider_select],
        outputs=[ollama_group, openai_group]
    )
    
    save_config_btn.click(
        fn=update_config,
        inputs=[
            provider_select,
            ollama_url,
            ollama_model,
            openai_url,
            openai_key,
            openai_model,
            system_prompt,
            temperature,
            max_tokens,
            chunk_size,
            chunk_overlap,
            k_retrieval
        ],
        outputs=[settings_status]
    )
    
    refresh_history_btn.click(
        fn=load_query_history,
        inputs=None,
        outputs=[history_display]
    )
    
    refresh_info_btn.click(
        fn=get_file_info,
        inputs=None,
        outputs=[doc_info]
    )

# ---------------------- Main Application ----------------------
def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description=f"{APP_NAME} v{APP_VERSION}")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Host: {args.host}, Port: {args.port}, Debug: {args.debug}")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
```

# Enhanced RAG Assistant Technical Design Specification (v2.0)

## 1. Overview

The Enhanced RAG Assistant is a sophisticated Retrieval Augmented Generation (RAG) system designed for efficient document analysis and question-answering. It combines a robust document processing pipeline, a hybrid retrieval mechanism leveraging both semantic and keyword-based search, and integration with powerful Large Language Models (LLMs). The system is built for flexibility, supporting various document formats, multiple LLM providers (both local and cloud-based), and extensive configuration options.  This updated specification incorporates feedback, addresses potential issues, and provides a more detailed and refined technical design.

### 1.1. Key Features (Expanded)

*   **Multi-Format Document Processing:**  Handles a wide array of document types:
    *   PDF (with OCR capabilities for scanned PDFs)
    *   DOCX (using Aspose.Words and a python-docx fallback)
    *   PPTX
    *   XLSX (sheet-by-sheet processing)
    *   CSV
    *   TXT, MD
    *   EPUB
    *   Images (JPG, PNG, JPEG, with OCR)
*   **Hybrid Retrieval System:** Combines the strengths of:
    *   **Semantic Search:**  Using vector embeddings (via OllamaEmbeddings) and ChromaDB for similarity-based retrieval.
    *   **Keyword-Based Search:**  Employing BM25 (Best Matching 25) for improved recall and handling of specific terminology.
    *   **Ensemble Retriever:**  A weighted combination of semantic and keyword search results, providing a balanced and comprehensive retrieval approach.
*   **Dual LLM Provider Support:**
    *   **Local LLM (Ollama):**  Enables offline operation, data privacy, and cost control.  Supports various models available through the Ollama API.
    *   **Cloud LLM (OpenAI):**  Provides access to powerful cloud-based models like GPT-3.5 Turbo (configurable).
*   **Advanced Document Chunking:**  Utilizes `RecursiveCharacterTextSplitter` from LangChain for intelligent text segmentation, respecting document structure and semantic boundaries.  Configurable chunk size and overlap for fine-grained control.
*   **Configurable RAG and Generation Parameters:**
    *   **System Prompt:**  Customizable to guide the LLM's response style and behavior.
    *   **Temperature:**  Controls the randomness/creativity of the generated text.
    *   **Max Tokens:**  Limits the length of the generated response.
    *   **Chunk Size/Overlap:**  Allows tuning of the document chunking process.
    *   **k_retrieval:**  Determines the number of chunks retrieved for context.
*   **Modern Web Interface (Gradio):**  Provides an intuitive and user-friendly interface for:
    *   Document Upload and Processing
    *   Question Input and Answer Display
    *   System Configuration
    *   Query History and Status Monitoring
*   **Comprehensive Logging and Error Handling:**  Facilitates debugging and issue resolution, with detailed logs stored in a dedicated directory.
*   **Security and Privacy Considerations:**
    *   File size limits and allowed file extensions.
    *   Input sanitization and content moderation.
    *   Local processing option (Ollama) for enhanced data privacy.
* **Session State Management:** Preserves processed files, embedding dimensions, and query history.
* **Metadata Extraction:** Stores and retrieves metadata like title, author, and page count for supported document types.
* **Caching:** Stores responses for identical questions to improve performance.
* **Content Moderation:** Uses a pre-trained model to flag and reject toxic queries.
* **Duplicate Query Detection:** Prevents redundant processing of the same question.

### 1.2. Improvements Over v1.0

*   **Hybrid Retrieval:**  The most significant enhancement, combining vector search with BM25 for improved accuracy and robustness.
*   **Detailed Error Handling:**  More specific error messages and improved logging.
*   **Input Sanitization:**  Enhanced to prevent potential security vulnerabilities and improve robustness.
*   **Content Moderation:** Added to filter out inappropriate queries.
*   **Configuration Management:**  More robust loading and merging of configuration settings.
*   **Session State Management:**  More comprehensive, storing embedding dimensions and processed file information.
*   **Metadata Handling:** Improved and stored in a meta-cache.
* **Code Clarity and Structure:** Improved code comments, organization and more detailed documentation.

## 2. System Architecture

### 2.1. High-Level Architecture (Revised)

```
Enhanced RAG Assistant v2.0
├── Data Ingestion & Processing Layer
│   ├── File Upload (Gradio)
│   ├── File Validation (Size, Type)
│   ├── File Type Detection (mimetypes, custom logic)
│   ├── Text Extraction (PyMuPDF, Aspose.Words, python-docx, pptx, pandas, ebooklib, pytesseract)
│   ├── Metadata Extraction (PyMuPDF, file stats)
│   └── Document Chunking (LangChain RecursiveCharacterTextSplitter)
├── Vector Storage & Retrieval Layer
│   ├── Embedding Generation (LangChain OllamaEmbeddings)
│   ├── Vector Storage (LangChain ChromaDB)
│   ├── Vector Search (ChromaDB)
│   ├── Keyword Search (LangChain BM25Retriever)
│   └── Hybrid Retrieval (LangChain EnsembleRetriever)
├── LLM Interaction Layer
│   ├── Input Sanitization & Moderation (custom logic, transformers pipeline)
│   ├── Prompt Engineering (f-strings, system prompt)
│   ├── LLM API Interaction (Ollama API, OpenAI API)
│   └── Response Generation & Formatting
└── User Interface Layer
    ├── Gradio Web Interface
    │   ├── Chat Tab (Input, Output, Document Upload, Status)
    │   ├── Settings Tab (LLM Provider, RAG Parameters, Generation Settings)
    │   └── Help Tab
    ├── Session Management (SessionState class)
    └── Configuration Management (YAML, config.yaml)
```

### 2.2. Data Flow

1.  **User Interaction:** The user interacts with the system through the Gradio web interface.
2.  **File Upload:** The user uploads one or more documents.
3.  **File Validation:** The system validates the uploaded files (size, type).
4.  **Document Processing:**
    *   **File Type Detection:** The system determines the file type using MIME types and custom logic.
    *   **Text Extraction:** The appropriate text extraction module is used based on the file type.
    *   **Metadata Extraction**: Relevant metadata is extracted from the file and saved to a metadata cache.
    *   **Chunking:** The extracted text is split into smaller chunks using `RecursiveCharacterTextSplitter`.
5.  **Vector Storage:**
    *   **Embedding Generation:**  Embeddings are generated for each chunk using the configured embedding model (default: `nomic-embed-text` via Ollama).
    *   **Vector Storage:**  Chunks and their embeddings are stored in ChromaDB.
6.  **Question Input:** The user enters a question.
7.  **Input Sanitization & Moderation:** The question is sanitized and checked for inappropriate content.
8.  **Retrieval:**
    *   **Vector Search:**  ChromaDB retrieves the most relevant chunks based on embedding similarity.
    *   **Keyword Search:** BM25 retrieves chunks based on keyword matching.
    *   **Hybrid Retrieval:**  The results from vector search and keyword search are combined.
9.  **LLM Interaction:**
    *   **Prompt Engineering:**  A prompt is constructed, including the system prompt, the user's question, and the retrieved context.
    *   **LLM API Call:** The prompt is sent to the selected LLM provider (Ollama or OpenAI).
    *   **Response Generation:** The LLM generates a response.
10. **Output:** The response, along with source document references, is displayed to the user.
11. **Session State:**  The query history, processed files, and other relevant information are stored in the session state.

## 3. Installation & Dependencies

### 3.1. Core Dependencies

```python
# Core system dependencies (Install with pip)
import os, sys, re, logging, shutil, mimetypes, json, time, argparse
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict, Union
from datetime import datetime

import yaml         # pip install pyyaml
import requests    # pip install requests
import numpy as np  # pip install numpy
from tqdm import tqdm # pip install tqdm
import gradio as gr # pip install gradio
```

### 3.2. Document Processing Dependencies

These are optional, but required for full functionality:

```python
# Document processing dependencies (Install with pip)
# Install only what's needed for your expected file types
try:
    import fitz  # PyMuPDF: pip install pymupdf
except ImportError:
    fitz = None

try:
    import pytesseract # pip install pytesseract
    from PIL import Image # pip install pillow
    # For pytesseract, you also need to install the Tesseract OCR engine:
    #  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    #  - macOS: `brew install tesseract`
    #  - Linux: `sudo apt install tesseract-ocr`
except ImportError:
    pytesseract = None

try:
    import aspose.words as aw  # pip install aspose-words
    # Requires a valid Aspose.Words license.  See: https://products.aspose.com/words/
except ImportError:
    aw = None

try:
    from pptx import Presentation  # pip install python-pptx
except ImportError:
    Presentation = None

try:
    import pandas as pd  # pip install pandas
except ImportError:
    pd = None

try:
    import ebooklib  # pip install ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except ImportError:
    epub = None
```

### 3.3. LLM and Vectorstore Dependencies

```python
# LLM and vectorstore dependencies (Install with pip)
try:
    import ollama # pip install ollama
    # Ollama also requires a local installation: https://ollama.com/
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings
    from langchain.schema import Document
    # pip install langchain langchain-community langchain-ollama chromadb

except ImportError:
    print("LangChain dependencies not found. Install with 'pip install langchain langchain-community langchain-ollama chromadb'")
    sys.exit(1)

try:
    from openai import OpenAI  # pip install openai
except ImportError:
    print("OpenAI package not found. OpenAI integration will not be available.")
```

### 3.4. Enhanced RAG Dependencies

```python
# Enhanced RAG dependencies (Install with pip)
try:
    from transformers import pipeline  # pip install transformers
    from sentence_transformers import CrossEncoder # pip install sentence-transformers
except ImportError:
    print("Some enhancement dependencies missing. Install with 'pip install sentence-transformers transformers'")
```

### 3.5. Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Create a `requirements.txt` file listing all the dependencies from sections 3.1 - 3.4)

4.  **Install External Tools:**
    *   **Tesseract OCR:** (See instructions in section 3.2)
    *   **Ollama:**  Download and install from [https://ollama.com/](https://ollama.com/).  Pull the necessary models (e.g., `ollama pull llama2`, `ollama pull nomic-embed-text`).

5.  **Configure (Optional):**  Edit `rag_data/config.yaml` to set API keys, model choices, etc.

6. **Run:**
   ```bash
   python your_script_name.py
   ```

## 4. Core Components

### 4.1. Configuration Management

The system uses a YAML-based configuration file (`config.yaml`) for managing settings. The `DEFAULT_CONFIG` dictionary provides default values, which are merged with user-provided settings from the YAML file. This ensures that all necessary configuration keys are always present.

```python
# Default configuration
DEFAULT_CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama2",
        "embedding_model": "nomic-embed-text"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-3.5-turbo"
    },
    "ui": {
        "theme": "default",
        "default_provider": "Ollama",
        "max_history": 20
    },
    "rag": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "k_retrieval": 5,
        "system_prompt": "You are a helpful assistant..."
    },
    "generation": {
        "temperature": 0.7,
        "max_tokens": 1024
    },
    "enhancements": {
        "content_moderation_model": "unitary/toxic-bert",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    },
    "security": {
        "max_file_size_mb": 100,
        "allowed_extensions": [".pdf", ".docx", ".txt", ".pptx", ".xlsx", ".csv", ".epub", ".md", ".jpg", ".png", ".jpeg"]
    }
}

# ... (SessionState class)

    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {CONFIG_FILE}")

                # Merge with defaults to ensure all keys exist
                merged_config = DEFAULT_CONFIG.copy()
                for section, values in config.items():
                    if section in merged_config:
                        merged_config[section].update(values)
                    else:
                        merged_config[section] = values

                return merged_config
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return DEFAULT_CONFIG.copy()
        else:
            # Save default config
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()

    def save_config(self, config: Dict) -> None:
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {CONFIG_FILE}")
            self.config = config  # Update the in-memory config
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
```

### 4.2. Session State Management

The `SessionState` class manages the application's state, including:

*   `vectorstore`: The ChromaDB vector store instance.
*   `retriever`: The LangChain retriever object.
*   `text_splitter`: The LangChain text splitter instance.
*   `processed_files`: A dictionary mapping file paths to their MD5 hashes.  This prevents reprocessing of unchanged files.
*   `config`: The loaded configuration dictionary.
*   `query_history`: A list of recent queries and responses.
*   `response_cache`: A dictionary caching responses to avoid redundant LLM calls.
*   `current_status`: A string indicating the current system status (e.g., "Ready", "Processing").
*   `last_error`: Stores the last error message, if any.
*   `embedding_dim`:  Stores the dimensionality of the embeddings, crucial for compatibility checks.
*   `sanitized_queries`: A set of previously sanitized queries, to detect duplicates.
*  `content_moderator`: The Hugging Face Transformers pipeline for content moderation.
*   `hybrid_retriever`: Stores the initialized hybrid retriever.
*   `meta_cache`: A dictionary that stores the extracted metadata information for files, with their MD5 as keys.

The session state is saved to and loaded from `session_state.json` to persist state across sessions.

```python
class SessionState:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.text_splitter = None
        self.processed_files = {}  # File path -> MD5 hash
        self.config = self.load_config()
        self.query_history = []
        self.response_cache = {}
        self.current_status = "Ready"
        self.last_error = None
        self.embedding_dim = None
        self.sanitized_queries = set()
        self.content_moderator = None
        self.hybrid_retriever = None
        self.meta_cache = {} # MD5 Hash -> Metadata Dict

    # ... (load_config, save_config, update_status, etc.)

    def reset(self) -> None:
        """Reset session state"""
        self.vectorstore = None
        self.retriever = None
        self.processed_files = {}
        self.query_history = []
        self.response_cache = {}
        self.current_status = "Ready"
        self.last_error = None
        self.embedding_dim = None
        self.sanitized_queries = set()
        self.content_moderator = None
        self.hybrid_retriever = None
        self.meta_cache = {} # Clear Metadata
        logger.info("Session state reset")
    
    def save_state(self) -> None:
        """Save session state to file for persistence"""
        try:
            state_dict = {
                "processed_files": self.processed_files,
                "embedding_dim": self.embedding_dim
            }
            with open(SESSION_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state_dict, f)
            logger.info("Session state saved")
        except Exception as e:
            logger.error(f"Error saving session state: {str(e)}")
```

### 4.3. Document Processing Pipeline

The document processing pipeline is the core of the data ingestion process.

#### 4.3.1. File Validation

```python
def validate_file(file_path: str) -> Tuple[bool, str]:
    """Validate file before processing"""
    # Check existence and type
    if not os.path.exists(file_path):
        return False, "File does not exist"
    if not os.path.isfile(file_path):
        return False, "Not a regular file"

    # Check size
    max_size = SESSION.config["security"]["max_file_size_mb"] * 1024 * 1024
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        return False, f"File too large ({format_file_size(file_size)})"

    # Check extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SESSION.config["security"]["allowed_extensions"]:
        return False, f"Unsupported file extension: {ext}"

    return True, "File is valid"
```

#### 4.3.2. File Type Detection

```python
def detect_file_type(file_path: str) -> str:
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
            '.csv': 'text/csv',
            '.md': 'text/markdown',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = type_map.get(ext, 'application/octet-stream')
    return mime_type
```

#### 4.3.3. Text Extraction

The `convert_to_text` function acts as a dispatcher, selecting the appropriate conversion function based on the detected file type.  Each conversion function handles a specific file format.

```python
def convert_to_text(input_file: str) -> str:
    """Convert file to text based on its MIME type."""
    SESSION.update_status(f"Converting {os.path.basename(input_file)} to text...")

    converters = {
        'application/pdf': convert_pdf_to_text,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': convert_pptx_to_text,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': convert_docx_to_text,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': convert_xlsx_to_text,
        'application/epub+zip': convert_epub_to_text,
        'text/plain': convert_text_file_to_text,
        'text/csv': convert_csv_to_text,
        'text/markdown': convert_text_file_to_text
    }

    mime_type = detect_file_type(input_file)

    # Handle image files
    if mime_type and mime_type.startswith('image/'):
        return convert_image_to_text(input_file)

    converter = converters.get(mime_type)
    if not converter:
        logger.warning(f"Unsupported file type: {mime_type} for {input_file}")
        return f"Unsupported file type: {mime_type}"

    try:
        text = converter(input_file)
        logger.info(f"Successfully converted {input_file}")
        return text
    except Exception as e:
        logger.error(f"Conversion error for {input_file}: {str(e)}")
        return f"Conversion error: {str(e)}"

# Example conversion function (PDF):
def convert_pdf_to_text(file_path: str) -> str:
    """Extract text from a PDF file with metadata."""
    try:
        doc = fitz.open(file_path)
        metadata = {}

        # Extract metadata
        if hasattr(doc, "metadata"):
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata = {
                    "title": pdf_metadata.get("title", ""),
                    "author": pdf_metadata.get("author", ""),
                    "subject": pdf_metadata.get("subject", ""),
                    "keywords": pdf_metadata.get("keywords", ""),
                    "page_count": len(doc)
                }
        # Store the metadata
        file_hash = calculate_md5(file_path)
        SESSION.meta_cache[file_hash] = metadata

        # Extract text with page numbers
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_parts.append(f"Page {page_num + 1}:\n{text}")

        # Try OCR if no text found
        if not text_parts and pytesseract is not None:
            return extract_text_from_pdf_with_ocr(file_path, doc)

        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error converting PDF to text: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"
```

#### 4.3.4. Metadata Extraction
The `extract_metadata` function retrieves basic file metadata (name, size, timestamps, MD5 hash) and, where possible, extracts format-specific metadata (e.g., PDF title, author, page count).

```python
def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from file"""
    try:
        stat = os.stat(file_path)
        metadata = {
            "filename": os.path.basename(file_path),
            "file_type": detect_file_type(file_path),
            "file_size": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "md5": calculate_md5(file_path)
        }

        # Extract additional metadata based on file type
        if metadata["file_type"] == "application/pdf":
            try:
                doc = fitz.open(file_path)
                metadata.update({
                    "page_count": len(doc),
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "keywords": doc.metadata.get("keywords", "")
                })
            except Exception as e:
                logger.error(f"Error extracting PDF metadata: {str(e)}")

        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {
            "filename": os.path.basename(file_path),
            "error": str(e),
            "file_type": detect_file_type(file_path)
        }

```

#### 4.3.5. Document Chunking

The `RecursiveCharacterTextSplitter` from LangChain is used to split the extracted text into semantically meaningful chunks.  The `chunk_size` and `chunk_overlap` parameters can be adjusted in the configuration.

```python
# Initialize text splitter (in process_documents)
chunk_size = SESSION.config["rag"]["chunk_size"]
chunk_overlap = SESSION.config["rag"]["chunk_overlap"]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Ordered list of separators
)
SESSION.text_splitter = text_splitter

# ... (Inside the file processing loop)

# Split into chunks with metadata
chunks = text_splitter.create_documents(
    texts=[text_content],
    metadatas=[{
        "source": file_name,
        "file_type": metadata["file_type"],
        "chunk": i  # Placeholder, will be updated
    } for i in range(1)]  # Initial metadata for the entire document
)

# Update chunk metadata (important for tracking)
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk"] = i + 1  # Correct chunk number
    chunk.metadata["total_chunks"] = len(chunks)
```

### 4.4. Vector Storage and Retrieval

#### 4.4.1. Embedding Generation

Embeddings are generated using `OllamaEmbeddings` from LangChain, which interfaces with the Ollama API.  The embedding model can be configured in `config.yaml`. The embedding dimensions are validated to ensure consistency with any existing vector store.

```python
# Initialize embeddings (in process_documents)
ollama_base_url = SESSION.config["ollama"]["base_url"]
ollama_model = SESSION.config["ollama"]["embedding_model"]
embeddings = OllamaEmbeddings(
    base_url=ollama_base_url,
    model=ollama_model
)

# Validate embedding dimensions
embedding_dim = validate_embedding_dimensions(embeddings)
collection_name = get_collection_name(ollama_model)
```

```python
def validate_embedding_dimensions(embeddings) -> int:
    """Validate embedding dimensions and return the dimension size"""
    try:
        # Get dimension from test embedding
        test_dim = len(embeddings.embed_query("test"))
        logger.info(f"Detected embedding dimension: {test_dim}")
        return test_dim
    except Exception as e:
        logger.error(f"Error validating embedding dimensions: {str(e)}")
        raise ValueError(f"Failed to validate embedding dimensions: {str(e)}")
```

#### 4.4.2. Vector Storage (ChromaDB)

ChromaDB is used as the vector store.  The system supports creating a new vector store or adding documents to an existing one.  A critical check is performed to ensure that the embedding dimensions of new documents match the existing vector store.

```python
# Update vector store (in process_documents)
if all_chunks:
    SESSION.update_status(f"Building vector store with {len(all_chunks)} chunks...")

    # Check if we should update existing store or create new one
    if SESSION.vectorstore is None:
        # Creating new vector store
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH,
            collection_name=collection_name  # Use a consistent collection name
        )
        SESSION.vectorstore = vectorstore
        SESSION.embedding_dim = embedding_dim # Store dimension
    else:
        # Validate existing vector store
        if SESSION.embedding_dim != embedding_dim:
            error_msg = (
                f"Embedding dimension mismatch. Current: {embedding_dim}, "
                f"Expected: {SESSION.embedding_dim}. Please reset the session "
                "or use a compatible embedding model."
            )
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Add new documents to existing store
        SESSION.vectorstore.add_documents(all_chunks)
```

#### 4.4.3. Hybrid Retrieval

This is a major improvement in v2.0.  The system implements a hybrid retrieval approach using `EnsembleRetriever` from LangChain, combining:

*   **Vector Search (Semantic):**  Uses the `as_retriever()` method of the ChromaDB vector store.
*   **BM25 Search (Keyword):**  Uses `BM25Retriever` from LangChain.

The `EnsembleRetriever` combines the results from both, using configurable weights (default: 70% vector search, 30% BM25). This leverages the strengths of both approaches, improving both precision and recall.

```python
# Implement hybrid retrieval (in generate_response)
try:
    from langchain_community.retrievers import BM25Retriever, EnsembleRetriever

    # Get all documents from vectorstore
    all_docs = SESSION.vectorstore.get() # Get ALL documents, not just k
    if all_docs and "documents" in all_docs:
        # Extract document texts properly
        doc_texts = []
        doc_metadata = []

        for doc, metadata in zip(all_docs["documents"], all_docs["metadatas"]):
          if isinstance(doc, (str, bytes)):
            doc_texts.append(str(doc))  # Ensure it's a string
          elif hasattr(doc, "page_content"): # For Langchain Document objects
            doc_texts.append(doc.page_content)
          doc_metadata.append(metadata)


        if doc_texts:
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_texts(
                texts=doc_texts,
                metadatas=doc_metadata  # Pass metadata
            )
            bm25_retriever.k = SESSION.config["rag"]["k_retrieval"]

            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[SESSION.retriever, bm25_retriever],
                weights=[0.7, 0.3]  # Configurable weights
            )
            retrieved_docs = ensemble_retriever.get_relevant_documents(question)
        else:
            retrieved_docs = SESSION.retriever.invoke(question)
    else:
        retrieved_docs = SESSION.retriever.invoke(question)

except (ImportError, Exception) as e:
    logger.warning(f"Falling back to standard retrieval due to error: {str(e)}")
    retrieved_docs = SESSION.retriever.invoke(question)  # Fallback
```

### 4.5. LLM Integration

#### 4.5.1. Input Sanitization and Moderation

Before sending the user's question to the LLM, it's crucial to sanitize the input and check for potentially harmful or inappropriate content.

```python
def sanitize_input(query: str) -> str:
    """Enhanced input sanitization with content moderation"""
    sanitized = query.strip().replace("\n", " ")  # Basic sanitization

    try:
        # Check for repeated queries
        if sanitized in SESSION.sanitized_queries:
            raise ValueError("Duplicate query detected")

        # Initialize content moderator if needed
        if SESSION.content_moderator is None:
            try:
                SESSION.content_moderator = pipeline(
                    "text-classification",
                    model=SESSION.config["enhancements"]["content_moderation_model"]
                )
            except Exception as e:
                logger.error(f"Failed to initialize content moderator: {e}")
                # Continue without moderation if it fails (but log the error)
                return sanitized

        # Content moderation
        if SESSION.content_moderator:
            result = SESSION.content_moderator(sanitized)[0]
            if result["label"] == "toxic" and result["score"] > 0.7:  # Adjustable threshold
                raise PermissionError("Query violates content policy")

        SESSION.sanitized_queries.add(sanitized)  # Add to set of seen queries
        return sanitized

    except (ValueError, PermissionError) as e:
        logger.warning(f"Input validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in input sanitization: {e}")
        raise
```

#### 4.5.2. Prompt Engineering

The `generate_response` function constructs a prompt that includes:

*   **System Prompt:**  Provides general instructions to the LLM (e.g., "You are a helpful assistant...").
*   **User's Question:**  The sanitized question from the user.
*   **Context:**  The text content of the retrieved document chunks.

```python
# Format the prompt (in generate_response, generate_ollama_response, generate_openai_response)
full_prompt = f"Question: {question}\n\nContext:\n{context}\n\nPlease answer the question based on the provided context."
```
#### 4.5.3 LLM API Interaction
The `generate_ollama_response` and `generate_openai_response` functions handle the interaction with the respective LLM APIs, constructing the API requests and parsing the responses.

```python
def generate_ollama_response(question: str, context: str, system_prompt: str) -> Dict[str, str]:
    """Generate response using Ollama API"""
    ollama_base_url = SESSION.config["ollama"]["base_url"]
    ollama_model = SESSION.config["ollama"]["model"]
    temperature = SESSION.config["generation"]["temperature"]
    max_tokens = SESSION.config["generation"]["max_tokens"]

    # Format the prompt
    full_prompt = f"Question: {question}\n\nContext:\n{context}\n\nPlease answer the question based on the provided context."

    # OpenAI-compatible API endpoint
    endpoint = "/api/chat"
    url = f"{ollama_base_url}{endpoint}"

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status() # Raise HTTPError for bad requests (4xx or 5xx)

        result = response.json()
        return {
            "content": result["message"]["content"],
            "model": ollama_model
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API error: {str(e)}")
        if hasattr(e, 'response') and e.response: # Log response details
            logger.error(f"Response: {e.response.text}")
        raise Exception(f"Ollama API error: {str(e)}")
```

```python
def generate_openai_response(question: str, context: str, system_prompt: str) -> Dict[str, str]:
    """Generate response using OpenAI API"""
    openai_base_url = SESSION.config["openai"]["base_url"]
    openai_api_key = SESSION.config["openai"]["api_key"]
    openai_model = SESSION.config["openai"]["model"]
    temperature = SESSION.config["generation"]["temperature"]
    max_tokens = SESSION.config["generation"]["max_tokens"]

    if not openai_api_key:
        raise Exception("OpenAI API key not configured. Please update your configuration.")

    # Format the prompt
    full_prompt = f"Question: {question}\n\nContext:\n{context}\n\nPlease answer the question based only on the provided context."

    try:
        client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)
        completion = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "content": completion.choices[0].message.content,
            "model": openai_model
        }
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise Exception(f"OpenAI API error: {str(e)}")
```

#### 4.5.4 Response Generation and Formatting
The LLM's response is extracted from the API response and included in the final result, along with metadata about the context (source documents, chunks).

```python
 # Format the final result (in generate_response)
    result = {
        "status": "success",
        "question": question,
        "response": response["content"],
        "model": response["model"],
        "context": context,
        "sources": [doc.metadata.get("source", "Unknown") for doc in retrieved_docs],
        "source_chunks": [f"{doc.metadata.get('source', 'Unknown')}:{doc.metadata.get('chunk', 'Unknown')}" for doc in retrieved_docs],
        "timestamp": datetime.now().isoformat()
    }
```

## 5. User Interface (Gradio)

The Gradio web interface provides a user-friendly way to interact with the system.

### 5.1. Chat Tab

*   **Document Upload:**  `gr.File` component for uploading multiple files.
*   **Process Documents Button:**  Triggers the `process_uploaded_files` function.
*   **Clear All Button:**  Resets the session state using the `clear_session` function.
*   **Question Input:**  `gr.Textbox` for entering questions.
*   **Ask Button:**  Triggers the `ask_question` function.
*   **Answer Output:** `gr.Markdown` for displaying the generated answer.
* **Document Analysis:** Shows the number of files and chunks.
*   **Query History:**  Displays the recent query history.
*   **Status:** Shows the current system status.

### 5.2. Settings Tab

*   **LLM Provider Selection:**  `gr.Radio` component to choose between Ollama and OpenAI.
*   **Ollama Settings:**  `gr.Textbox` for Ollama API URL and `gr.Dropdown` for the Ollama model.
*   **OpenAI Settings:**  `gr.Textbox` for OpenAI API URL, API key (with `type="password"`), and model.
*   **RAG Settings:**  `gr.Textbox` for the system prompt, `gr.Number` for chunk size, chunk overlap, and k\_retrieval.
*   **Generation Settings:** `gr.Slider` for temperature and `gr.Number` for max tokens.
* **Save Settings Button:** Triggers the `update_config` function.

### 5.3. Help Tab

Provides documentation and instructions for using the application.

### 5.4. Gradio Code Structure

```python
with gr.Blocks(title=APP_NAME) as interface:
    gr.Markdown(f"# {APP_NAME} v{APP_VERSION}")
    gr.Markdown("An advanced Retrieval Augmented Generation system for document analysis and Q&A")

    with gr.Tabs() as tabs:
        # Main Tab - Chat Interface
        with gr.TabItem("Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### Upload Documents")
                        upload_files = gr.File(
                            file_count="multiple",
                            label="Upload files",
                            type="filepath"  # Use "filepath" for local files
                        )
                        with gr.Row():
                            process_btn = gr.Button("Process Documents", variant="primary")
                            clear_btn = gr.Button("Clear All", variant="secondary")
                        upload_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Group():
                        gr.Markdown("### Ask Questions")
                        question_input = gr.Textbox(
                            label="Enter your question",
                            placeholder="What information are you looking for?",
                            lines=2
                        )
                        ask_btn = gr.Button("Ask", variant="primary")
                        answer_output = gr.Markdown(label="Answer")

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Document Analysis")
                        doc_info = gr.Markdown("Upload documents to see information")
                        refresh_info_btn = gr.Button("Refresh Info")

                    with gr.Accordion("Query History", open=False):
                        history_display = gr.Markdown("No query history yet.")
                        refresh_history_btn = gr.Button("Refresh History")

                    with gr.Accordion("Status", open=False):
                        status_display = gr.Markdown("System ready.")

        # Settings Tab
        with gr.TabItem("Settings"):
            with gr.Group():
                gr.Markdown("### LLM Provider Settings")
                provider_select = gr.Radio(
                    choices=["Ollama", "OpenAI"],
                    label="Select Provider",
                    value=SESSION.config["ui"]["default_provider"]
                )

                with gr.Group(visible=(SESSION.config["ui"]["default_provider"] == "Ollama")) as ollama_group:
                    gr.Markdown("#### Ollama Settings")
                    ollama_url = gr.Textbox(
                        label="Ollama API URL",
                        value=SESSION.config["ollama"]["base_url"],
                        placeholder="http://localhost:11434"
                    )
                    # Get available models (dynamic)
                    available_models = get_available_ollama_models()
                    ollama_model = gr.Dropdown(
                        label="Ollama Model",
                        choices=available_models,
                        value=SESSION.config["ollama"]["model"],
                        allow_custom_value=True  # Allow custom model names
                    )

                with gr.Group(visible=(SESSION.config["ui"]["default_provider"] == "OpenAI")) as openai_group:
                    gr.Markdown("#### OpenAI Settings")
                    openai_url = gr.Textbox(
                        label="OpenAI API URL",
                        value=SESSION.config["openai"]["base_url"],
                        placeholder="https://api.openai.com/v1"
                    )
                    openai_key = gr.Textbox(
                        label="OpenAI API Key",
                        value=SESSION.config["openai"]["api_key"],
                        placeholder="sk-...",
                        type="password"  # Hide the API key
                    )
                    openai_model = gr.Textbox(
                        label="OpenAI Model",
                        value=SESSION.config["openai"]["model"],
                        placeholder="gpt-3.5-turbo"
                    )

            with gr.Group():
                gr.Markdown("### RAG Settings")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=SESSION.config["rag"]["system_prompt"],
                    lines=3
                )
                with gr.Row():
                    chunk_size = gr.Number(
                        label="Chunk Size",
                        value=SESSION.config["rag"]["chunk_size"],
                        precision=0  # Integer value
                    )
                    chunk_overlap = gr.Number(
                        label="Chunk Overlap",
                        value=SESSION.config["rag"]["chunk_overlap"],
                        precision=0  # Integer value
                    )
                    k_retrieval = gr.Number(
                        label="Number of Retrieved Chunks",
                        value=SESSION.config["rag"]["k_retrieval"],
                        precision=0  # Integer value
                    )

            with gr.Group():
                gr.Markdown("### Generation Settings")
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        value=SESSION.config["generation"]["temperature"],
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1
                    )
                    max_tokens = gr.Number(
                        label="Max Tokens",
                        value=SESSION.config["generation"]["max_tokens"],
                        precision=0  # Integer value
                    )

            save_config_btn = gr.Button("Save Settings", variant="primary")
            settings_status = gr.Markdown()

        # Help Tab (Simplified for brevity)
        with gr.TabItem("Help"):
            gr.Markdown("...") # Help text

    # Event handlers
    process_btn.click(
        fn=process_uploaded_files,
        inputs=[upload_files],
        outputs=[upload_status]
    )

    clear_btn.click(
        fn=clear_session,
        inputs=None,
        outputs=[upload_files, upload_status]
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output]
    )

    provider_select.change(
        fn=update_provider_visibility,
        inputs=[provider_select],
        outputs=[ollama_group, openai_group]  # Show/hide settings groups
    )

    save_config_btn.click(
        fn=update_config,
        inputs=[
            provider_select,
            ollama_url,
            ollama_model,
            openai_url,
            openai_key,
            openai_model,
            system_prompt,
            temperature,
            max_tokens,
            chunk_size,
            chunk_overlap,
            k_retrieval
        ],
        outputs=[settings_status]
    )

    refresh_history_btn.click(
        fn=load_query_history,
        inputs=None,
        outputs=[history_display]
    )

    refresh_info_btn.click(
        fn=get_file_info,
        inputs=None,
        outputs=[doc_info]
    )

```

## 6. Performance Considerations

### 6.1. Memory Management

*   **Chunking:**  Processing large documents in smaller chunks prevents memory exhaustion.
*   **Resource Cleanup:**  Explicitly releasing resources (e.g., closing file handles) when they are no longer needed.

### 6.2. Optimized Vector Storage

*   **ChromaDB Indexing:**  ChromaDB automatically creates indexes for efficient similarity search.
*   **Persistence:**  The `persist_directory` option in ChromaDB ensures that the vector store is saved to disk and can be reloaded quickly.

### 6.3. Caching

*   **Response Cache:**  The `response_cache` in `SessionState` stores responses to previously asked questions, avoiding redundant LLM calls.  This significantly improves performance for repeated queries.

### 6.4. Asynchronous Operations (Future Enhancement)

*   While not implemented in the current version, using asynchronous operations (e.g., `asyncio`) could further improve performance, especially for I/O-bound tasks like network requests and file processing. This is an area for future development.

### 6.5 Batched Processing (Future Enhancement)
Processing a list of files could be potentially sped up by processing multiple files concurrently.

## 7. Security & Privacy

### 7.1. Input Validation

*   **File Size Limit:**  The `max_file_size_mb` setting in `config.yaml` prevents excessively large files from being processed, mitigating denial-of-service risks.
*   **Allowed File Extensions:**  The `allowed_extensions` setting restricts uploads to known safe file types, reducing the risk of malicious file uploads.
* **Filename Sanitization:** The `sanitize_filename` function removes potentially dangerous characters from filenames.
*   **Input Sanitization:**  The `sanitize_input` function removes potentially harmful characters and patterns from user queries.
*   **Content Moderation:**  The `content_moderator` (using a Hugging Face Transformers pipeline) filters out toxic or inappropriate queries.
* **Duplicate Query Prevention:** The `sanitized_queries` set prevents repeated, potentially resource-intensive, queries.

### 7.2. File Security

*   **Secure File Handling:**  Uploaded files are copied to a dedicated `rag_data/sources` directory, preventing direct access to the original files.
*   **Sanitized Filenames:**  Filenames are sanitized to prevent path traversal vulnerabilities.

### 7.3. Data Privacy

*   **Local LLM Option (Ollama):**  Using Ollama allows users to process data locally without sending it to external servers, enhancing privacy.
*   **OpenAI API Key Protection:**  The OpenAI API key is stored securely in `config.yaml` and handled with care in the code. The Gradio interface uses `type="password"` to prevent it from being displayed.
* **No Third-Party Data Sharing (Unless Using OpenAI):** When using Ollama, the system does not share data with any third parties. When using OpenAI, data is sent to OpenAI's servers, subject to their privacy policy.

## 8. Testing & Quality Assurance

### 8.1. Logging

*   **Comprehensive Logging:** The system uses the `logging` module to log events, errors, and debug information. Logs are written to both the console and a file in the `logs` directory, with timestamps and log levels.  This facilitates debugging and monitoring.
```python
# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"rag_assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RAG_Assistant")
```

### 8.2. Error Handling

*   **Try-Except Blocks:**  The code uses `try-except` blocks extensively to catch potential errors and handle them gracefully.  This prevents the application from crashing and provides informative error messages to the user.
*   **Specific Error Messages:**  Error messages are designed to be informative, helping users understand the cause of the problem.
*   **Graceful Degradation:**  In some cases (e.g., if a non-essential dependency is missing), the system can continue to operate with reduced functionality rather than failing completely.
*   **Status Updates:** The `update_status` method in `SessionState` and the `upload_status` Gradio component keep the user informed about the system's progress and any errors.

### 8.3. Unit Tests (Future Enhancement)

*   While not included in the current code, adding unit tests (e.g., using the `unittest` or `pytest` frameworks) would significantly improve the system's robustness and maintainability.  Tests should cover:
    *   Utility functions (e.g., `calculate_md5`, `format_file_size`, `detect_file_type`, `sanitize_filename`)
    *   Document processing functions (e.g., `convert_pdf_to_text`, `convert_docx_to_text`, etc.)
    *   LLM integration functions (e.g., `generate_ollama_response`, `generate_openai_response`)
    *   Session state management (e.g., loading, saving, resetting)
    *   Input sanitization and moderation

### 8.4. Integration Tests (Future Enhancement)

*   Integration tests would verify the interaction between different components of the system, such as the document processing pipeline, vector store, and LLM integration.

## 9. Conclusion & Recommendations

### 9.1. Current Strengths

*   **Robust and Comprehensive Document Processing:**  Handles a wide variety of file formats effectively.
*   **Flexible LLM Integration:** Supports both local (Ollama) and cloud-based (OpenAI) LLMs.
*   **Advanced Hybrid Retrieval:** Combines semantic and keyword-based search for improved accuracy.
*   **User-Friendly Interface:**  Provides a clean and intuitive web interface with Gradio.
*   **Good Configuration Management:** YAML-based configuration with sensible defaults.
*   **Security and Privacy Features:**  Includes file validation, input sanitization, and content moderation.
*   **Detailed Logging:**  Facilitates debugging and monitoring.
* **Session State Management:** Preserves data between sessions.

### 9.2. Recommendations for Future Improvements

*   **Asynchronous Operations:** Implement asynchronous processing using `asyncio` to improve performance, especially for I/O-bound operations.
*   **Unit and Integration Tests:**  Add comprehensive tests to improve code quality and maintainability.
*   **Streaming Responses:**  Implement streaming responses from the LLM to provide a more interactive user experience.
*   **Advanced Search Features:**  Add support for more advanced search features, such as filtering by metadata, date ranges, etc.
*   **User Authentication and Authorization:**  Add user accounts and access control for multi-user environments.
*   **Document Versioning:**  Implement version control for uploaded documents.
*   **Scalability Improvements:**  Explore options for scaling the system to handle larger datasets and higher query loads (e.g., distributed processing, optimized database queries).
* **More Robust Error Messages in UI:** Improve error handling to provide more detailed and user-friendly error messages in the UI.
* **Progress Bar for Processing:** Show a progress bar during document processing, especially for large files or multiple files.
* **Reranking:** Implement a reranking step after retrieval using a cross-encoder model to further refine the relevance of the retrieved chunks.
* **Feedback Mechanism:** Add a way for users to provide feedback on the quality of the answers, which could be used to improve the system over time.
* **More Detailed Documentation:** Continue expanding and refining the documentation.
* **Dockerization:** Containerize the application using Docker for easier deployment and portability.

This expanded technical design specification provides a detailed and comprehensive overview of the Enhanced RAG Assistant, version 2.0. The system is well-designed, robust, and addresses many of the limitations of a basic RAG system. The modular architecture and extensive configuration options allow for flexibility and adaptation to different use cases. The recommendations for future improvements outline a path for continued development and enhancement.
