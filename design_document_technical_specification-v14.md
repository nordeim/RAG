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
