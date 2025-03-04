# Enhanced RAG Assistant Technical Design Specification

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation & Dependencies](#installation--dependencies)
4. [Core Components](#core-components)
5. [Implementation Details](#implementation-details)
6. [User Interface](#user-interface)
7. [Performance Considerations](#performance-considerations)
8. [Security & Privacy](#security--privacy)
9. [Testing & Quality Assurance](#testing--quality-assurance)
10. [Conclusion & Recommendations](#conclusion--recommendations)

## Overview

The Enhanced RAG Assistant is a sophisticated Retrieval Augmented Generation system that combines document processing, vector storage, and large language models to provide accurate question-answering capabilities. The system supports multiple document formats and offers flexible configuration options for both local and cloud-based LLM providers.

### Key Features
- Multi-format document processing (PDF, DOCX, PPTX, TXT, CSV, EPUB, images)
- Hybrid retrieval system combining vector search and BM25
- Support for both local (Ollama) and cloud (OpenAI) LLM providers
- Advanced document chunking and embedding strategies
- Configurable RAG parameters and generation settings
- Modern web interface using Gradio
- Comprehensive logging and error handling

## System Architecture

### High-Level Architecture
```
Enhanced RAG Assistant
├── Document Processing Layer
│   ├── File Type Detection
│   ├── Text Extraction
│   └── Metadata Extraction
├── Vector Storage Layer
│   ├── Text Chunking
│   ├── Embedding Generation
│   └── Vector Storage (Chroma)
├── Retrieval Layer
│   ├── Vector Search
│   ├── BM25 Search
│   └── Hybrid Retrieval
├── LLM Integration Layer
│   ├── Ollama Provider
│   └── OpenAI Provider
└── User Interface Layer
    └── Gradio Web Interface
```

## Installation & Dependencies

### Core Dependencies
```python
# Core system dependencies
import os, sys, re, logging, shutil, mimetypes, json, time, argparse
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict, Union
from datetime import datetime

# External dependencies
import yaml
import requests
import numpy as np
from tqdm import tqdm
import gradio as gr
```

### Optional Dependencies
The system supports multiple document formats through optional dependencies:

```python
# Document processing dependencies
try:
    import fitz  # PyMuPDF for PDF processing
    import pytesseract  # OCR for images
    from PIL import Image
    import aspose.words as aw  # For DOCX
    from pptx import Presentation  # For PPTX
    import pandas as pd  # For Excel/CSV
    import ebooklib  # For EPUB
    from bs4 import BeautifulSoup
except ImportError:
    # Graceful degradation if dependencies missing
```

### LLM Dependencies
```python
# LLM and vectorstore dependencies
try:
    import ollama
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings
    from langchain.schema import Document
except ImportError:
    print("LangChain dependencies not found")
```

## Core Components

### 1. Configuration Management

The system uses a YAML-based configuration system with smart defaults:

```python
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
    "rag": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "k_retrieval": 5,
        "system_prompt": "..."
    }
    # Additional settings...
}
```

### 2. Session State Management

The system maintains state through a SessionState class:

```python
class SessionState:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.text_splitter = None
        self.processed_files = {}
        self.config = self.load_config()
        # Additional state variables...
```

### 3. Document Processing Pipeline

The document processing pipeline consists of several stages:

#### a. File Type Detection
```python
def detect_file_type(file_path: str) -> str:
    """Improved MIME type detection with fallback"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        ext = os.path.splitext(file_path)[1].lower()
        type_map = {
            '.epub': 'application/epub+zip',
            '.pdf': 'application/pdf',
            # Additional mappings...
        }
        mime_type = type_map.get(ext, 'application/octet-stream')
    return mime_type
```

#### b. Text Extraction
The system supports multiple document formats through specialized converters:

```python
def convert_to_text(input_file: str) -> str:
    """Convert file to text based on its MIME type."""
    converters = {
        'application/pdf': convert_pdf_to_text,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': convert_pptx_to_text,
        # Additional converters...
    }
    
    mime_type = detect_file_type(input_file)
    converter = converters.get(mime_type)
    if not converter:
        return f"Unsupported file type: {mime_type}"
    
    return converter(input_file)
```

### 4. Vector Storage and Retrieval

#### a. Document Chunking
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)
```

#### b. Hybrid Retrieval System
```python
def generate_response(question: str, system_prompt: Optional[str] = None):
    try:
        # Implement hybrid retrieval
        from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
        
        # Combine vector and BM25 retrieval
        ensemble_retriever = EnsembleRetriever(
            retrievers=[SESSION.retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        retrieved_docs = ensemble_retriever.get_relevant_documents(question)
```

### 5. LLM Integration

#### a. Ollama Integration
```python
def generate_ollama_response(question: str, context: str, system_prompt: str) -> Dict[str, str]:
    """Generate response using Ollama API"""
    ollama_base_url = SESSION.config["ollama"]["base_url"]
    ollama_model = SESSION.config["ollama"]["model"]
    
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
    # Implementation details...
```

#### b. OpenAI Integration
```python
def generate_openai_response(question: str, context: str, system_prompt: str) -> Dict[str, str]:
    """Generate response using OpenAI API"""
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
    # Implementation details...
```

## User Interface

The system provides a modern web interface using Gradio with three main tabs:

### 1. Chat Interface
```python
with gr.Tabs() as tabs:
    with gr.TabItem("Chat"):
        with gr.Row():
            with gr.Column(scale=2):
                # Document upload and chat interface
            with gr.Column(scale=1):
                # Status and history
```

### 2. Settings Interface
```python
with gr.TabItem("Settings"):
    with gr.Group():
        gr.Markdown("### LLM Provider Settings")
        provider_select = gr.Radio(
            choices=["Ollama", "OpenAI"],
            label="Select Provider"
        )
        # Additional settings...
```

## Performance Considerations

### 1. Caching
The system implements multiple caching strategies:
- Document processing cache
- Response cache
- Embedding cache

### 2. Async Processing
Future improvements could include:
- Async document processing
- Parallel embedding generation
- Streaming responses

## Security & Privacy

### 1. Input Validation
```python
def sanitize_input(query: str) -> str:
    """Enhanced input sanitization with content moderation"""
    try:
        # Content moderation
        if SESSION.content_moderator:
            result = SESSION.content_moderator(sanitized)[0]
            if result["label"] == "toxic" and result["score"] > 0.7:
                raise PermissionError("Query violates content policy")
```

### 2. File Security
- Secure file handling
- Sanitized filenames
- Access control

## Testing & Quality Assurance

### 1. Logging
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
```

### 2. Error Handling
- Comprehensive try-except blocks
- Graceful degradation
- User feedback

## Conclusion & Recommendations

### Current Strengths
1. Robust document processing pipeline
2. Flexible LLM provider integration
3. Advanced retrieval system
4. Modern user interface

### Recommendations for Future Improvements
1. Implement async processing for better scalability
2. Add document preprocessing options (e.g., OCR quality settings)
3. Enhance the hybrid retrieval system with more algorithms
4. Add support for streaming responses
5. Implement user authentication and multi-user support
6. Add document versioning and change tracking
7. Enhance the caching system with Redis or similar
8. Add support for custom embedding models

### Performance Optimization Opportunities
1. Implement batch processing for large document sets
2. Add document preprocessing queue
3. Optimize vector store for larger datasets
4. Implement progressive loading in UI
5. Add support for distributed processing

This technical specification provides a comprehensive overview of the Enhanced RAG Assistant system. The modular architecture and extensive configuration options make it highly adaptable to different use cases while maintaining robust performance and security features.

For optimal results, regular monitoring of system performance and user feedback should guide future development priorities. The system's architecture allows for easy extension and modification to meet evolving requirements.
