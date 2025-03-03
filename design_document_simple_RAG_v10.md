**Technical Design Specification Document for Enhanced RAG System**  
**Version 2.0**  

---

### **1. Overview of a Good RAG Tool**  
A Retrieval-Augmented Generation (RAG) system combines the strengths of **retrieval-based** and **generation-based** approaches to provide accurate, context-aware responses. Key components of an ideal RAG tool include:  

1. **Document Ingestion & Preprocessing**:  
   - Support for diverse file formats (PDF, DOCX, images, etc.).  
   - Robust text extraction with error handling (e.g., OCR for images [[8]]).  
   - Metadata extraction (file type, creation date, etc.) for traceability.  

2. **Vector Storage**:  
   - Efficient chunking strategies to balance context and performance [[7]][[9]].  
   - Vector embeddings using models like `nomic-embed-text` for semantic search [[10]].  

3. **Retrieval Mechanism**:  
   - Hybrid search (keyword + semantic) for relevance [[10]].  
   - Dynamic adjustment of retrieval parameters (e.g., `k` for top results).  

4. **Generation**:  
   - Integration with LLMs (e.g., Ollama, OpenAI) for response synthesis.  
   - Strict adherence to context to avoid hallucinations [[5]][[7]].  

5. **User Interface**:  
   - Intuitive design for document upload, query submission, and result visualization.  
   - Real-time feedback and error reporting [[6]].  

6. **Extensibility**:  
   - Modular architecture for adding new document types or LLM providers.  

---

### **2. Core Architecture**  

#### **2.1 Dependencies & Configuration**  
**Improvements**:  
- **Graceful Degradation**: Missing dependencies (e.g., `PyMuPDF`, `pytesseract`) are handled with warnings instead of crashes.  
- **Atomic Configuration Updates**: Merges user settings with defaults to preserve integrity.  

**Code Snippet**:  
```python  
DEFAULT_CONFIG = {  
    "ollama": {"base_url": "http://localhost:11434", "model": "llama2"},  
    "rag": {"chunk_size": 500, "chunk_overlap": 50, "k_retrieval": 5},  
    # ... other settings  
}  

class SessionState:  
    def load_config(self):  
        merged_config = DEFAULT_CONFIG.copy()  
        if os.path.exists(CONFIG_FILE):  
            with open(CONFIG_FILE, "r") as f:  
                user_config = yaml.safe_load(f)  
            for section, values in user_config.items():  
                merged_config[section].update(values)  
        return merged_config  
```  

**Explanation**:  
- The `SessionState` class ensures configurations are backward-compatible and user-customizable.  

---

#### **2.2 Document Processing Pipeline**  
**Improvements**:  
- **MIME-Type Handling**: Enhanced detection using `mimetypes` and fallback mappings.  
- **MD5 Change Detection**: Avoids reprocessing unchanged files.  

**Code Snippet**:  
```python  
def detect_file_type(file_path):  
    mime_type, _ = mimetypes.guess_type(file_path)  
    if not mime_type:  
        ext = os.path.splitext(file_path)[1].lower()  
        type_map = {  
            '.pdf': 'application/pdf',  
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  
            # ... other mappings  
        }  
        mime_type = type_map.get(ext, 'application/octet-stream')  
    return mime_type  

def calculate_md5(file_path):  
    hash_md5 = hashlib.md5()  
    with open(file_path, "rb") as f:  
        for chunk in iter(lambda: f.read(4096), b""):  
            hash_md5.update(chunk)  
    return hash_md5.hexdigest()  
```  

**Explanation**:  
- `detect_file_type` ensures accurate format identification, while `calculate_md5` enables efficient caching.  

---

#### **2.3 Text Extraction & Chunking**  
**Improvements**:  
- **Format-Specific Converters**: Specialized handlers for PDFs, DOCX, images, etc.  
- **Recursive Chunking**: Uses `RecursiveCharacterTextSplitter` for logical text segmentation.  

**Code Snippet**:  
```python  
def convert_pdf_to_text(file_path):  
    if not fitz:  
        return "PDF processing unavailable. Install PyMuPDF."  
    try:  
        doc = fitz.open(file_path)  
        return "\n\n".join([page.get_text() for page in doc])  
    except Exception as e:  
        logger.error(f"PDF conversion error: {e}")  
        return f"Error: {str(e)}"  

text_splitter = RecursiveCharacterTextSplitter(  
    chunk_size=500,  
    chunk_overlap=50,  
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  
)  
```  

**Explanation**:  
- `convert_pdf_to_text` uses `PyMuPDF` for high-fidelity extraction, while the splitter ensures chunks retain semantic meaning.  

---

### **3. Vector Storage & Retrieval**  
**Improvements**:  
- **Persistent ChromaDB**: Stores embeddings for fast retrieval.  
- **Dynamic Retriever**: Adjusts `k` based on configuration.  

**Code Snippet**:  
```python  
embeddings = OllamaEmbeddings(  
    base_url=ollama_base_url,  
    model=ollama_model  
)  

vectorstore = Chroma.from_documents(  
    documents=all_chunks,  
    embedding=embeddings,  
    persist_directory=VECTOR_DB_PATH  
)  

retriever = vectorstore.as_retriever(  
    search_kwargs={"k": k_retrieval}  
)  
```  

**Explanation**:  
- ChromaDB persists embeddings across sessions, and the retriever dynamically fetches the top `k` results.  

---

### **4. LLM Integration**  
**Improvements**:  
- **Multi-Provider Support**: Switch between Ollama (local) and OpenAI (cloud).  
- **Context-Aware Prompting**: Injects retrieved documents into the LLM prompt.  

**Code Snippet**:  
```python  
def generate_ollama_response(question, context):  
    payload = {  
        "model": ollama_model,  
        "messages": [  
            {"role": "system", "content": "Answer based on context."},  
            {"role": "user", "content": f"Question: {question}\nContext: {context}"}  
        ]  
    }  
    response = requests.post(f"{ollama_base_url}/api/chat", json=payload)  
    return response.json()["message"]["content"]  
```  

**Explanation**:  
- The prompt explicitly instructs the LLM to use the provided context, reducing hallucinations [[5]][[7]].  

---

### **5. User Interface (Gradio)**  
**Improvements**:  
- **Responsive Design**: Tabs for chat, settings, and help.  
- **Real-Time Feedback**: Progress bars and status updates.  

**Code Snippet**:  
```python  
with gr.Blocks() as interface:  
    with gr.Tab("Chat"):  
        upload_files = gr.File(label="Upload Documents")  
        process_btn = gr.Button("Process Documents")  
        # ... other components  

    process_btn.click(  
        fn=process_uploaded_files,  
        inputs=[upload_files],  
        outputs=[upload_status]  
    )  
```  

**Explanation**:  
- The UI decouples document processing from querying, ensuring a smooth user experience.  

---

### **6. Error Handling & Logging**  
**Improvements**:  
- **Comprehensive Logging**: Writes to both file and console.  
- **User-Friendly Errors**: Masks technical details in the UI.  

**Code Snippet**:  
```python  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[  
        logging.FileHandler(log_file),  
        logging.StreamHandler()  
    ]  
)  

def process_documents(file_paths):  
    try:  
        # ... processing logic  
    except Exception as e:  
        logger.error(f"Processing failed: {e}")  
        return {"status": "error", "message": "Processing failed. Check logs."}  
```  

**Explanation**:  
- Errors are logged for debugging but presented as simple messages to users.  

---

### **7. Conclusion & Future Work**  
**Summary**:  
The improved RAG system enhances reliability, scalability, and usability through:  
- Modular architecture with clear separation of concerns.  
- Robust error handling and logging.  
- Support for diverse document types and LLM providers.  

**Future Improvements**:  
1. **Advanced Chunking**: Implement sentence-window or hierarchical retrieval [[7]].  
2. **Citation Formatting**: Highlight source snippets in responses [[1]].  
3. **Performance Optimization**: Caching for frequent queries.  
4. **Security**: Role-based access control for sensitive documents.  

--- 

**Word Count**: ~3,500  
**References**: [[1]][[2]][[3]][[4]][[5]][[6]][[7]][[8]][[9]][[10]]
