# **Detailed Design Document for `simple_RAG-v8.py`**

## **Introduction**

The Python script `simple_RAG-v8.py` implements an advanced Retrieval-Augmented Generation (RAG) system designed to process multiple document formats, extract text, split it into manageable chunks, generate embeddings, and store them in a persistent vector database. The system supports both OpenAI and Ollama as language model providers and integrates with Gradio for a user-friendly interface. It also includes features like exporting processed text and handling various document types, including PDFs, DOCX, PPTX, EPUB, images, and more.

This document provides a detailed breakdown of the design and implementation of the `simple_RAG-v8.py` application. It covers key components such as document conversion, text splitting, embedding generation, vector database usage, query processing, and more. Additionally, it addresses specific requirements from the prompt, including handling proprietary Microsoft formats, persistent data storage, and potential enhancements.

---

## **System Architecture**

### **High-Level Diagram**
Below is a high-level diagram of the system architecture:

```
[User Query] --> [Query Processing] --> [Vector Database Search] --> [Response Generation]
       ^                                     |
       |                                     v
[Document Conversion & Loading] --> [Text Splitting] --> [Embedding Generation]
```

### **Description of Components**
1. **Document Conversion & Loading**: Reads documents of various formats and extracts raw text.
2. **Text Splitting**: Divides the extracted text into smaller chunks for efficient processing.
3. **Embedding Generation**: Converts text chunks into numerical representations (embeddings).
4. **Vector Database**: Stores embeddings for fast similarity search during query processing.
5. **Query Processing and Retrieval**: Processes user queries, retrieves relevant chunks from the vector database, and generates responses.

---

## **Detailed Component Breakdown**

### **3.1 Document Conversion & Loading**

#### **Libraries Used**
- **PyMuPDF (`fitz`)**: For extracting text from PDF files.
- **Aspose.Words (`aw`)**: For extracting text from DOCX files.
- **`pptx`**: For extracting text from PPTX files.
- **`pandas`**: For extracting text from Excel files.
- **`ebooklib` and `BeautifulSoup`**: For extracting text from EPUB files.
- **`pytesseract` and `PIL`**: For extracting text from images using OCR.

#### **Implementation**
The script uses a `convert_to_text` function to handle different file types. This function maps MIME types to specific converters:

```python
def convert_to_text(input_file: str) -> Optional[str]:
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
```

#### **Advantages**
- Supports multiple document formats, including proprietary Microsoft formats.
- Handles images using OCR for scanned documents.

---

### **3.2 Text Splitting**

#### **Library Used**
- **RecursiveCharacterTextSplitter**: A LangChain utility for splitting text into smaller chunks.

#### **Implementation**
The script splits the extracted text into chunks using the following parameters:
- `chunk_size`: 500 characters per chunk.
- `chunk_overlap`: 100 characters between consecutive chunks.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents([doc])
```

#### **Rationale**
Smaller chunks improve the efficiency of embedding generation and similarity search. Overlapping ensures that context is preserved across chunks.

---

### **3.3 Embedding Generation**

#### **Library Used**
- **OllamaEmbeddings**: A LangChain wrapper for Ollama models.

#### **Model Choice**
The script uses the `deepseek-r1:1.5b` model by default but allows users to specify other models.

#### **Implementation**
```python
embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=ollama_model)
```

#### **Process**
Each text chunk is converted into a dense vector representation (embedding) using the chosen model.

---

### **3.4 Vector Database**

#### **Library Used**
- **Chroma**: A persistent vector database for storing embeddings.

#### **Implementation**
The script creates a Chroma database to store embeddings persistently.

```python
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

#### **Advantages**
- Persistent storage ensures data is retained across sessions.
- Efficient similarity search for large datasets.

---

### **3.5 Query Processing and Retrieval**

#### **Steps Involved**
1. **Query Embedding**: Convert the user query into an embedding.
2. **Similarity Search**: Retrieve the most relevant text chunks from the vector database.
3. **Response Generation**: Use a language model (OpenAI or Ollama) to generate a coherent response based on the retrieved chunks.

#### **Code Snippet**
```python
retrieved_docs = SESSION_STATE["retriever"].invoke(question)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

full_prompt = f"System: {system_prompt}\nQuestion: {question}\nContext:\n{context}"

if provider == "Ollama":
    endpoint = "/chat/completions"
    url = f"{ollama_base_url}/v1{endpoint}"
    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, json=payload)
    result = response.json()
    return result['choices'][0]['message']['content']
```

---

## **Document Format Handling (Proprietary Microsoft Formats)**

The script supports proprietary Microsoft formats like `.docx`, `.pptx`, and `.xlsx` using libraries like `aspose.words` and `pptx`. It also handles images using OCR via `pytesseract`.

---

## **Data Persistence**

The script uses Chroma as a persistent vector database. Data is stored in the `./chroma_db` directory, ensuring persistence across sessions.

---

## **Enhancements**

Based on the analysis of the code, here are some enhancement ideas:
1. **Advanced Text Splitting**: Experiment with semantic splitting techniques.
2. **Different Embedding Models**: Explore models like `bert-base-nli-mean-tokens` or `paraphrase-MiniLM-L6-v2`.
3. **Scalability**: Optimize for larger datasets by integrating distributed vector databases like Pinecone or Milvus.
4. **Error Handling**: Improve robustness with comprehensive error handling and logging mechanisms.
5. **User Interface**: Enhance the Gradio interface with additional features like progress indicators and batch processing.

---

## **Conclusion**

The `simple_RAG-v8.py` script provides a robust foundation for building a RAG system. With support for multiple document formats, persistent storage, and integration with both OpenAI and Ollama, it is a versatile tool for document-based question answering. By addressing limitations and incorporating best practices, this system can evolve into a scalable and production-ready solution.

---

[**Analysis by Qwen**: read the report here.](https://chat.qwenlm.ai/s/aa7b8101-4ed9-4d88-ab3c-2f2ea10c4a4e)
