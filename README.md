*The application implements a Retrieval-Augmented Generation (RAG) pipeline that processes various types of documents, extracts relevant information, and answers user queries using a Large Language Model (LLM)*

```bash
pip install langchain langchain-community langchain-ollama ollama gradio openai python-dotx python-pptx pytesseract
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers rank-bm25
python simple_RAG-v14.py
```
![image](https://github.com/user-attachments/assets/1fd5fa5e-cd3b-44b6-bc99-85d7ff89e1cf)

```
2025-03-05 07:22:33,758 - RAG_Assistant - INFO - Loaded configuration from rag_data/config.yaml
2025-03-05 07:22:33,982 - RAG_Assistant - INFO - Starting Enhanced RAG Assistant v2.0.0
2025-03-05 07:22:33,982 - RAG_Assistant - INFO - Host: 0.0.0.0, Port: 7860, Debug: False
* Running on local URL:  http://0.0.0.0:7860
2025-03-05 07:22:34,152 - httpx - INFO - HTTP Request: GET http://localhost:7860/gradio_api/startup-events "HTTP/1.1 200 OK"
2025-03-05 07:22:34,176 - httpx - INFO - HTTP Request: HEAD http://localhost:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
2025-03-05 07:22:35,180 - httpx - INFO - HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
2025-03-05 07:24:19,124 - python_multipart.multipart - WARNING - Skipping data after last boundary
2025-03-05 07:24:21,407 - RAG_Assistant - INFO - Status: Processing documents...
2025-03-05 07:24:22,213 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2025-03-05 07:24:22,214 - RAG_Assistant - INFO - Detected embedding dimension: 768
Processing files:   0%|                                                                                                                                                                                               | 0/1 [00:00<?, ?it/s]2025-03-05 07:24:22,217 - RAG_Assistant - INFO - Status: Converting English Course - Basic Grammar.pdf to text...
2025-03-05 07:24:22,225 - RAG_Assistant - INFO - Successfully converted rag_data/sources/English Course - Basic Grammar.pdf
Processing files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 90.03it/s]
2025-03-05 07:24:22,226 - RAG_Assistant - INFO - Status: Building vector store with 11 chunks...
2025-03-05 07:24:22,911 - chromadb.telemetry.product.posthog - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-03-05 07:25:24,507 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2025-03-05 07:25:24,887 - RAG_Assistant - INFO - Status: Vector store updated with 11 chunks from 1 files
Device set to use cpu
2025-03-05 07:26:49,638 - RAG_Assistant - INFO - Status: Retrieving context for: use all the content you have, create a complete nicely formatted report that is detailed and formatted using WhatsApp style of formatting
2025-03-05 07:26:49,647 - RAG_Assistant - WARNING - Falling back to standard retrieval due to error: cannot import name 'EnsembleRetriever' from 'langchain_community.retrievers' (/cdrom/venv/mychat/lib/python3.12/site-packages/langchain_community/retrievers/__init__.py)
2025-03-05 07:26:50,873 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2025-03-05 07:26:50,880 - RAG_Assistant - INFO - Status: Generating response using OpenAI...
2025-03-05 07:26:55,484 - httpx - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-03-05 07:26:55,493 - RAG_Assistant - INFO - Status: Ready
```

---
```bash
python simple_RAG-v9.py
```

![image](https://github.com/user-attachments/assets/4db5321b-aa4a-4ef4-b701-a9eefce808ca)

---
```bash
pip install langchain_community ollama gradio openai python-dotx python-pptx pytesseract
python simple_RAG_v8.py
```
![image](https://github.com/user-attachments/assets/20a68e79-a853-4af2-983e-6af7148ef164)

![image](https://github.com/user-attachments/assets/3de49bea-f387-477b-983a-2660dff8ca05)

![image](https://github.com/user-attachments/assets/911b45a7-d8e3-460b-b131-ce61d55745d7)

project details can be found in this link: https://bit.ly/4hpF3du — You may need to sign up (for free!) to DataLab

https://youtu.be/hOsZzcMYMLI

https://www.perplexity.ai/search/think-of-yourself-as-a-great-s-312se3iLTAi9SVZNerBC.g
