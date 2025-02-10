# project details can be found in this link: https://bit.ly/4hpF3du — You may need to sign up (for free!) to DataLab

# https://youtu.be/hOsZzcMYMLI

how to set up DeepSeek R1 on your local machine to securely query PDF documents using retrieval-augmented generation (RAG). We walk through every step, from downloading and configuring the model with Ollama to building a Gradio-based web app that processes PDF files using LangChain and vector databases.
Whether you’re on a Mac or Windows, this video covers data preprocessing, text embedding, and semantic search, giving you a comprehensive understanding of local AI-assisted document queries without any reliance on the cloud.

(app) H:\app>python app_simple_RAG.py
H:\app\app_simple_RAG.py:13: LangChainDeprecationWarning: Importing Chroma from langchain.vectorstores is deprecated. Please replace deprecated imports:

>> from langchain.vectorstores import Chroma

with new imports of:

>> from langchain_community.vectorstores import Chroma
You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/docs/versions/v0_2/>
  from langchain.vectorstores import Chroma
* Running on local URL:  http://127.0.0.1:7860
INFO:httpx:HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET http://127.0.0.1:7860/gradio_api/startup-events "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD http://127.0.0.1:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
WARNING:python_multipart.multipart:Skipping data after last boundary
H:\app\app_simple_RAG.py:27: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
INFO:chromadb.telemetry.product.posthog:Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
INFO:root:PDF processed: 26 chunks generated
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"

![image](https://github.com/user-attachments/assets/2082e8fb-7bb8-4b41-a9b4-40b0307cace4)

![image](https://github.com/user-attachments/assets/3de49bea-f387-477b-983a-2660dff8ca05)


