\<think\>
Let me analyze the changes needed:

1. Config File Updates:
- Need to add "enhancements" section for content moderation and reranker models
- Keep existing user configuration values
- Add new fields while preserving user customizations

2. New Models Required:
- Content moderation model: "unitary/toxic-bert"
- Reranker model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
- These require the transformers and sentence-transformers packages

3. Installation Steps:
- Need to install new Python packages
- Need to download the models</think>

Here's the updated configuration file and installation instructions:
Adding the enhancements section while preserving existing settings:

```
$ cat rag_data/config.yaml 
generation:
  max_tokens: 8192
  temperature: 1.0
ollama:
  base_url: http://localhost:11434
  embedding_model: nomic-embed-text
  model: deepseek-r1:latest
openai:
  api_key: sk-proj-IL_xxxxA
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
rag:
  chunk_overlap: 50
  chunk_size: 500
  k_retrieval: 5
  system_prompt: You are a helpful research assistant that provides accurate information
    based on the given context. If the answer is not in the context, say you don't
    know.
ui:
  default_provider: OpenAI
  max_history: 10
  theme: default
enhancements:
  content_moderation_model: unitary/toxic-bert
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

```

Installation Instructions
To install the required new packages and models, run these commands in your virtual environment:
```
# Install required packages
pip install transformers sentence-transformers

# Optional: Pre-download the models to avoid first-run delays
python -c "
from transformers import pipeline
from sentence_transformers import CrossEncoder

# Download content moderation model
_ = pipeline('text-classification', model='unitary/toxic-bert')

# Download reranker model
_ = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
"
```

The new models will be automatically downloaded when first used if you don't pre-download them. They require:

- unitary/toxic-bert: ~500MB
- cross-encoder/ms-marco-MiniLM-L-6-v2: ~200MB

These models will be cached locally after the first download. The content moderation model helps ensure appropriate queries, while the reranker model improves the relevance of retrieved document chunks.

Note: The existing configuration values from your config.yaml (like OpenAI key, temperature settings, etc.) have been preserved while adding the new enhancement section.
