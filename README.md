# Chatbot Project

A lightweight question-answering and PDF information retrieval demo combining:

- FastAPI backend for semantic question matching over a Q&A CSV dataset.
- Streamlit frontend UI for interactive querying.
- Embedding generation + cosine similarity using SentenceTransformers.
- Notebooks for (1) building embeddings for the FAQ dataset and (2) PDF chunking + retrieval + Gemini summarization.

---
## Quick Start

### 1. Clone / Open
Place the project folder locally (already done if you're reading this here).

### 2. Create & Activate Virtual Environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```
(If a package is missing at runtime, install it manually, e.g. `pip install sentence-transformers faiss-cpu google-generativeai streamlit fastapi uvicorn`.)

### 4. Prepare Data & Embeddings
Ensure the dataset CSV exists at `./data/news_CSV.csv`.

Generate and save question embeddings (creates `./embeddings/question_embeddings.npy`):
- Option A (Notebook): Open `chat.ipynb` and run all cells.
- Option B (Inline Python):
  ```python
  from sentence_transformers import SentenceTransformer
  import pandas as pd, numpy as np
  df = pd.read_csv('./data/news_CSV.csv', encoding='latin1')
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embs = model.encode(df['context'].tolist(), show_progress_bar=True)
  import os; os.makedirs('embeddings', exist_ok=True)
  np.save('./embeddings/question_embeddings.npy', embs)
  ```

### 5. Run the API Server
```powershell
python api.py
```
The FastAPI app serves at: http://127.0.0.1:8000

Docs (interactive): http://127.0.0.1:8000/docs

### 6. Run the Streamlit UI (in a second terminal / same env)
```powershell
streamlit run streamlit_app.py
```
Open the displayed local URL (usually http://localhost:8501).

### 7. Ask Questions
Use the UI or call the API directly:
```powershell
# Example with PowerShell Invoke-RestMethod
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -Body (@{question='Can I drink alcohol after covid vaccine?'} | ConvertTo-Json) -ContentType 'application/json'
```

---
## File / Component Overview

| File | Purpose |
|------|---------|
| `api.py` | FastAPI backend. Loads model + precomputed question embeddings, exposes endpoints: `/questions` (top sample questions) and `/ask` (semantic match + answer). |
| `streamlit_app.py` | Streamlit frontend UI. Lets users click sample questions or type their own, sends requests to the FastAPI server, displays answer, similar question, and similarity score. |
| `chat.ipynb` | Notebook to: (1) load Q&A CSV, (2) compute sentence-transformer embeddings for `context` questions, (3) save them to `embeddings/question_embeddings.npy`, (4) demonstrate similarity lookup and answer retrieval logic. |
| `IR.ipynb` | Information Retrieval for PDFs: (1) chunk a PDF with sliding window, (2) embed chunks (SentenceTransformers), (3) build FAISS index, (4) query similar chunks, (5) optional Gemini API summarization focusing on a user question. Demonstrates retrieval-augmented summarization flow. |
| `requirements.txt` | Python dependencies list. Install with `pip install -r requirements.txt`. |
| `data/` | Input data directory (e.g., `news_CSV.csv` for FAQ pairs, plus PDFs you may process). |
| `embeddings/` | Stores precomputed embeddings: question embeddings, PDF chunk embeddings, and metadata pickles/NumPy arrays. |

---
## API Endpoints

### GET /questions
Returns first 10 questions for quick UI buttons.

### POST /ask
Request body:
```json
{ "question": "your question here" }
```
Response fields:
- `question`: original query
- `answer`: matched answer text (or fallback message if similarity < 0.70)
- `similar_question`: the closest question in dataset
- `similarity_score`: cosine similarity float (0–1)

Similarity threshold currently hard-coded at `0.70` in `api.py`.

---
## PDF Retrieval (IR.ipynb) Workflow
1. Set `pdf_path` to target PDF (file placed in `data/` or root).
2. Run cells to chunk & embed.
3. FAISS index is built in-memory (can be extended to persist).
4. Use `query_bot()` to retrieve top-k relevant chunks.
5. Summarization (Gemini) cell builds a custom prompt referencing the original question.

### Gemini API Key
Do NOT hardcode keys in notebooks. Instead set environment variable before launching Jupyter/VSCode shell:
```powershell
$Env:GEMINI_API_KEY = 'your_real_key_here'
```
Then in the notebook:
```python
import os
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
```
Remove/replace any placeholder key present in saved notebooks before sharing.

---
## Customization Tips
- Adjust similarity threshold in `api.py` to tune recall vs precision.
- Add caching or a vector DB (e.g., FAISS persist, Chroma, Pinecone) for scale.
- Expand `IR.ipynb` to multi-PDF ingestion (concatenate chunk lists; store doc IDs).
- Add rate limiting / auth for production usage.
- Stream responses in Streamlit for longer generation tasks (currently simple POST).

---
## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| ImportError: sentence_transformers | Package missing | `pip install sentence-transformers` |
| FileNotFoundError: question_embeddings.npy | Embeddings not generated | Run `chat.ipynb` or the embedding script snippet above |
| 422 Unprocessable Entity (POST /ask) | Bad JSON body | Ensure correct field name `question` |
| Slow first request | Model cold start | Keep process warm, or preload with a dummy encode |
| CORS error (if calling from browser) | No CORS middleware | Add FastAPI CORSMiddleware (not required for Streamlit same-origin) |

---
## Extending
- Wrap embedding generation into a CLI script (e.g., `python build_embeddings.py`).
- Add logging & metrics (uvicorn --log-level info, `logging` module).
- Switch to `faiss.IndexIDMap` for better tracking of chunk metadata in PDF retrieval.
- Introduce reranking (e.g., cross-encoder) for improved answer quality.

---
## License / Use
Internal demo / educational template. Add a formal LICENSE file if distributing.

---
## At a Glance
```text
Question -> Embedding -> Cosine Similarity -> Retrieve Best Match -> Answer
(PDF mode) Query -> Embed -> FAISS Search -> Relevant Chunks -> (LLM Summarize)
```

Happy building! 🚀
