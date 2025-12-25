# InsightForge — AI-Powered BI Assistant (Enterprise Version)

## Features
- KPI computation (monthly/quarterly trends, top products, top regions, segmentation)
- RAG over computed KPI tables (FAISS + embeddings)
- Conversation memory (persistent JSON)
- Streamlit dashboards + chat UI
- Evaluation tab using QAEvalChain

## Run locally (Groq)
```bash
pip install -r requirements.txt
export GROQ_API_KEY="YOUR_KEY"
export LLM_PROVIDER="groq"
streamlit run app.py
```

## Run locally (Ollama)
1) Install Ollama
2) `ollama pull llama3`
```bash
export LLM_PROVIDER="ollama"
export OLLAMA_MODEL="llama3"
streamlit run app.py
```

## Streamlit Cloud
- Put this repo on GitHub.
- In Streamlit Cloud → Deploy → select repo → main file `app.py`
- Set **Secrets**:
  - `GROQ_API_KEY="..."`
  - `LLM_PROVIDER="groq"`
