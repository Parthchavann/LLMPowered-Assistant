# LLM-Powered Customer Support Assistant

A production-ready RAG (Retrieval-Augmented Generation) system for customer support using free, open-source tools.

## Features
- Document ingestion (PDF, TXT, MD)
- Semantic search using sentence transformers
- LLM-powered response generation
- Real-time chat interface
- Token usage monitoring
- Response feedback system
- Docker deployment ready

## Tech Stack
- **LLM**: Ollama (LLaMA 3.2 or Mistral)
- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- **Vector DB**: Qdrant (self-hosted)
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Infrastructure**: Docker & Docker Compose

## Quick Start

### Prerequisites
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull LLaMA 3.2 (3B model - lightweight)
ollama pull llama3.2:3b

# Clone repository
git clone <repo-url>
cd customer-support-rag

# Install dependencies
pip install -r requirements.txt
```

### Running Locally
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (new terminal)
cd frontend
streamlit run app.py
```

### Using Docker Compose
```bash
docker-compose up --build
```

## Project Structure
```
customer-support-rag/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models.py             # Pydantic models
│   ├── rag_engine.py         # RAG implementation
│   └── vector_store.py       # Qdrant integration
├── frontend/
│   └── app.py                # Streamlit UI
├── data/
│   ├── documents/            # Source documents
│   └── embeddings/           # Cached embeddings
├── monitoring/
│   └── metrics.py            # Usage tracking
├── utils/
│   ├── document_loader.py    # Document processing
│   └── embeddings.py         # Embedding generation
└── docker-compose.yml
```

## API Documentation
Once running, visit:
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8501
- Qdrant UI: http://localhost:6333/dashboard

## Environment Variables
```
OLLAMA_HOST=http://localhost:11434
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=support_docs
MODEL_NAME=llama3.2:3b
EMBEDDING_MODEL=all-MiniLM-L6-v2
```