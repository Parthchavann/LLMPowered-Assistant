import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.models import (
    QueryRequest, QueryResponse, DocumentUploadRequest, DocumentUploadResponse,
    HealthResponse, FeedbackRequest, FeedbackResponse, MetricsResponse
)
from backend.rag_engine import RAGEngine
from backend.vector_store import QdrantVectorStore
from utils.document_loader import DocumentLoader
from utils.embeddings import EmbeddingGenerator
from monitoring.metrics import MetricsCollector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for components
rag_engine = None
vector_store = None
embedding_generator = None
metrics_collector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global rag_engine, vector_store, embedding_generator, metrics_collector
    
    # Startup
    logger.info("Starting up Customer Support RAG API...")
    
    try:
        # Initialize components
        logger.info("Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        
        logger.info("Initializing vector store...")
        vector_store = QdrantVectorStore(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("COLLECTION_NAME", "support_docs"),
            vector_dimension=embedding_generator.get_dimension()
        )
        
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            model_name=os.getenv("MODEL_NAME", "llama3.2:3b"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        
        logger.info("Initializing metrics collector...")
        metrics_collector = MetricsCollector()
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Customer Support RAG API",
    description="A production-ready RAG system for customer support using LLM and vector search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Support RAG API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store
        vector_store_healthy = vector_store.health_check() if vector_store else False
        vector_store_info = vector_store.get_collection_info() if vector_store else {}
        
        # Check model
        model_info = rag_engine.get_model_info() if rag_engine else {}
        
        services = {
            "vector_store": vector_store_healthy,
            "embedding_model": embedding_generator is not None,
            "llm_model": rag_engine is not None and "error" not in model_info,
            "metrics": metrics_collector is not None
        }
        
        overall_status = "healthy" if all(services.values()) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            services=services,
            vector_store_info=vector_store_info,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    if not rag_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    try:
        start_time = time.time()
        
        # Process query
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Record metrics
        response_time_ms = (time.time() - start_time) * 1000
        if metrics_collector:
            metrics_collector.record_query(
                query=request.question,
                response_time_ms=response_time_ms,
                success="error" not in result
            )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        if metrics_collector:
            metrics_collector.record_query(
                query=request.question,
                response_time_ms=0,
                success=False
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    overwrite: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """Upload and process documents"""
    if not all([vector_store, embedding_generator]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required services not initialized"
        )
    
    try:
        start_time = time.time()
        
        # Save uploaded files temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        saved_files = []
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(str(file_path))
        
        # Process documents
        document_loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = []
        
        for file_path in saved_files:
            if file_path.endswith('.pdf'):
                docs = document_loader.load_pdf(file_path)
            else:
                docs = document_loader.load_text_file(file_path)
            documents.extend(docs)
        
        # Split into chunks
        chunks = document_loader.split_documents(documents)
        
        # Generate embeddings
        embedded_chunks = embedding_generator.embed_documents(chunks)
        
        # Store in vector database
        if overwrite:
            vector_store.delete_collection()
            vector_store._create_collection_if_not_exists()
        
        success = vector_store.add_documents(embedded_chunks)
        
        # Cleanup temp files
        for file_path in saved_files:
            os.remove(file_path)
        temp_dir.rmdir()
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            success=success,
            message=f"Processed {len(files)} files successfully" if success else "Failed to process documents",
            documents_processed=len(documents),
            chunks_created=len(chunks),
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        # Cleanup on error
        try:
            for file_path in saved_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            if temp_dir.exists():
                temp_dir.rmdir()
        except:
            pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
        )

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback"""
    if not metrics_collector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics collector not initialized"
        )
    
    try:
        feedback_id = metrics_collector.record_feedback(
            query=request.query,
            answer=request.answer,
            rating=request.rating,
            feedback_text=request.feedback_text,
            session_id=request.session_id
        )
        
        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}"
        )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics"""
    if not metrics_collector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics collector not initialized"
        )
    
    try:
        metrics = metrics_collector.get_metrics()
        return MetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )

@app.get("/collection/info")
async def get_collection_info():
    """Get vector collection information"""
    if not vector_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized"
        )
    
    return vector_store.get_collection_info()

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )