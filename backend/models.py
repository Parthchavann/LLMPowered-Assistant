from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question", min_length=1)
    top_k: int = Field(default=5, description="Number of similar documents to retrieve", ge=1, le=20)
    score_threshold: float = Field(default=0.3, description="Minimum similarity score", ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, description="Maximum tokens in response", ge=100, le=4096)
    temperature: float = Field(default=0.7, description="Generation temperature", ge=0.0, le=2.0)

class Source(BaseModel):
    source: str = Field(..., description="Source file name")
    chunk_index: int = Field(..., description="Chunk index in the source")
    score: float = Field(..., description="Similarity score")
    preview: str = Field(..., description="Preview of the content")

class RetrievalStats(BaseModel):
    documents_retrieved: int = Field(..., description="Number of documents retrieved")
    avg_score: float = Field(..., description="Average similarity score")
    score_threshold: float = Field(..., description="Score threshold used")

class GenerationStats(BaseModel):
    model: str = Field(..., description="Model used for generation")
    total_duration_ms: int = Field(..., description="Total generation time in milliseconds")
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(..., description="Sources used for the answer")
    context_used: bool = Field(..., description="Whether context was used")
    retrieval_stats: RetrievalStats = Field(..., description="Retrieval statistics")
    generation_stats: Optional[GenerationStats] = Field(None, description="Generation statistics")
    error: Optional[str] = Field(None, description="Error message if any")

class DocumentUploadRequest(BaseModel):
    overwrite: bool = Field(default=False, description="Whether to overwrite existing documents")
    chunk_size: int = Field(default=500, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=50, description="Chunk overlap for text splitting")

class DocumentUploadResponse(BaseModel):
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    services: Dict[str, bool] = Field(..., description="Status of individual services")
    vector_store_info: Dict[str, Any] = Field(..., description="Vector store information")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")

class FeedbackRequest(BaseModel):
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    rating: int = Field(..., description="User rating (1-5)", ge=1, le=5)
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    session_id: Optional[str] = Field(None, description="Optional session ID")

class FeedbackResponse(BaseModel):
    success: bool = Field(..., description="Whether feedback was saved")
    message: str = Field(..., description="Status message")
    feedback_id: Optional[str] = Field(None, description="Feedback ID if saved")

class MetricsResponse(BaseModel):
    total_queries: int = Field(..., description="Total number of queries")
    avg_response_time_ms: float = Field(..., description="Average response time in milliseconds")
    avg_rating: Optional[float] = Field(None, description="Average user rating")
    most_common_queries: List[Dict[str, Any]] = Field(..., description="Most common queries")
    error_rate: float = Field(..., description="Error rate percentage")