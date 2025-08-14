import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import threading
from dataclasses import dataclass, asdict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryMetric:
    """Data class for query metrics"""
    id: str
    timestamp: datetime
    query: str
    response_time_ms: float
    success: bool
    documents_retrieved: int = 0
    avg_similarity_score: float = 0.0
    model_used: str = ""
    session_id: Optional[str] = None

@dataclass
class FeedbackMetric:
    """Data class for feedback metrics"""
    id: str
    timestamp: datetime
    query: str
    answer: str
    rating: int
    feedback_text: Optional[str] = None
    session_id: Optional[str] = None

class MetricsCollector:
    """Centralized metrics collection and analysis"""
    
    def __init__(self, storage_path: str = "monitoring/metrics_data"):
        self.storage_path = storage_path
        self.queries: List[QueryMetric] = []
        self.feedback: List[FeedbackMetric] = []
        self.lock = threading.Lock()
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from storage"""
        try:
            queries_file = os.path.join(self.storage_path, "queries.json")
            feedback_file = os.path.join(self.storage_path, "feedback.json")
            
            # Load queries
            if os.path.exists(queries_file):
                with open(queries_file, 'r') as f:
                    queries_data = json.load(f)
                    self.queries = [
                        QueryMetric(
                            id=q['id'],
                            timestamp=datetime.fromisoformat(q['timestamp']),
                            query=q['query'],
                            response_time_ms=q['response_time_ms'],
                            success=q['success'],
                            documents_retrieved=q.get('documents_retrieved', 0),
                            avg_similarity_score=q.get('avg_similarity_score', 0.0),
                            model_used=q.get('model_used', ''),
                            session_id=q.get('session_id')
                        )
                        for q in queries_data
                    ]
            
            # Load feedback
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                    self.feedback = [
                        FeedbackMetric(
                            id=f['id'],
                            timestamp=datetime.fromisoformat(f['timestamp']),
                            query=f['query'],
                            answer=f['answer'],
                            rating=f['rating'],
                            feedback_text=f.get('feedback_text'),
                            session_id=f.get('session_id')
                        )
                        for f in feedback_data
                    ]
            
            logger.info(f"Loaded {len(self.queries)} query metrics and {len(self.feedback)} feedback entries")
            
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_metrics(self):
        """Save metrics to storage"""
        try:
            queries_file = os.path.join(self.storage_path, "queries.json")
            feedback_file = os.path.join(self.storage_path, "feedback.json")
            
            # Save queries
            queries_data = []
            for q in self.queries:
                query_dict = asdict(q)
                query_dict['timestamp'] = q.timestamp.isoformat()
                queries_data.append(query_dict)
            
            with open(queries_file, 'w') as f:
                json.dump(queries_data, f, indent=2)
            
            # Save feedback
            feedback_data = []
            for f in self.feedback:
                feedback_dict = asdict(f)
                feedback_dict['timestamp'] = f.timestamp.isoformat()
                feedback_data.append(feedback_dict)
            
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def record_query(
        self,
        query: str,
        response_time_ms: float,
        success: bool,
        documents_retrieved: int = 0,
        avg_similarity_score: float = 0.0,
        model_used: str = "",
        session_id: Optional[str] = None
    ) -> str:
        """Record a query metric"""
        
        metric = QueryMetric(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            query=query,
            response_time_ms=response_time_ms,
            success=success,
            documents_retrieved=documents_retrieved,
            avg_similarity_score=avg_similarity_score,
            model_used=model_used,
            session_id=session_id
        )
        
        with self.lock:
            self.queries.append(metric)
            self._save_metrics()
        
        logger.info(f"Recorded query metric: {metric.id}")
        return metric.id
    
    def record_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        feedback_text: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Record user feedback"""
        
        feedback = FeedbackMetric(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            query=query,
            answer=answer,
            rating=rating,
            feedback_text=feedback_text,
            session_id=session_id
        )
        
        with self.lock:
            self.feedback.append(feedback)
            self._save_metrics()
        
        logger.info(f"Recorded feedback: {feedback.id}, Rating: {rating}")
        return feedback.id
    
    def get_metrics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with self.lock:
            # Filter recent queries
            recent_queries = [q for q in self.queries if q.timestamp >= cutoff_date]
            recent_feedback = [f for f in self.feedback if f.timestamp >= cutoff_date]
        
        # Basic stats
        total_queries = len(recent_queries)
        successful_queries = [q for q in recent_queries if q.success]
        failed_queries = [q for q in recent_queries if not q.success]
        
        # Response time stats
        response_times = [q.response_time_ms for q in successful_queries]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Error rate
        error_rate = (len(failed_queries) / total_queries * 100) if total_queries > 0 else 0
        
        # Feedback stats
        ratings = [f.rating for f in recent_feedback]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Most common queries
        query_counter = Counter([q.query.lower() for q in recent_queries])
        most_common_queries = [
            {"query": query, "count": count}
            for query, count in query_counter.most_common(10)
        ]
        
        # Usage over time (daily)
        daily_usage = defaultdict(int)
        for q in recent_queries:
            date_key = q.timestamp.date().isoformat()
            daily_usage[date_key] += 1
        
        return {
            "total_queries": total_queries,
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "error_rate": round(error_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "total_feedback": len(recent_feedback),
            "most_common_queries": most_common_queries,
            "daily_usage": dict(daily_usage),
            "rating_distribution": dict(Counter(ratings)) if ratings else {}
        }
    
    def get_detailed_analytics(self) -> Dict[str, Any]:
        """Get detailed analytics for monitoring dashboard"""
        
        with self.lock:
            all_queries = self.queries.copy()
            all_feedback = self.feedback.copy()
        
        # Performance metrics
        response_times = [q.response_time_ms for q in all_queries if q.success]
        
        performance_stats = {
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "p50_response_time": sorted(response_times)[len(response_times)//2] if response_times else 0,
            "p95_response_time": sorted(response_times)[int(len(response_times)*0.95)] if response_times else 0,
        }
        
        # Retrieval stats
        docs_retrieved = [q.documents_retrieved for q in all_queries if q.success]
        similarity_scores = [q.avg_similarity_score for q in all_queries if q.success and q.avg_similarity_score > 0]
        
        retrieval_stats = {
            "avg_docs_retrieved": sum(docs_retrieved) / len(docs_retrieved) if docs_retrieved else 0,
            "avg_similarity_score": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
        }
        
        # User satisfaction
        satisfaction_stats = {
            "total_ratings": len(all_feedback),
            "satisfaction_rate": len([f for f in all_feedback if f.rating >= 4]) / len(all_feedback) * 100 if all_feedback else 0,
            "dissatisfaction_rate": len([f for f in all_feedback if f.rating <= 2]) / len(all_feedback) * 100 if all_feedback else 0,
        }
        
        # Recent trends (last 7 days vs previous 7 days)
        now = datetime.now()
        last_7_days = [q for q in all_queries if (now - q.timestamp).days <= 7]
        prev_7_days = [q for q in all_queries if 7 < (now - q.timestamp).days <= 14]
        
        trends = {
            "queries_growth": len(last_7_days) - len(prev_7_days),
            "response_time_trend": (
                sum(q.response_time_ms for q in last_7_days if q.success) / len([q for q in last_7_days if q.success]) -
                sum(q.response_time_ms for q in prev_7_days if q.success) / len([q for q in prev_7_days if q.success])
            ) if last_7_days and prev_7_days else 0,
        }
        
        return {
            "performance_stats": performance_stats,
            "retrieval_stats": retrieval_stats,
            "satisfaction_stats": satisfaction_stats,
            "trends": trends,
            "total_sessions": len(set(q.session_id for q in all_queries if q.session_id)),
            "data_points": {
                "total_queries": len(all_queries),
                "total_feedback": len(all_feedback),
                "date_range": {
                    "earliest": min(q.timestamp for q in all_queries).isoformat() if all_queries else None,
                    "latest": max(q.timestamp for q in all_queries).isoformat() if all_queries else None
                }
            }
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in various formats"""
        
        with self.lock:
            data = {
                "queries": [asdict(q) for q in self.queries],
                "feedback": [asdict(f) for f in self.feedback],
                "export_timestamp": datetime.now().isoformat()
            }
        
        # Convert datetime objects to strings
        for query in data["queries"]:
            query["timestamp"] = query["timestamp"].isoformat() if hasattr(query["timestamp"], 'isoformat') else str(query["timestamp"])
        
        for feedback in data["feedback"]:
            feedback["timestamp"] = feedback["timestamp"].isoformat() if hasattr(feedback["timestamp"], 'isoformat') else str(feedback["timestamp"])
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_old_metrics(self, days_to_keep: int = 90):
        """Clear metrics older than specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.lock:
            old_queries = len(self.queries)
            old_feedback = len(self.feedback)
            
            self.queries = [q for q in self.queries if q.timestamp >= cutoff_date]
            self.feedback = [f for f in self.feedback if f.timestamp >= cutoff_date]
            
            self._save_metrics()
        
        logger.info(f"Cleaned up metrics: {old_queries - len(self.queries)} queries, {old_feedback - len(self.feedback)} feedback entries removed")

# Prometheus metrics (if prometheus_client is available)
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    
    # Define Prometheus metrics
    QUERIES_TOTAL = Counter('rag_queries_total', 'Total number of queries', ['status'])
    QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Query processing time')
    DOCUMENTS_RETRIEVED = Histogram('rag_documents_retrieved', 'Number of documents retrieved per query')
    SIMILARITY_SCORE = Histogram('rag_similarity_score', 'Average similarity score per query')
    USER_RATING = Histogram('rag_user_rating', 'User satisfaction rating')
    ACTIVE_SESSIONS = Gauge('rag_active_sessions', 'Number of active sessions')
    
    class PrometheusMetricsCollector(MetricsCollector):
        """Extended metrics collector with Prometheus support"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prometheus_port = kwargs.get('prometheus_port', 8001)
            
            # Start Prometheus metrics server
            try:
                start_http_server(self.prometheus_port)
                logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {str(e)}")
        
        def record_query(self, query: str, response_time_ms: float, success: bool, **kwargs):
            """Record query with Prometheus metrics"""
            
            # Record in parent class
            query_id = super().record_query(query, response_time_ms, success, **kwargs)
            
            # Record Prometheus metrics
            status = 'success' if success else 'error'
            QUERIES_TOTAL.labels(status=status).inc()
            QUERY_DURATION.observe(response_time_ms / 1000)  # Convert to seconds
            
            if success and kwargs.get('documents_retrieved', 0) > 0:
                DOCUMENTS_RETRIEVED.observe(kwargs['documents_retrieved'])
            
            if success and kwargs.get('avg_similarity_score', 0) > 0:
                SIMILARITY_SCORE.observe(kwargs['avg_similarity_score'])
            
            return query_id
        
        def record_feedback(self, query: str, answer: str, rating: int, **kwargs):
            """Record feedback with Prometheus metrics"""
            
            # Record in parent class
            feedback_id = super().record_feedback(query, answer, rating, **kwargs)
            
            # Record Prometheus metrics
            USER_RATING.observe(rating)
            
            return feedback_id

except ImportError:
    logger.info("prometheus_client not available, using basic metrics collector")
    PrometheusMetricsCollector = MetricsCollector