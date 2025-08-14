#!/usr/bin/env python3

"""
Customer Support RAG System Demo Script

This script demonstrates the full capabilities of the RAG system including:
- Document ingestion and processing
- Vector database operations
- Query processing and response generation
- Performance metrics and monitoring

Run with: python scripts/demo.py
"""

import sys
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.document_loader import DocumentLoader
from utils.embeddings import EmbeddingGenerator
from backend.vector_store import QdrantVectorStore
from backend.rag_engine import RAGEngine
from monitoring.metrics import MetricsCollector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class RAGSystemDemo:
    def __init__(self):
        self.components = {}
        self.demo_queries = [
            "How do I reset my password?",
            "What payment methods do you accept?",
            "How do I create a new project?",
            "The application is running slowly, what should I do?",
            "How do I enable two-factor authentication?",
            "Can I get a refund for my subscription?",
            "What are the different user roles?",
            "How do I invite team members?",
            "How is my data protected?",
            "What are your support hours?"
        ]
        
    def print_header(self, text: str):
        """Print a formatted header"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
    
    def print_step(self, step: str, status: str = "RUNNING"):
        """Print a step with status"""
        status_colors = {
            "RUNNING": "\033[93m",  # Yellow
            "SUCCESS": "\033[92m",  # Green
            "ERROR": "\033[91m",    # Red
            "INFO": "\033[94m"      # Blue
        }
        reset_color = "\033[0m"
        
        color = status_colors.get(status, "")
        print(f"{color}[{status}]{reset_color} {step}")
    
    def initialize_components(self):
        """Initialize all RAG system components"""
        self.print_header("INITIALIZING RAG SYSTEM COMPONENTS")
        
        try:
            # Initialize embedding generator
            self.print_step("Loading embedding model...")
            self.components['embedding_generator'] = EmbeddingGenerator(
                model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            )
            self.print_step("Embedding model loaded", "SUCCESS")
            
            # Initialize vector store
            self.print_step("Connecting to vector database...")
            self.components['vector_store'] = QdrantVectorStore(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                collection_name=os.getenv("COLLECTION_NAME", "support_docs"),
                vector_dimension=self.components['embedding_generator'].get_dimension()
            )
            self.print_step("Vector database connected", "SUCCESS")
            
            # Initialize RAG engine
            self.print_step("Initializing RAG engine...")
            self.components['rag_engine'] = RAGEngine(
                vector_store=self.components['vector_store'],
                embedding_generator=self.components['embedding_generator'],
                model_name=os.getenv("MODEL_NAME", "llama3.2:3b"),
                ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )
            self.print_step("RAG engine initialized", "SUCCESS")
            
            # Initialize metrics collector
            self.print_step("Setting up metrics collection...")
            self.components['metrics'] = MetricsCollector(
                storage_path="monitoring/demo_metrics"
            )
            self.print_step("Metrics collector ready", "SUCCESS")
            
        except Exception as e:
            self.print_step(f"Component initialization failed: {str(e)}", "ERROR")
            raise
    
    def check_system_health(self):
        """Check the health of all system components"""
        self.print_header("SYSTEM HEALTH CHECK")
        
        health_status = {}
        
        # Check vector store
        try:
            vector_health = self.components['vector_store'].health_check()
            health_status['vector_store'] = vector_health
            status = "SUCCESS" if vector_health else "ERROR"
            self.print_step(f"Vector store health: {'Healthy' if vector_health else 'Unhealthy'}", status)
        except Exception as e:
            health_status['vector_store'] = False
            self.print_step(f"Vector store health check failed: {str(e)}", "ERROR")
        
        # Check collection info
        try:
            collection_info = self.components['vector_store'].get_collection_info()
            if collection_info:
                self.print_step(f"Collection '{collection_info['name']}' has {collection_info['points_count']} documents", "INFO")
                health_status['collection'] = True
            else:
                self.print_step("No collection information available", "ERROR")
                health_status['collection'] = False
        except Exception as e:
            health_status['collection'] = False
            self.print_step(f"Collection check failed: {str(e)}", "ERROR")
        
        # Check LLM model
        try:
            model_info = self.components['rag_engine'].get_model_info()
            if 'error' not in model_info:
                self.print_step(f"LLM model '{model_info['name']}' is available", "SUCCESS")
                health_status['llm'] = True
            else:
                self.print_step(f"LLM model error: {model_info['error']}", "ERROR")
                health_status['llm'] = False
        except Exception as e:
            health_status['llm'] = False
            self.print_step(f"LLM model check failed: {str(e)}", "ERROR")
        
        return health_status
    
    def process_sample_documents(self):
        """Process sample documents and add to vector store"""
        self.print_header("PROCESSING SAMPLE DOCUMENTS")
        
        documents_path = project_root / "data" / "documents"
        
        if not documents_path.exists() or not list(documents_path.glob("*")):
            self.print_step("No documents found in data/documents directory", "ERROR")
            return False
        
        try:
            # Load and process documents
            self.print_step("Loading documents...")
            doc_loader = DocumentLoader(
                chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
            )
            
            documents = doc_loader.process_documents(str(documents_path))
            self.print_step(f"Loaded and chunked {len(documents)} document segments", "SUCCESS")
            
            # Generate embeddings
            self.print_step("Generating embeddings...")
            embedded_docs = self.components['embedding_generator'].embed_documents(documents)
            self.print_step(f"Generated embeddings for {len(embedded_docs)} chunks", "SUCCESS")
            
            # Store in vector database
            self.print_step("Storing in vector database...")
            success = self.components['vector_store'].add_documents(embedded_docs)
            
            if success:
                self.print_step("Documents stored successfully", "SUCCESS")
                return True
            else:
                self.print_step("Failed to store documents", "ERROR")
                return False
                
        except Exception as e:
            self.print_step(f"Document processing failed: {str(e)}", "ERROR")
            return False
    
    def run_demo_queries(self):
        """Run a series of demo queries to showcase the system"""
        self.print_header("RUNNING DEMO QUERIES")
        
        results = []
        
        for i, query in enumerate(self.demo_queries, 1):
            self.print_step(f"Query {i}/{len(self.demo_queries)}: {query}")
            
            try:
                # Record start time
                start_time = time.time()
                
                # Process query
                result = self.components['rag_engine'].query(
                    question=query,
                    top_k=5,
                    score_threshold=0.3,
                    max_tokens=1024,
                    temperature=0.7
                )
                
                # Calculate response time
                response_time = (time.time() - start_time) * 1000
                
                # Record metrics
                self.components['metrics'].record_query(
                    query=query,
                    response_time_ms=response_time,
                    success='error' not in result,
                    documents_retrieved=result.get('retrieval_stats', {}).get('documents_retrieved', 0),
                    avg_similarity_score=result.get('retrieval_stats', {}).get('avg_score', 0.0)
                )
                
                # Store result
                result['query'] = query
                result['response_time_ms'] = response_time
                results.append(result)
                
                # Display result summary
                if 'error' not in result:
                    sources_count = len(result.get('sources', []))
                    self.print_step(
                        f"✓ Answer generated ({response_time:.0f}ms, {sources_count} sources)", 
                        "SUCCESS"
                    )
                    
                    # Show first 100 characters of answer
                    answer_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
                    print(f"    Answer: {answer_preview}")
                    
                else:
                    self.print_step(f"✗ Query failed: {result.get('error', 'Unknown error')}", "ERROR")
                
                # Small delay between queries
                time.sleep(0.5)
                
            except Exception as e:
                self.print_step(f"✗ Query failed with exception: {str(e)}", "ERROR")
                self.components['metrics'].record_query(
                    query=query,
                    response_time_ms=0,
                    success=False
                )
        
        return results
    
    def demonstrate_feedback(self):
        """Demonstrate the feedback system"""
        self.print_header("DEMONSTRATING FEEDBACK SYSTEM")
        
        # Simulate some feedback
        sample_feedback = [
            ("How do I reset my password?", "To reset your password...", 5, "Very helpful!"),
            ("What payment methods do you accept?", "We accept credit cards...", 4, "Good information"),
            ("The application is running slowly", "If you're experiencing slow performance...", 3, "Partially helpful"),
        ]
        
        for query, answer, rating, feedback_text in sample_feedback:
            try:
                feedback_id = self.components['metrics'].record_feedback(
                    query=query,
                    answer=answer,
                    rating=rating,
                    feedback_text=feedback_text
                )
                self.print_step(f"Recorded feedback (Rating: {rating}/5): {feedback_id[:8]}...", "SUCCESS")
            except Exception as e:
                self.print_step(f"Failed to record feedback: {str(e)}", "ERROR")
    
    def show_performance_metrics(self):
        """Display performance metrics and analytics"""
        self.print_header("PERFORMANCE METRICS & ANALYTICS")
        
        try:
            metrics = self.components['metrics'].get_metrics()
            
            # Basic metrics
            self.print_step("Basic Statistics:", "INFO")
            print(f"    Total Queries: {metrics['total_queries']}")
            print(f"    Successful Queries: {metrics['successful_queries']}")
            print(f"    Failed Queries: {metrics['failed_queries']}")
            print(f"    Error Rate: {metrics['error_rate']}%")
            print(f"    Average Response Time: {metrics['avg_response_time_ms']:.0f}ms")
            
            if metrics['avg_rating']:
                print(f"    Average User Rating: {metrics['avg_rating']:.1f}/5")
                print(f"    Total Feedback: {metrics['total_feedback']}")
            
            # Top queries
            if metrics['most_common_queries']:
                self.print_step("Most Common Queries:", "INFO")
                for i, query_info in enumerate(metrics['most_common_queries'][:5], 1):
                    print(f"    {i}. {query_info['query']} ({query_info['count']} times)")
            
            # Detailed analytics
            detailed_metrics = self.components['metrics'].get_detailed_analytics()
            
            self.print_step("Performance Details:", "INFO")
            perf_stats = detailed_metrics['performance_stats']
            print(f"    Min Response Time: {perf_stats['min_response_time']:.0f}ms")
            print(f"    Max Response Time: {perf_stats['max_response_time']:.0f}ms")
            print(f"    P50 Response Time: {perf_stats['p50_response_time']:.0f}ms")
            print(f"    P95 Response Time: {perf_stats['p95_response_time']:.0f}ms")
            
            retrieval_stats = detailed_metrics['retrieval_stats']
            print(f"    Avg Documents Retrieved: {retrieval_stats['avg_docs_retrieved']:.1f}")
            print(f"    Avg Similarity Score: {retrieval_stats['avg_similarity_score']:.3f}")
            
        except Exception as e:
            self.print_step(f"Failed to generate metrics: {str(e)}", "ERROR")
    
    def save_demo_results(self, results: List[Dict[str, Any]]):
        """Save demo results to file"""
        self.print_header("SAVING DEMO RESULTS")
        
        try:
            # Create demo results directory
            results_dir = project_root / "demo_results"
            results_dir.mkdir(exist_ok=True)
            
            # Save query results
            results_file = results_dir / f"demo_queries_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.print_step(f"Query results saved to: {results_file}", "SUCCESS")
            
            # Save metrics
            metrics_export = self.components['metrics'].export_metrics()
            metrics_file = results_dir / f"demo_metrics_{int(time.time())}.json"
            with open(metrics_file, 'w') as f:
                f.write(metrics_export)
            
            self.print_step(f"Metrics saved to: {metrics_file}", "SUCCESS")
            
        except Exception as e:
            self.print_step(f"Failed to save results: {str(e)}", "ERROR")
    
    def run_interactive_mode(self):
        """Run interactive query mode"""
        self.print_header("INTERACTIVE MODE")
        self.print_step("Enter your questions (type 'quit' to exit):", "INFO")
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print("🤖 Thinking...")
                start_time = time.time()
                
                result = self.components['rag_engine'].query(
                    question=query,
                    top_k=5,
                    score_threshold=0.3
                )
                
                response_time = (time.time() - start_time) * 1000
                
                print(f"\n💬 Answer ({response_time:.0f}ms):")
                print(result['answer'])
                
                if result.get('sources'):
                    print(f"\n📚 Sources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source['source']} (Score: {source['score']:.3f})")
                
                # Record metrics
                self.components['metrics'].record_query(
                    query=query,
                    response_time_ms=response_time,
                    success='error' not in result
                )
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        self.print_step("Interactive mode ended", "INFO")
    
    def run_full_demo(self):
        """Run the complete demo"""
        print("\n🚀 Customer Support RAG System - Full Demo")
        print("=" * 60)
        
        try:
            # Initialize system
            self.initialize_components()
            
            # Health check
            health_status = self.check_system_health()
            
            # Only continue if core components are healthy
            if not (health_status.get('vector_store', False) and health_status.get('llm', False)):
                self.print_step("Critical components are unhealthy. Cannot continue demo.", "ERROR")
                return
            
            # Process documents (if collection is empty)
            collection_info = self.components['vector_store'].get_collection_info()
            if collection_info.get('points_count', 0) == 0:
                self.process_sample_documents()
            else:
                self.print_step(f"Using existing collection with {collection_info['points_count']} documents", "INFO")
            
            # Run demo queries
            results = self.run_demo_queries()
            
            # Demonstrate feedback
            self.demonstrate_feedback()
            
            # Show metrics
            self.show_performance_metrics()
            
            # Save results
            self.save_demo_results(results)
            
            # Ask if user wants interactive mode
            self.print_header("DEMO COMPLETE")
            response = input("\nWould you like to try interactive mode? [y/N]: ")
            if response.lower().startswith('y'):
                self.run_interactive_mode()
            
            self.print_step("Demo completed successfully! 🎉", "SUCCESS")
            
        except Exception as e:
            self.print_step(f"Demo failed: {str(e)}", "ERROR")
            raise

def main():
    """Main function"""
    demo = RAGSystemDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()