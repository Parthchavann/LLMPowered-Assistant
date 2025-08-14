import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import ollama
from utils.embeddings import EmbeddingGenerator
from backend.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_generator: EmbeddingGenerator,
        model_name: str = "llama3.2:3b",
        ollama_host: str = "http://localhost:11434"
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.client = ollama.Client(host=ollama_host)
        
        # Verify model is available
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if the specified model is available"""
        try:
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                logger.info(f"Attempting to pull {self.model_name}")
                self.client.pull(self.model_name)
            
            logger.info(f"Model {self.model_name} is available")
            
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            raise
    
    def _create_context_prompt(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Create a prompt with retrieved context"""
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            source = doc['metadata']['source'].split('/')[-1]  # Get filename
            chunk_info = f"[Source: {source}, Chunk {doc['metadata']['chunk_index']}, Score: {doc['metadata']['score']:.3f}]"
            context_parts.append(f"Context {i}: {chunk_info}\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a helpful customer support assistant. Use the provided context to answer the user's question accurately and helpfully.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise but comprehensive
- Include relevant details from the context
- If you mention information, reference which source it came from
- Use a helpful and professional tone

ANSWER:"""
        
        return prompt
    
    def _generate_response(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response using Ollama"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                    'stop': ['</s>', '<|end|>', '<|im_end|>']
                }
            )
            
            return {
                'response': response['response'].strip(),
                'model': response['model'],
                'total_duration': response.get('total_duration', 0),
                'load_duration': response.get('load_duration', 0),
                'prompt_eval_count': response.get('prompt_eval_count', 0),
                'eval_count': response.get('eval_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def query(
        self, 
        question: str, 
        top_k: int = 5, 
        score_threshold: float = 0.3,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Main RAG query method
        
        Args:
            question: User's question
            top_k: Number of similar documents to retrieve
            score_threshold: Minimum similarity score for retrieved documents
            max_tokens: Maximum tokens in response
            temperature: Generation temperature
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Step 1: Generate embedding for the question
            logger.info(f"Processing query: {question[:100]}...")
            question_embedding = self.embedding_generator.generate_single_embedding(question)
            
            # Step 2: Retrieve similar documents
            similar_docs = self.vector_store.similarity_search(
                query_embedding=question_embedding,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            if not similar_docs:
                return {
                    'answer': "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or contact support directly.",
                    'sources': [],
                    'context_used': False,
                    'retrieval_stats': {
                        'documents_retrieved': 0,
                        'avg_score': 0.0
                    }
                }
            
            # Step 3: Create prompt with context
            prompt = self._create_context_prompt(question, similar_docs)
            
            # Step 4: Generate response
            response_data = self._generate_response(prompt, max_tokens, temperature)
            
            # Step 5: Prepare sources information
            sources = []
            for doc in similar_docs:
                sources.append({
                    'source': doc['metadata']['source'].split('/')[-1],
                    'chunk_index': doc['metadata']['chunk_index'],
                    'score': round(doc['metadata']['score'], 3),
                    'preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                })
            
            # Calculate retrieval statistics
            scores = [doc['metadata']['score'] for doc in similar_docs]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            result = {
                'answer': response_data['response'],
                'sources': sources,
                'context_used': True,
                'retrieval_stats': {
                    'documents_retrieved': len(similar_docs),
                    'avg_score': round(avg_score, 3),
                    'score_threshold': score_threshold
                },
                'generation_stats': {
                    'model': response_data['model'],
                    'total_duration_ms': response_data.get('total_duration', 0) // 1000000,  # Convert to ms
                    'prompt_tokens': response_data.get('prompt_eval_count', 0),
                    'completion_tokens': response_data.get('eval_count', 0)
                }
            }
            
            logger.info(f"Query processed successfully. Retrieved {len(similar_docs)} documents.")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'context_used': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            models = self.client.list()
            current_model = None
            
            for model in models['models']:
                if model['name'] == self.model_name:
                    current_model = model
                    break
            
            if current_model:
                return {
                    'name': current_model['name'],
                    'size': current_model.get('size', 0),
                    'modified_at': current_model.get('modified_at', ''),
                    'digest': current_model.get('digest', '')
                }
            else:
                return {'error': f'Model {self.model_name} not found'}
                
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}