"""
Advanced RAG Techniques Implementation

This module implements cutting-edge RAG techniques including:
- HyDE (Hypothetical Document Embeddings)
- Query Expansion with semantic similarity
- Multi-hop reasoning
- Contextual re-ranking
- Adaptive retrieval strategies
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import ollama
from utils.embeddings import EmbeddingGenerator
from backend.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAGEngine:
    """
    Advanced RAG engine with state-of-the-art retrieval techniques
    """
    
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
        self.client = ollama.Client(host=ollama_host)
        
        # Cache for query expansions and hypothetical documents
        self.query_cache = {}
        self.hyde_cache = {}
        
    def _generate_hypothetical_document(self, query: str) -> str:
        """
        HyDE: Generate a hypothetical document that would answer the query
        This technique often retrieves more relevant documents than the raw query
        """
        if query in self.hyde_cache:
            return self.hyde_cache[query]
        
        hyde_prompt = f"""You are an expert customer support agent. For the following question, write a detailed, helpful answer that a customer support document might contain. Write it as if it were already in our knowledge base.

Question: {query}

Write a comprehensive answer (2-3 paragraphs) that would typically be found in support documentation:"""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=hyde_prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 300,
                    'stop': ['Question:', 'Q:', '\n\nQuestion']
                }
            )
            
            hypothetical_doc = response['response'].strip()
            self.hyde_cache[query] = hypothetical_doc
            
            logger.info(f"Generated HyDE document for query: {query[:50]}...")
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"Failed to generate HyDE document: {str(e)}")
            return query
    
    def _expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Generate semantically similar queries for better retrieval coverage
        """
        if query in self.query_cache:
            return self.query_cache[query]
        
        expansion_prompt = f"""Generate {num_expansions} alternative ways to ask the same question. These should cover different phrasings, synonyms, and related concepts that a customer might use.

Original question: {query}

Alternative questions:
1."""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=expansion_prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': 200,
                    'stop': ['Original question:', 'Note:', '\n\n']
                }
            )
            
            # Parse the expanded queries
            expanded_text = response['response'].strip()
            expanded_queries = []
            
            for line in expanded_text.split('\n'):
                line = line.strip()
                if line and any(line.startswith(str(i)) for i in range(1, 10)):
                    # Remove numbering and clean up
                    clean_query = line.split('.', 1)[-1].strip()
                    if clean_query and len(clean_query) > 10:
                        expanded_queries.append(clean_query)
            
            # Always include the original query
            all_queries = [query] + expanded_queries[:num_expansions]
            self.query_cache[query] = all_queries
            
            logger.info(f"Expanded query into {len(all_queries)} variations")
            return all_queries
            
        except Exception as e:
            logger.error(f"Failed to expand query: {str(e)}")
            return [query]
    
    def _rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Advanced re-ranking using multiple signals:
        - Semantic similarity
        - Query-document overlap
        - Document quality indicators
        """
        if len(documents) <= top_k:
            return documents
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            # Calculate multiple ranking signals
            for doc in documents:
                content = doc['content']
                
                # 1. Semantic similarity (already have from vector search)
                semantic_score = doc['metadata']['score']
                
                # 2. Query-document term overlap
                query_words = set(query.lower().split())
                doc_words = set(content.lower().split())
                overlap_score = len(query_words & doc_words) / len(query_words | doc_words) if query_words | doc_words else 0
                
                # 3. Document length penalty (very short or very long docs might be less useful)
                length = len(content)
                optimal_length = 500  # characters
                length_score = 1.0 - abs(length - optimal_length) / optimal_length if length > 0 else 0
                length_score = max(0, min(1, length_score))
                
                # 4. Source type preference (some file types might be more authoritative)
                source_score = 1.0
                file_type = doc['metadata'].get('file_type', '')
                if file_type in ['md', 'pdf']:
                    source_score = 1.1
                elif file_type == 'txt':
                    source_score = 0.9
                
                # Combine scores with weights
                combined_score = (
                    0.5 * semantic_score +
                    0.2 * overlap_score +
                    0.2 * length_score +
                    0.1 * source_score
                )
                
                doc['metadata']['rerank_score'] = combined_score
                doc['metadata']['breakdown'] = {
                    'semantic': semantic_score,
                    'overlap': overlap_score,
                    'length': length_score,
                    'source': source_score
                }
            
            # Sort by combined score and return top_k
            reranked = sorted(documents, key=lambda x: x['metadata']['rerank_score'], reverse=True)
            
            logger.info(f"Re-ranked {len(documents)} documents, returning top {top_k}")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {str(e)}")
            return documents[:top_k]
    
    def _cluster_documents(self, documents: List[Dict[str, Any]], n_clusters: int = 3) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster retrieved documents to identify different aspects of the query
        """
        if len(documents) < n_clusters:
            return {0: documents}
        
        try:
            # Extract embeddings from documents
            embeddings = []
            for doc in documents:
                # Use the document's embedding if available
                if 'embedding' in doc['metadata']:
                    embeddings.append(doc['metadata']['embedding'])
                else:
                    # Generate embedding for the content
                    embedding = self.embedding_generator.generate_single_embedding(doc['content'])
                    embeddings.append(embedding)
            
            embeddings_array = np.array(embeddings)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Group documents by cluster
            clustered_docs = {}
            for doc, label in zip(documents, cluster_labels):
                if label not in clustered_docs:
                    clustered_docs[label] = []
                clustered_docs[label].append(doc)
                doc['metadata']['cluster'] = int(label)
            
            logger.info(f"Clustered {len(documents)} documents into {len(clustered_docs)} groups")
            return clustered_docs
            
        except Exception as e:
            logger.error(f"Document clustering failed: {str(e)}")
            return {0: documents}
    
    def _adaptive_retrieval_strategy(
        self, 
        query: str, 
        query_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Adapt retrieval parameters based on query characteristics
        """
        # Analyze query characteristics
        query_length = len(query.split())
        has_specific_terms = any(term in query.lower() for term in [
            'how to', 'what is', 'error', 'problem', 'issue', 'bug', 'fail'
        ])
        
        # Determine query type if not provided
        if query_type == "general":
            if has_specific_terms or query_length > 8:
                query_type = "specific"
            elif any(term in query.lower() for term in ['help', 'support', 'contact']):
                query_type = "support"
            else:
                query_type = "exploratory"
        
        # Adapt parameters based on query type
        strategies = {
            "specific": {
                "top_k": 8,
                "score_threshold": 0.4,
                "use_hyde": True,
                "use_expansion": False,
                "rerank": True,
                "cluster": False
            },
            "exploratory": {
                "top_k": 12,
                "score_threshold": 0.2,
                "use_hyde": False,
                "use_expansion": True,
                "rerank": True,
                "cluster": True
            },
            "support": {
                "top_k": 6,
                "score_threshold": 0.3,
                "use_hyde": True,
                "use_expansion": True,
                "rerank": True,
                "cluster": False
            }
        }
        
        strategy = strategies.get(query_type, strategies["specific"])
        strategy["query_type"] = query_type
        
        logger.info(f"Using '{query_type}' retrieval strategy for query")
        return strategy
    
    async def advanced_query(
        self,
        question: str,
        strategy_override: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Advanced query processing with multiple RAG techniques
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Determine retrieval strategy
            strategy = strategy_override or self._adaptive_retrieval_strategy(question)
            
            all_documents = []
            retrieval_methods = []
            
            # 1. Standard query retrieval
            query_embedding = self.embedding_generator.generate_single_embedding(question)
            standard_docs = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=strategy["top_k"],
                score_threshold=strategy["score_threshold"]
            )
            all_documents.extend(standard_docs)
            retrieval_methods.append(f"standard({len(standard_docs)})")
            
            # 2. HyDE retrieval if enabled
            if strategy.get("use_hyde", False):
                hyde_doc = self._generate_hypothetical_document(question)
                hyde_embedding = self.embedding_generator.generate_single_embedding(hyde_doc)
                hyde_docs = self.vector_store.similarity_search(
                    query_embedding=hyde_embedding,
                    top_k=strategy["top_k"] // 2,
                    score_threshold=strategy["score_threshold"]
                )
                
                # Mark as HyDE retrieved
                for doc in hyde_docs:
                    doc['metadata']['retrieval_method'] = 'hyde'
                
                all_documents.extend(hyde_docs)
                retrieval_methods.append(f"hyde({len(hyde_docs)})")
            
            # 3. Query expansion retrieval if enabled
            if strategy.get("use_expansion", False):
                expanded_queries = self._expand_query(question)
                for exp_query in expanded_queries[1:]:  # Skip original query
                    exp_embedding = self.embedding_generator.generate_single_embedding(exp_query)
                    exp_docs = self.vector_store.similarity_search(
                        query_embedding=exp_embedding,
                        top_k=strategy["top_k"] // len(expanded_queries),
                        score_threshold=strategy["score_threshold"] * 0.8  # Lower threshold for expansions
                    )
                    
                    # Mark as expansion retrieved
                    for doc in exp_docs:
                        doc['metadata']['retrieval_method'] = 'expansion'
                        doc['metadata']['expansion_query'] = exp_query
                    
                    all_documents.extend(exp_docs)
                
                retrieval_methods.append(f"expansion({len(expanded_queries)-1})")
            
            # 4. Remove duplicates based on content
            unique_docs = []
            seen_content = set()
            for doc in all_documents:
                content_hash = hash(doc['content'][:200])  # Hash first 200 chars
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            logger.info(f"Deduplicated {len(all_documents)} -> {len(unique_docs)} documents")
            
            # 5. Re-ranking if enabled
            if strategy.get("rerank", False):
                unique_docs = self._rerank_documents(question, unique_docs, strategy["top_k"])
            
            # 6. Document clustering if enabled
            clustered_docs = {}
            if strategy.get("cluster", False) and len(unique_docs) > 3:
                clustered_docs = self._cluster_documents(unique_docs)
            
            # 7. Generate response with enhanced context
            if not unique_docs:
                return {
                    'answer': "I couldn't find any relevant information to answer your question. Please try rephrasing or contact support directly.",
                    'sources': [],
                    'context_used': False,
                    'advanced_features': {
                        'strategy': strategy,
                        'retrieval_methods': retrieval_methods,
                        'processing_time_ms': (asyncio.get_event_loop().time() - start_time) * 1000
                    }
                }
            
            # Create enhanced prompt with cluster information if available
            context_parts = []
            if clustered_docs:
                for cluster_id, cluster_docs in clustered_docs.items():
                    context_parts.append(f"\n--- Topic Group {cluster_id + 1} ---")
                    for i, doc in enumerate(cluster_docs[:3]):  # Top 3 from each cluster
                        source = doc['metadata']['source'].split('/')[-1]
                        method = doc['metadata'].get('retrieval_method', 'standard')
                        context_parts.append(
                            f"Source {len(context_parts)}: {source} [{method}] (Score: {doc['metadata']['score']:.3f})\n{doc['content']}\n"
                        )
            else:
                for i, doc in enumerate(unique_docs[:8], 1):  # Top 8 documents
                    source = doc['metadata']['source'].split('/')[-1]
                    method = doc['metadata'].get('retrieval_method', 'standard')
                    context_parts.append(
                        f"Context {i}: {source} [{method}] (Score: {doc['metadata']['score']:.3f})\n{doc['content']}\n"
                    )
            
            context = "\n".join(context_parts)
            
            # Enhanced prompt
            prompt = f"""You are an expert customer support assistant with access to comprehensive documentation. Use the provided context to give accurate, helpful answers.

CONTEXT (from multiple retrieval methods):
{context}

QUESTION: {question}

INSTRUCTIONS:
- Synthesize information from multiple sources when relevant
- If sources provide conflicting information, note the discrepancy
- Include specific details and step-by-step instructions when applicable
- Reference the most relevant sources
- If the context doesn't fully answer the question, clearly state what's missing

ANSWER:"""

            # Generate response
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9
                }
            )
            
            # Prepare sources with enhanced metadata
            sources = []
            for doc in unique_docs[:6]:  # Top 6 for display
                sources.append({
                    'source': doc['metadata']['source'].split('/')[-1],
                    'chunk_index': doc['metadata']['chunk_index'],
                    'score': round(doc['metadata']['score'], 3),
                    'retrieval_method': doc['metadata'].get('retrieval_method', 'standard'),
                    'rerank_score': doc['metadata'].get('rerank_score'),
                    'cluster': doc['metadata'].get('cluster'),
                    'preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                })
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                'answer': response['response'].strip(),
                'sources': sources,
                'context_used': True,
                'advanced_features': {
                    'strategy': strategy,
                    'retrieval_methods': retrieval_methods,
                    'documents_processed': len(all_documents),
                    'unique_documents': len(unique_docs),
                    'clusters': len(clustered_docs) if clustered_docs else 0,
                    'processing_time_ms': round(processing_time, 2)
                },
                'generation_stats': {
                    'model': response['model'],
                    'total_duration_ms': response.get('total_duration', 0) // 1000000,
                    'prompt_tokens': response.get('prompt_eval_count', 0),
                    'completion_tokens': response.get('eval_count', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Advanced query processing failed: {str(e)}")
            return {
                'answer': f"I encountered an error while processing your advanced query: {str(e)}",
                'sources': [],
                'context_used': False,
                'error': str(e),
                'advanced_features': {'error': True}
            }

class MultiHopRAG:
    """
    Multi-hop reasoning for complex queries requiring information synthesis
    """
    
    def __init__(self, advanced_rag: AdvancedRAGEngine):
        self.advanced_rag = advanced_rag
        self.reasoning_cache = {}
    
    def _decompose_query(self, complex_query: str) -> List[str]:
        """
        Break down complex queries into simpler sub-queries
        """
        decomposition_prompt = f"""Break down this complex question into 2-4 simpler questions that, when answered together, would provide a complete response to the original question.

Complex Question: {complex_query}

Sub-questions:
1."""
        
        try:
            response = self.advanced_rag.client.generate(
                model=self.advanced_rag.model_name,
                prompt=decomposition_prompt,
                options={'temperature': 0.3, 'num_predict': 200}
            )
            
            # Parse sub-questions
            text = response['response'].strip()
            sub_queries = []
            
            for line in text.split('\n'):
                line = line.strip()
                if line and any(line.startswith(str(i)) for i in range(1, 10)):
                    clean_query = line.split('.', 1)[-1].strip()
                    if clean_query and len(clean_query) > 5:
                        sub_queries.append(clean_query)
            
            return sub_queries[:4]  # Limit to 4 sub-queries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {str(e)}")
            return [complex_query]
    
    async def multi_hop_query(self, complex_query: str) -> Dict[str, Any]:
        """
        Process multi-hop queries by decomposing and synthesizing
        """
        try:
            # Decompose the query
            sub_queries = self._decompose_query(complex_query)
            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            
            # Process each sub-query
            sub_results = []
            for sub_query in sub_queries:
                result = await self.advanced_rag.advanced_query(
                    question=sub_query,
                    max_tokens=512  # Shorter responses for sub-queries
                )
                sub_results.append({
                    'sub_query': sub_query,
                    'answer': result['answer'],
                    'sources': result.get('sources', [])
                })
            
            # Synthesize final answer
            synthesis_context = "\n\n".join([
                f"Sub-question: {sr['sub_query']}\nAnswer: {sr['answer']}"
                for sr in sub_results
            ])
            
            synthesis_prompt = f"""Based on the answers to related sub-questions, provide a comprehensive answer to the original complex question.

Original Question: {complex_query}

Sub-question Answers:
{synthesis_context}

Instructions:
- Synthesize the information into a coherent, complete answer
- Identify connections between the sub-answers
- Address all aspects of the original question
- If there are any gaps in information, mention them

Comprehensive Answer:"""
            
            final_response = self.advanced_rag.client.generate(
                model=self.advanced_rag.model_name,
                prompt=synthesis_prompt,
                options={'temperature': 0.5, 'num_predict': 1024}
            )
            
            # Collect all sources
            all_sources = []
            for sr in sub_results:
                all_sources.extend(sr['sources'])
            
            # Remove duplicate sources
            unique_sources = []
            seen = set()
            for source in all_sources:
                key = (source['source'], source['chunk_index'])
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(source)
            
            return {
                'answer': final_response['response'].strip(),
                'sources': unique_sources[:8],  # Top 8 unique sources
                'context_used': True,
                'multi_hop_info': {
                    'original_query': complex_query,
                    'sub_queries': sub_queries,
                    'sub_results': len(sub_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-hop query failed: {str(e)}")
            return {
                'answer': f"I encountered an error processing this complex query: {str(e)}",
                'sources': [],
                'context_used': False,
                'error': str(e)
            }