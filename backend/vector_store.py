import logging
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
from langchain.schema import Document
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantVectorStore:
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6333, 
        collection_name: str = "support_docs",
        vector_dimension: int = 384
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant instance"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            
            # Create collection if it doesn't exist
            self._create_collection_if_not_exists()
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents with embeddings to the vector store"""
        try:
            points = []
            
            for doc in documents:
                if 'embedding' not in doc.metadata:
                    logger.warning(f"Document missing embedding: {doc.metadata.get('source', 'unknown')}")
                    continue
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=doc.metadata['embedding'],
                    payload={
                        'content': doc.page_content,
                        'source': doc.metadata.get('source', ''),
                        'file_type': doc.metadata.get('file_type', ''),
                        'chunk_index': doc.metadata.get('chunk_index', 0),
                        'total_chunks': doc.metadata.get('total_chunks', 1),
                        'chunk_size': doc.metadata.get('chunk_size', len(doc.page_content))
                    }
                )
                points.append(point)
            
            if not points:
                logger.warning("No valid documents to add")
                return False
            
            # Upsert points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Added {len(points)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            results = []
            for point in search_result:
                result = {
                    'content': point.payload['content'],
                    'metadata': {
                        'source': point.payload.get('source', ''),
                        'file_type': point.payload.get('file_type', ''),
                        'chunk_index': point.payload.get('chunk_index', 0),
                        'score': point.score
                    }
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def search_by_source(self, source: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by source file"""
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source)
                        )
                    ]
                ),
                limit=top_k
            )
            
            results = []
            for point in search_result[0]:  # scroll returns (points, next_page_offset)
                result = {
                    'content': point.payload['content'],
                    'metadata': {
                        'source': point.payload.get('source', ''),
                        'file_type': point.payload.get('file_type', ''),
                        'chunk_index': point.payload.get('chunk_index', 0)
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by source: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.config.name,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'config': {
                    'vector_size': info.config.params.vectors.size,
                    'distance': info.config.params.vectors.distance.name
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False