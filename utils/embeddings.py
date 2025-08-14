import os
import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model"""
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.dimension = len(test_embedding[0])
            
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.model:
            raise ValueError("Embedding model not loaded")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # Convert to list of lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = self.generate_embeddings([text])
        return embeddings[0]
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Add embeddings to document metadata"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to document metadata
        for doc, embedding in zip(documents, embeddings):
            doc.metadata['embedding'] = embedding
        
        return documents
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension