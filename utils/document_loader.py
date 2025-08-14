import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and extract text from PDF file"""
        documents = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "file_type": "pdf",
                        "total_pages": len(pdf_reader.pages)
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            
        return documents
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """Load text from TXT or MD file"""
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_type": Path(file_path).suffix[1:],
                    "file_size": len(content)
                }
            )
            documents.append(doc)
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            
        return documents
    
    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """Load all supported documents from directory"""
        all_documents = []
        supported_extensions = {'.pdf', '.txt', '.md'}
        
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Loading document: {file_path}")
                
                if file_path.suffix.lower() == '.pdf':
                    docs = self.load_pdf(str(file_path))
                else:
                    docs = self.load_text_file(str(file_path))
                
                all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} documents")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk index to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content)
                })
            
            all_chunks.extend(chunks)
        
        logger.info(f"Split into {len(all_chunks)} chunks")
        return all_chunks

    def process_documents(self, directory: str) -> List[Document]:
        """Complete document processing pipeline"""
        # Load documents
        documents = self.load_documents_from_directory(directory)
        
        if not documents:
            logger.warning("No documents found to process")
            return []
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        return chunks