#!/bin/bash

# Customer Support RAG System Setup Script
# This script sets up the entire system for development and production

set -e

echo "🚀 Setting up Customer Support RAG System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if running on supported OS
check_os() {
    print_header "🔍 Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_status "Detected Windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check if required tools are installed
check_dependencies() {
    print_header "🔧 Checking dependencies..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_status "pip3 found"
    else
        print_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_status "Docker found"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker not found - Docker deployment will not be available"
        DOCKER_AVAILABLE=false
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_status "Docker Compose found"
        DOCKER_COMPOSE_AVAILABLE=true
    else
        print_warning "Docker Compose not found - use 'docker compose' instead"
        DOCKER_COMPOSE_AVAILABLE=false
    fi
    
    # Check curl
    if command -v curl &> /dev/null; then
        print_status "curl found"
    else
        print_error "curl is required but not installed"
        exit 1
    fi
}

# Install Ollama
install_ollama() {
    print_header "🦙 Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        print_status "Ollama already installed"
        return
    fi
    
    if [[ "$OS" == "linux" ]] || [[ "$OS" == "macos" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OS" == "windows" ]]; then
        print_warning "Please install Ollama manually from https://ollama.com/download"
        print_warning "After installation, restart your terminal and run this script again"
        exit 1
    fi
    
    print_status "Ollama installed successfully"
}

# Install Python dependencies
install_python_deps() {
    print_header "🐍 Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python packages..."
    pip install -r requirements.txt
    
    print_status "Python dependencies installed successfully"
}

# Setup Qdrant
setup_qdrant() {
    print_header "🗄️ Setting up Qdrant vector database..."
    
    if [[ "$DOCKER_AVAILABLE" == true ]]; then
        print_status "Starting Qdrant with Docker..."
        docker run -d --name qdrant-rag \
            -p 6333:6333 \
            -p 6334:6334 \
            -v qdrant_storage:/qdrant/storage:z \
            qdrant/qdrant:v1.7.0
        
        # Wait for Qdrant to start
        print_status "Waiting for Qdrant to start..."
        sleep 10
        
        # Test connection
        if curl -f http://localhost:6333/health &> /dev/null; then
            print_status "Qdrant is running and healthy"
        else
            print_warning "Qdrant may not be ready yet. It will continue starting in the background."
        fi
    else
        print_warning "Docker not available. Please install Qdrant manually:"
        print_warning "Visit: https://qdrant.tech/documentation/guides/installation/"
    fi
}

# Pull and setup Ollama model
setup_ollama_model() {
    print_header "🤖 Setting up Ollama model..."
    
    # Check if Ollama is running
    if ! pgrep -x "ollama" > /dev/null; then
        print_status "Starting Ollama service..."
        if [[ "$OS" == "macos" ]]; then
            # On macOS, Ollama runs as a background service
            ollama serve &
            sleep 5
        elif [[ "$OS" == "linux" ]]; then
            # On Linux, start as background process
            nohup ollama serve > /dev/null 2>&1 &
            sleep 5
        fi
    fi
    
    # Pull the model
    print_status "Pulling LLaMA 3.2 3B model (this may take a while)..."
    ollama pull llama3.2:3b
    
    # Test the model
    print_status "Testing model..."
    if ollama list | grep -q "llama3.2:3b"; then
        print_status "Model pulled successfully"
    else
        print_error "Failed to pull model"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_header "📁 Creating directories..."
    
    mkdir -p data/documents
    mkdir -p data/embeddings  
    mkdir -p models
    mkdir -p monitoring/metrics_data
    mkdir -p logs
    
    print_status "Directories created"
}

# Setup environment file
setup_environment() {
    print_header "⚙️ Setting up environment..."
    
    if [[ ! -f .env ]]; then
        print_status "Environment file already exists"
        return
    fi
    
    print_status "Environment file created with default settings"
}

# Load sample data
load_sample_data() {
    print_header "📄 Processing sample documents..."
    
    # Activate virtual environment
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Run the ingestion script
    print_status "Processing documents and creating embeddings..."
    python3 -c "
import sys
sys.path.append('.')
from utils.document_loader import DocumentLoader
from utils.embeddings import EmbeddingGenerator
from backend.vector_store import QdrantVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

print('Loading documents...')
doc_loader = DocumentLoader()
documents = doc_loader.process_documents('data/documents')

print(f'Loaded {len(documents)} document chunks')

print('Generating embeddings...')
embedding_gen = EmbeddingGenerator()
embedded_docs = embedding_gen.embed_documents(documents)

print('Storing in vector database...')
vector_store = QdrantVectorStore(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', '6333')),
    collection_name=os.getenv('COLLECTION_NAME', 'support_docs'),
    vector_dimension=embedding_gen.get_dimension()
)

success = vector_store.add_documents(embedded_docs)
if success:
    print('Sample data loaded successfully!')
else:
    print('Failed to load sample data')
"
}

# Display setup completion info
show_completion() {
    print_header "✅ Setup completed successfully!"
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                    🎉 SETUP COMPLETE!                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    print_status "Services are ready to start:"
    echo ""
    echo -e "  ${YELLOW}1. Backend API:${NC}"
    echo -e "     ${BLUE}cd $(pwd) && source venv/bin/activate && python -m backend.main${NC}"
    echo -e "     ${BLUE}API will be available at: http://localhost:8000${NC}"
    echo ""
    echo -e "  ${YELLOW}2. Frontend UI:${NC}"
    echo -e "     ${BLUE}cd $(pwd) && source venv/bin/activate && streamlit run frontend/app.py${NC}"
    echo -e "     ${BLUE}UI will be available at: http://localhost:8501${NC}"
    echo ""
    
    if [[ "$DOCKER_AVAILABLE" == true ]]; then
        echo -e "  ${YELLOW}3. Or use Docker Compose (all services):${NC}"
        echo -e "     ${BLUE}docker-compose up --build${NC}"
        echo ""
    fi
    
    print_status "Service URLs:"
    echo -e "  - Frontend: ${BLUE}http://localhost:8501${NC}"
    echo -e "  - Backend API: ${BLUE}http://localhost:8000${NC}"
    echo -e "  - API Docs: ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "  - Qdrant UI: ${BLUE}http://localhost:6333/dashboard${NC}"
    echo ""
    
    print_status "Next steps:"
    echo "  1. Start the services using the commands above"
    echo "  2. Upload your own documents through the web UI"
    echo "  3. Start asking questions!"
    echo ""
    
    print_warning "Note: If you encounter any issues, check the troubleshooting guide in README.md"
}

# Main setup function
main() {
    print_header "🚀 Customer Support RAG System Setup"
    echo "This script will set up the complete system including:"
    echo "  - Python dependencies"
    echo "  - Ollama LLM (LLaMA 3.2)"
    echo "  - Qdrant vector database"
    echo "  - Sample data processing"
    echo ""
    
    read -p "Continue with setup? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled"
        exit 0
    fi
    
    check_os
    check_dependencies
    create_directories
    setup_environment
    install_python_deps
    install_ollama
    setup_ollama_model
    
    if [[ "$DOCKER_AVAILABLE" == true ]]; then
        setup_qdrant
    else
        print_warning "Skipping Qdrant setup - Docker not available"
        print_warning "Please install Qdrant manually or install Docker"
    fi
    
    # Only load sample data if Qdrant is available
    if curl -f http://localhost:6333/health &> /dev/null; then
        load_sample_data
    else
        print_warning "Qdrant not available - skipping sample data load"
        print_warning "You can load sample data later once Qdrant is running"
    fi
    
    show_completion
}

# Run main function
main "$@"