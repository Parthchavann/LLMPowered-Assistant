import streamlit as st
import requests
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Customer Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-items: flex-end;
    }
    .assistant-message {
        background-color: #f5f5f5;
        align-items: flex-start;
    }
    .message-content {
        max-width: 80%;
        padding: 0.5rem 1rem;
        border-radius: 18px;
    }
    .user-content {
        background-color: #1976d2;
        color: white;
    }
    .assistant-content {
        background-color: white;
        border: 1px solid #e0e0e0;
    }
    .source-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def query_api(question, top_k=5, score_threshold=0.3, max_tokens=2048, temperature=0.7):
    """Query the RAG API"""
    try:
        payload = {
            "question": question,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except requests.RequestException as e:
        return None, f"Connection Error: {str(e)}"

def submit_feedback(query, answer, rating, feedback_text=""):
    """Submit user feedback"""
    try:
        payload = {
            "query": query,
            "answer": answer,
            "rating": rating,
            "feedback_text": feedback_text
        }
        
        response = requests.post(f"{API_BASE_URL}/feedback", json=payload)
        return response.status_code == 200
    except:
        return False

def get_metrics():
    """Get system metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def upload_documents(files, overwrite=False, chunk_size=500, chunk_overlap=50):
    """Upload documents to the system"""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.read(), file.type)))
        
        data = {
            "overwrite": overwrite,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files_data,
            data=data,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Upload failed: {response.text}"
    except Exception as e:
        return None, f"Upload error: {str(e)}"

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_healthy" not in st.session_state:
        st.session_state.api_healthy = check_api_health()

    # Sidebar
    with st.sidebar:
        st.title("🤖 Customer Support")
        st.markdown("---")
        
        # API Status
        if st.session_state.api_healthy:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Disconnected")
            if st.button("Retry Connection"):
                st.session_state.api_healthy = check_api_health()
                st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        with st.expander("Query Settings"):
            top_k = st.slider("Top K Results", 1, 20, 5)
            score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.3, 0.1)
            max_tokens = st.slider("Max Tokens", 100, 4096, 2048, 100)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        
        st.markdown("---")
        
        # Document Management
        st.subheader("📄 Document Management")
        
        with st.expander("Upload Documents"):
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'md'],
                accept_multiple_files=True
            )
            
            overwrite = st.checkbox("Overwrite existing documents")
            
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("Chunk Size", 100, 1000, 500)
            with col2:
                chunk_overlap = st.number_input("Chunk Overlap", 0, 200, 50)
            
            if st.button("Upload Documents") and uploaded_files:
                with st.spinner("Processing documents..."):
                    result, error = upload_documents(
                        uploaded_files, overwrite, chunk_size, chunk_overlap
                    )
                    
                    if result:
                        st.success(f"✅ {result['message']}")
                        st.info(f"📊 Processed: {result['documents_processed']} docs, {result['chunks_created']} chunks")
                        st.info(f"⏱️ Time: {result['processing_time_seconds']}s")
                    else:
                        st.error(f"❌ {error}")
        
        st.markdown("---")
        
        # Clear Chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Main content
    st.title("Customer Support Assistant")
    st.markdown("Ask questions about your documents and get AI-powered answers!")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "ℹ️ System Info"])
    
    with tab1:
        # Chat interface
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-content user-content">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="message-content assistant-content">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander(f"📚 Sources ({len(message['sources'])})"):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {i}:</strong> {source['source']} (Chunk {source['chunk_index']}, Score: {source['score']})<br>
                                    <em>{source['preview']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Show stats if available
                    if "stats" in message:
                        stats = message["stats"]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Response Time", f"{stats.get('response_time_ms', 0):.0f}ms")
                        with col2:
                            st.metric("Documents Used", stats.get('documents_retrieved', 0))
                        with col3:
                            st.metric("Avg Score", f"{stats.get('avg_score', 0):.3f}")
                    
                    # Feedback section
                    feedback_key = f"feedback_{len(st.session_state.messages)-1}"
                    if feedback_key not in st.session_state:
                        st.session_state[feedback_key] = False
                    
                    if not st.session_state[feedback_key]:
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button("👍👎 Rate", key=f"rate_btn_{len(st.session_state.messages)-1}"):
                                st.session_state[f"show_feedback_{len(st.session_state.messages)-1}"] = True
                        
                        if st.session_state.get(f"show_feedback_{len(st.session_state.messages)-1}", False):
                            with st.form(f"feedback_form_{len(st.session_state.messages)-1}"):
                                rating = st.radio("Rating", [1, 2, 3, 4, 5], horizontal=True)
                                feedback_text = st.text_area("Additional feedback (optional)")
                                
                                if st.form_submit_button("Submit Feedback"):
                                    # Get the corresponding user message
                                    user_msg = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
                                    
                                    success = submit_feedback(user_msg, message["content"], rating, feedback_text)
                                    if success:
                                        st.success("Thank you for your feedback!")
                                        st.session_state[feedback_key] = True
                                        st.rerun()
                                    else:
                                        st.error("Failed to submit feedback")

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.api_healthy:
                st.error("❌ API is not available. Please check the connection.")
                return
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Query the API
            with st.spinner("Thinking..."):
                start_time = time.time()
                result, error = query_api(prompt, top_k, score_threshold, max_tokens, temperature)
                response_time = (time.time() - start_time) * 1000
                
                if result:
                    # Add assistant response
                    assistant_message = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "stats": {
                            "response_time_ms": response_time,
                            "documents_retrieved": result.get("retrieval_stats", {}).get("documents_retrieved", 0),
                            "avg_score": result.get("retrieval_stats", {}).get("avg_score", 0)
                        }
                    }
                    st.session_state.messages.append(assistant_message)
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Sorry, I encountered an error: {error}"
                    })
            
            st.rerun()
    
    with tab2:
        # Analytics tab
        st.subheader("📊 System Analytics")
        
        metrics = get_metrics()
        if metrics:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #1976d2;">Total Queries</h3>
                    <h2>{}</h2>
                </div>
                """.format(metrics.get('total_queries', 0)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #388e3c;">Avg Response Time</h3>
                    <h2>{:.0f}ms</h2>
                </div>
                """.format(metrics.get('avg_response_time_ms', 0)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #f57c00;">Avg Rating</h3>
                    <h2>{:.1f}/5</h2>
                </div>
                """.format(metrics.get('avg_rating', 0) or 0), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #d32f2f;">Error Rate</h3>
                    <h2>{:.1f}%</h2>
                </div>
                """.format(metrics.get('error_rate', 0)), unsafe_allow_html=True)
            
            # Most common queries
            if metrics.get('most_common_queries'):
                st.subheader("🔍 Most Common Queries")
                queries_df = pd.DataFrame(metrics['most_common_queries'])
                if not queries_df.empty:
                    fig = px.bar(queries_df, x='count', y='query', orientation='h',
                               title="Query Frequency")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No analytics data available yet. Start asking questions to see metrics!")
    
    with tab3:
        # System info tab
        st.subheader("ℹ️ System Information")
        
        # API Health Check
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("**API Health Status**")
        with col2:
            if st.button("🔄 Refresh"):
                st.session_state.api_healthy = check_api_health()
                st.rerun()
        
        if st.session_state.api_healthy:
            try:
                health_response = requests.get(f"{API_BASE_URL}/health")
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    
                    # Services status
                    st.write("**Services Status:**")
                    services = health_data.get('services', {})
                    for service, status in services.items():
                        icon = "🟢" if status else "🔴"
                        st.write(f"{icon} {service.replace('_', ' ').title()}: {'Healthy' if status else 'Unhealthy'}")
                    
                    # Vector store info
                    if health_data.get('vector_store_info'):
                        st.write("**Vector Store Information:**")
                        vsi = health_data['vector_store_info']
                        st.json({
                            "Collection": vsi.get('name', 'N/A'),
                            "Documents": vsi.get('points_count', 0),
                            "Vector Dimension": vsi.get('config', {}).get('vector_size', 'N/A'),
                            "Distance Metric": vsi.get('config', {}).get('distance', 'N/A')
                        })
                    
                    # Model info
                    if health_data.get('model_info'):
                        st.write("**LLM Model Information:**")
                        model_info = health_data['model_info']
                        if 'error' not in model_info:
                            st.json({
                                "Model Name": model_info.get('name', 'N/A'),
                                "Model Size": f"{model_info.get('size', 0) / 1e9:.1f}GB" if model_info.get('size') else 'N/A',
                                "Last Modified": model_info.get('modified_at', 'N/A')[:19] if model_info.get('modified_at') else 'N/A'
                            })
                        else:
                            st.error(f"Model Error: {model_info['error']}")
            except Exception as e:
                st.error(f"Failed to fetch system information: {str(e)}")
        else:
            st.error("🔴 API is not responding. Please check if the backend service is running.")
            
            st.markdown("""
            **Troubleshooting Steps:**
            1. Make sure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`
            2. Make sure Ollama is running: `ollama serve`
            3. Make sure the model is pulled: `ollama pull llama3.2:3b`
            4. Start the backend: `cd backend && uvicorn main:app --reload`
            """)

if __name__ == "__main__":
    main()