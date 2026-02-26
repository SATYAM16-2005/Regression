import streamlit as st
import os
import tempfile
import certifi
from backend.rag_engine import RAGEngine

# Fix for SSL certificate issues in some Windows environments
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    .upload-section {
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📄 RAG Document Prototype")
    st.markdown("---")

    # Initialize RAG Engine in session state
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for API Key
    with st.sidebar:
        st.header("Settings")
        previous_key = os.environ.get("GOOGLE_API_KEY", "")
        api_key = st.text_input("Google API Key", type="password", value=previous_key)
        
        if api_key and api_key != previous_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            # Reset engine and chat history if key changes
            st.session_state.rag_engine = RAGEngine()
            st.session_state.chat_history = []
            st.success("API Key applied and engine reset!")
        
        st.markdown("---")
        st.info("Upload a PDF to start chatting with your document.")

    # Check if API Key is set
    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("Please enter your Google API Key in the sidebar to begin.")
        st.stop()

    # File Upload Section
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                # Use project-local temp directory to avoid path issues
                temp_dir = "temp_docs"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                tmp_path = os.path.abspath(os.path.join(temp_dir, uploaded_file.name))
                
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    num_chunks = st.session_state.rag_engine.process_pdf(tmp_path)
                    st.session_state.processed_file = uploaded_file.name
                    st.success(f"PDF processed successfully! Created {num_chunks} chunks.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    # Print full error trace to console for easier debugging
                    import traceback
                    print(traceback.format_exc())
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    # Chat Interface
    st.markdown("### Chat with your Document")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_engine.query(prompt)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
