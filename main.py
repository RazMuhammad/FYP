import streamlit as st
import os
import re
from typing import Optional, List, Dict, Any
import tempfile
import time
import threading
import concurrent.futures

from app.core.decision_maker import take_decision
from app.agents.uni_agent import uni_agent
from app.agents.web_agent import web_agent
from app.agents.university_tutor import university_tutor, summarize_file
from app.utils.embeddings import set_embeddings, get_embedding_model

# Set page configuration with custom theme
st.set_page_config(
    page_title="University AI Assistant",   
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
    
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stSidebar {
        background-color: #e9ecef;
    }
    .stButton>button {
        background-color: #4b7bec;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3867d6;
        transform: translateY(-2px);
    }
    /* Style chat messages */
    .user-message {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-bottom-right-radius: 5px;
    }
    .assistant-message {
        background-color: #d6e4ff;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-bottom-left-radius: 5px;
    }
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 80px;
        text-align: center;
    }
    .loading-spinner div {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #4b7bec;
        border-radius: 100%;
        animation: loading-spinner 1.4s infinite ease-in-out both;
    }
    .loading-spinner div:nth-child(1) {
        animation-delay: -0.32s;
    }
    .loading-spinner div:nth-child(2) {
        animation-delay: -0.16s;
    }
    @keyframes loading-spinner {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1.0); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def remove_think_tags(text: str) -> str:
    """
    Removes all content between <think> and </think> tags, including the tags themselves.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def load_environment_variables():
    """Load environment variables with fallback to empty strings"""
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

def display_sidebar():
    """Display sidebar with information and options"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/student-center.png", width=100)
        st.title("University AI Assistant")
        st.markdown("### Your AI-powered academic companion")
        st.markdown("---")
        
        # File uploader in sidebar with nicer UI
        with st.expander("📁 Upload Documents", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload files for analysis",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "csv"],
                help="Supported formats: PDF, Word, Text, and CSV files"
            )
        
        st.markdown("---")
        
        with st.expander("📚 How to use:", expanded=True):
            st.markdown("1. Upload any documents (optional)")
            st.markdown("2. Type your question in the chat")
            st.markdown("3. Get intelligent responses")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.files = {}
                st.session_state.thinking_steps = ""
                st.rerun()
                
        with col2:
            if st.button("ℹ️ Help", use_container_width=True):
                st.session_state.show_help = True
                st.rerun()
        
        st.markdown("---")
        
        # Show thinking process toggle
        show_thinking = st.toggle("Show thinking process", value=False)
        if show_thinking:
            st.session_state.show_thinking = True
        else:
            st.session_state.show_thinking = False
            
        st.markdown("---")
        #   st.markdown("<div style='text-align: center'>Powered by Raz - 2025</div>", unsafe_allow_html=True)
        
        return uploaded_files

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temporary location and return the path"""
    try:
        # Create a unique filename to avoid collisions
        file_extension = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            return temp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return ""

@st.cache_data(ttl=300)
def process_response(query: str, uploaded_files, file_paths=[]) -> str:
    """Process the query and get the appropriate response"""
    # Store intermediate thinking steps for UI
    thinking_steps = []
    
    # If file is uploaded, prioritize file summarization
    if uploaded_files:
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                file_paths.append(file_path)
                thinking_steps.append(f"Processing file: {uploaded_file.name}")
                
        if file_paths:
            thinking_steps.append("Analyzing documents...")
            result = summarize_file(query, file_paths)
            return remove_think_tags(result), thinking_steps
    
    # Get the agent type classification
    thinking_steps.append("Determining the best agent for your question...")
    llm_output = take_decision(query)
    thinking_steps.append(f"Selected agent: {llm_output}")
    
    # Get response from the appropriate agent
    if llm_output == "university":
        thinking_steps.append("Searching university knowledge base...")
        embeddings = get_embedding_model()
        result = uni_agent(query, embeddings)
    elif llm_output == "web search":
        thinking_steps.append("Searching the web for current information...")
        result = web_agent(query)
    elif llm_output == "general":
        thinking_steps.append("Analyzing your academic question...")
        result = university_tutor(query)
    else:
        result = f"I'm sorry, I couldn't process your request. Please try rephrasing your question."
    
    # Clean up the response
    result = remove_think_tags(result)
    return result, thinking_steps

def display_chat_message(message, is_user=False):
    """Display a chat message with improved styling"""
    if is_user:
        st.markdown(f"""<div class="user-message">
            <strong>You:</strong><br>{message["content"]}
        </div>""", unsafe_allow_html=True)
        
        # Display file attachments if any
        if "files" in message and message["files"]:
            st.caption("Attached files:")
            for file_name in message["files"]:
                st.caption(f"- {file_name}")
    else:
        st.markdown(f"""<div class="assistant-message">
            <strong>Assistant:</strong><br>{message["content"]}
        </div>""", unsafe_allow_html=True)

def stream_response(placeholder, response, thinking_steps=None):
    """Simulate streaming for better UX"""
    full_response = response
    displayed_response = ""
    
    # First show thinking steps if available
    if thinking_steps and st.session_state.show_thinking:
        for step in thinking_steps:
            placeholder.markdown(f"🤔 {step}")
            time.sleep(0.5)
    
    # Then stream the actual response
    for i in range(len(full_response) // 10 + 1):
        chunk_end = min((i + 1) * 10, len(full_response))
        displayed_response = full_response[:chunk_end]
        placeholder.markdown(displayed_response)
        time.sleep(0.01)
    
    return displayed_response

def main():
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "files" not in st.session_state:
        st.session_state.files = {}
        
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = ""
        
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False
        
    if "show_help" not in st.session_state:
        st.session_state.show_help = False
        
    if "thinking_steps" not in st.session_state:
        st.session_state.thinking_steps = ""
    
    # Load environment variables
    load_environment_variables()
    
    # Initialize embeddings model once at startup in background thread
    if "embedding_model" not in st.session_state:
        with st.spinner("Initializing AI capabilities..."):
            st.session_state.embedding_model = set_embeddings()
    
    # Display sidebar and get uploaded files
    uploaded_files = display_sidebar()
    
    # Display help modal if requested
    if st.session_state.show_help:
        with st.container():
            st.markdown("## 📚 University AI Assistant Help")
            st.markdown("""
            ### What can this assistant do?
            - Answer university-specific questions about programs, policies, and campus resources
            - Provide academic explanations and conceptual understanding
            - Search the web for current information
            - Analyze and summarize uploaded documents
            
            ### Best practices:
            - Be specific in your questions
            - Upload relevant documents for more personalized answers
            - Use clear language and provide context
            
            ### Document support:
            - PDF files: Course materials, research papers, etc.
            - Word documents: Essays, assignments, etc.
            - Text files: Notes, data, etc.
            - CSV files: Data for analysis
            """)
            if st.button("Close"):
                st.session_state.show_help = False
                st.rerun()
    
    # Main content area with improved styling
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0; flex-grow: 1;">Chat with your University AI Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container with improved styling
    chat_container = st.container()
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(message, is_user=(message["role"] == "user"))
    
    # Chat input with improved styling
    prompt = st.chat_input("Ask your question here...")
    
    # Process input and files
    if prompt or (uploaded_files and not st.session_state.messages):
        # Create placeholder prompt if only files are uploaded
        if not prompt and uploaded_files:
            prompt = "Please summarize these files for me."
        
        # Ensure prompt is not None
        if prompt is None:
            return
            
        # Store file names
        file_names = [f.name for f in uploaded_files] if uploaded_files else []
        
        # Display user message
        with chat_container:
            display_chat_message({"role": "user", "content": prompt, "files": file_names}, is_user=True)
        
        # Add to session state
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "files": file_names
        })
        
        # Process and display assistant response
        with chat_container:
            message_placeholder = st.empty()
            
            # Display loading indicator
            loading_html = """
            <div class="loading-spinner">
                <div></div><div></div><div></div>
            </div>
            <span style="margin-left: 10px;">Thinking...</span>
            """
            message_placeholder.markdown(loading_html, unsafe_allow_html=True)
            
            # Process in background to not block UI
            response_future = concurrent.futures.ThreadPoolExecutor().submit(
                process_response, prompt, uploaded_files
            )
            
            # Wait for response
            try:
                response, thinking_steps = response_future.result(timeout=60)
                
                # Stream the response for better UX
                final_response = stream_response(message_placeholder, response, thinking_steps)
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_response
                })
                st.session_state.thinking_steps = thinking_steps
                
            except concurrent.futures.TimeoutError:
                message_placeholder.error("The request took too long to process. Please try a simpler question or try again later.")

if __name__ == "__main__":
    main()