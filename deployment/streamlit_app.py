"""
Streamlit chat interface for the multi-tool portfolio agent.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from agent_client import query_agent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Portfolio Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better typography and readability
st.markdown("""
<style>
    /* Increase base font size */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        font-size: 17px !important;
        line-height: 1.7 !important;
    }
    
    /* Header sizing - works in both light and dark mode */
    .stMarkdown h2 {
        font-size: 22px !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        color: inherit !important;
    }
    
    .stMarkdown h3 {
        font-size: 19px !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        color: inherit !important;
    }
    
    /* Improve bullet points - force proper list rendering */
    .stMarkdown ul, .stMarkdown ol {
        font-size: 17px !important;
        line-height: 1.7 !important;
        margin-left: 1.5rem !important;
        padding-left: 0.5rem !important;
    }
    
    .stMarkdown li {
        margin-bottom: 0.5rem !important;
        display: list-item !important;
    }
    
    /* Better link styling */
    .stMarkdown a {
        font-size: 17px !important;
        font-weight: 500 !important;
        text-decoration: underline !important;
    }
    
    /* Code and inline elements */
    .stMarkdown code {
        font-size: 16px !important;
    }
    
    /* Improve spacing for better readability */
    .stMarkdown p {
        margin-bottom: 1rem !important;
    }
    
    /* Chat message container improvements */
    .stChatMessage {
        font-size: 17px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Sidebar with enhanced content
with st.sidebar:
    st.title("🤖 Mohit's AI Assistant")
    st.markdown("---")
    
    st.markdown("""
    ### About This Chatbot
    I can help you explore Mohit Aggarwal's work in Data Science and Generative AI.
    
    **I have access to:**
    - 📚 **Technical Articles & Blog Posts**
    - 💻 **GitHub Projects**  
    - 🎓 **Portfolio Information** (Experience, Education, Skills)
    """)
    
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("- [**My Website**](https://mohitagr18.github.io)")
    st.markdown("- [**My Medium**](https://medium.com/@mohitagr18)")
    st.markdown("- [**My GitHub**](https://github.com/mohitagr18)")
    
    st.markdown("---")
    st.markdown("""
    ### Example Queries
    - *"Tell me about Mohit"*
    - *"What projects does Mohit have?"*
    - *"List my articles"*
    - *"Summarize the hackathon article"*
    - *"What are Mohit's technical skills?"*
    - *"Tell me about mcp_home_automation"*
    
    💡 **Tip:** Use underscores for GitHub project names
    """)
    
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# Main title
st.title("🤖 Mohit's Portfolio Chatbot")
st.markdown("**Built with Google ADK • Vertex AI Agent Engine • Streamlit on Cloud Run**")
st.caption("📊 Data sources: Vertex AI RAG Engine • Medium RSS Feed • GitHub API")


# Welcome message when chat is empty
if not st.session_state.messages:
    st.info("👋 Hi! I'm Mohit's AI Portfolio Assistant. Ask me about projects, articles, skills, or experience!")

# Always show example queries in an expander (whether chat is empty or not)
with st.expander("💡 Example Queries", expanded=(not st.session_state.messages)):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **About Mohit:**
        - Tell me about Mohit
        - What are Mohit's technical skills?
        - Where did Mohit go to school?
        """)
        
    with col2:
        st.markdown("""
        **Projects & Articles:**
        - What projects does Mohit have?
        - List my articles
        - Summarize the hackathon article
        """)

# Display chat messages 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display citations if present
        if message.get("citations"):
            with st.expander("📚 Sources"):
                for i, cite in enumerate(message["citations"], 1):
                    st.markdown(f"**[{i}]** {cite['source_uri']}")
                    if cite.get('text'):
                        st.caption(f"_{cite['text']}_")

# Chat input 
if prompt := st.chat_input("Ask me anything about projects, articles, skills, or experience..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get resource name from env
                resource_name = os.environ.get("AGENT_RESOURCE_NAME")
                if not resource_name:
                    st.error("AGENT_RESOURCE_NAME not set in environment")
                    st.stop()
                
                # Query the agent
                result = query_agent(
                    resource_name=resource_name,
                    message=prompt,
                    session_id=st.session_state.session_id
                )
                
                # Update session_id for continuity
                st.session_state.session_id = result.get("session_id")
                
                # Parse response from events using correct structure
                response_text = ""
                citations = []
                events = result.get("events", [])
                
                # Process events to extract final text and contexts
                for event in events:
                    if isinstance(event, dict) and "content" in event:
                        content = event["content"]
                        
                        # Extract final model text response
                        if content.get("role") == "model" and "parts" in content:
                            for part in content["parts"]:
                                if "text" in part:
                                    response_text = part["text"]  # Use latest text
                        
                        # Extract citations from function_response parts
                        if content.get("role") == "user" and "parts" in content:
                            for part in content["parts"]:
                                if "function_response" in part:
                                    fr = part["function_response"]
                                    if "response" in fr and isinstance(fr["response"], dict):
                                        contexts = fr["response"].get("contexts", [])
                                        for ctx in contexts:
                                            if isinstance(ctx, dict) and "text" in ctx:
                                                citations.append({
                                                    "text": ctx.get("text", "")[:200],
                                                    "source_uri": ctx.get("source_uri", "RAG Corpus"),
                                                    "distance": ctx.get("distance", "")
                                                })
                
                # Fallback to result.response if no text extracted
                if not response_text:
                    response_text = result.get("response", "No response received")
                
                # Display response
                st.markdown(response_text)
                
                # Display citations if present
                if citations:
                    with st.expander(f"📚 Sources ({len(citations)} passages)"):
                        for i, cite in enumerate(citations, 1):
                            st.markdown(f"**Source {i}**")
                            st.caption(f"📄 _{cite['text'][:150]}..._")
                            if cite.get('distance'):
                                st.caption(f"Relevance: {1 - float(cite['distance']):.2%}")
                            st.markdown("---")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "citations": citations
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
