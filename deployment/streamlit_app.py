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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Sidebar
with st.sidebar:
    st.title("🤖 Portfolio Assistant")
    st.markdown("""
    Ask me about:
    - Projects and portfolio
    - Medium articles
    - GitHub repositories
    - Skills and experience
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()
    
    st.markdown("---")
    st.caption("Powered by Vertex AI Agent Engine + RAG")

# Main title
st.title("Portfolio Chatbot")

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
if prompt := st.chat_input("Ask me anything about the portfolio..."):
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
                
                # Parse response - handle the dict format we saw in testing
                response_text = ""
                citations = result.get("citations", [])
                
                # Extract text from complex response format
                events = result.get("events", [])
                for event in events:
                    if isinstance(event, dict):
                        # Look for final model response
                        if event.get("role") == "model" and "parts" in event:
                            for part in event["parts"]:
                                if "text" in part:
                                    response_text += part["text"]
                        # Also extract from contexts if present
                        if "contexts" in event:
                            for ctx in event["contexts"]:
                                if "source_uri" in ctx and ctx not in citations:
                                    citations.append(ctx)
                
                # Fallback to result.response if no text extracted
                if not response_text:
                    response_text = result.get("response", "No response received")
                
                # Display response
                st.markdown(response_text)
                
                # Display citations
                if citations:
                    with st.expander("📚 Sources"):
                        for i, cite in enumerate(citations, 1):
                            st.markdown(f"**[{i}]** {cite.get('source_uri', 'Unknown')}")
                            if cite.get('text'):
                                st.caption(f"_{cite['text'][:200]}_")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "citations": citations
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
