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
