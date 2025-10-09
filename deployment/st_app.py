"""
Streamlit chat interface for the multi-tool portfolio agent.
- 6 prompt buttons in 2-column layout (3 per column)
- Theme-adaptive button styling (works in light and dark mode)
- Real-time tool call visibility with st.status (fixed for actual real-time updates)
- Differentiated sidebar example queries
"""

import os
import streamlit as st
from dotenv import load_dotenv
from agent_client import query_agent

load_dotenv()

# ---------- Page config ----------
st.set_page_config(page_title="Portfolio Chatbot", page_icon="🤖", layout="wide")

# ---------- CSS ----------
st.markdown(
    """
    <style>
      /* Constrain chat content width */
      .main > div { max-width: 1100px; margin-left: auto; margin-right: auto; }
      
      /* Quick link buttons */
      .link-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
      .link-btn {
        border: 1px solid rgba(0,0,0,0.15);
        border-radius: 8px;
        padding: 6px 12px;
        text-decoration: none;
        font-size: 0.9rem;
        background: white;
        color: #1a73e8;
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }
      .link-btn:hover { background: #f6f9ff; }
      .link-btn:focus { outline: 3px solid #9ec5ff; outline-offset: 2px; }
      
      /* Styled prompt buttons with adaptive colors */
      div[data-testid="column"] button[kind="secondary"] {
        background: rgba(59, 130, 246, 0.12) !important;
        color: rgba(30, 58, 138, 1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        height: 70px !important;
        white-space: normal !important;
        text-align: left !important;
        line-height: 1.3 !important;
      }
      div[data-testid="column"] button[kind="secondary"]:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15) !important;
      }
      
      /* Dark mode adjustments */
      @media (prefers-color-scheme: dark) {
        div[data-testid="column"] button[kind="secondary"] {
          background: rgba(96, 165, 250, 0.15) !important;
          color: rgba(191, 219, 254, 1) !important;
          border-color: rgba(96, 165, 250, 0.4) !important;
        }
        div[data-testid="column"] button[kind="secondary"]:hover {
          background: rgba(96, 165, 250, 0.25) !important;
          border-color: rgba(96, 165, 250, 0.6) !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "is_sending" not in st.session_state:
    st.session_state.is_sending = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 Mohit's AI Assistant")
    st.markdown("---")
    st.markdown(
        """
        ### About This Chatbot
        Explore Mohit Aggarwal's work in Data Science and Generative AI.

        **I have access to:**
        - 📚 Technical Articles & Blog Posts
        - 💻 GitHub Projects
        - 🎓 Portfolio Information (Experience, Education, Skills)
        """
    )

    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown(
        """
        <div class="link-row">
          <a class="link-btn" href="https://mohitagr18.github.io" target="_blank">🌐 Website</a>
          <a class="link-btn" href="https://medium.com/@mohitagr18" target="_blank">✍️ Medium</a>
          <a class="link-btn" href="https://github.com/mohitagr18" target="_blank">💻 GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        ### Example Queries
        - Where did Mohit go to school?
        - Summarize Multimodal Style Coach article
        - Tell me about mcp_home_automation
        - Show me Mohit's GitHub repositories
        - Summarize the travel_itinerary project
        - What technologies does Mohit work with?

        💡 **Tip:** Use underscores for GitHub project names
        """
    )

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# ---------- Header ----------
st.title("🤖 Mohit's Portfolio Chatbot")
st.markdown("Built with Google ADK • Vertex AI Agent Engine • Streamlit on Cloud Run")
st.caption("📊 Data sources: Vertex AI RAG Engine • Medium RSS Feed • GitHub API")

# ---------- Intro message ----------
if not st.session_state.messages:
    st.info("👋 Hi! Ask about projects, articles, skills, or experience!")

# ---------- Prompt buttons: 2 columns, 3 buttons each ----------
st.markdown("**Quick prompts:**")

col1, col2 = st.columns(2)

# Left column: About Mohit queries
with col1:
    if st.button("Tell me about Mohit", key="chip_1", type="secondary", use_container_width=True):
        st.session_state.pending_prompt = "Tell me about Mohit"
    if st.button("What are Mohit's technical skills?", key="chip_2", type="secondary", use_container_width=True):
        st.session_state.pending_prompt = "What are Mohit's technical skills?"
    if st.button("Where did Mohit go to school?", key="chip_3", type="secondary", use_container_width=True):
        st.session_state.pending_prompt = "Where did Mohit go to school?"

# Right column: Projects & Content queries
with col2:
    if st.button("What projects does Mohit have?", key="chip_4", type="secondary", use_container_width=True):
        st.session_state.pending_prompt = "What projects does Mohit have?"
    if st.button("List my articles", key="chip_5", type="secondary", use_container_width=True):
        st.session_state.pending_prompt = "List my articles"
    if st.button("Summarize the hackathon article", key="chip_6", type="secondary", use_container_width=True):
        st.session_state.pending_prompt = "Summarize the hackathon article"

st.markdown("---")

# ---------- Display history ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("📚 Sources"):
                for i, cite in enumerate(message["citations"], 1):
                    st.markdown(f"**[{i}]** {cite.get('source_uri','RAG Corpus')}")
                    if cite.get("text"):
                        st.caption(f"_{cite['text']}_")

# ---------- Unified send handler with ACTUAL real-time tool visibility ----------
def send_prompt(prompt: str):
    if not prompt or st.session_state.is_sending:
        return
    st.session_state.is_sending = True

    # Append and echo user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant with real-time tool call visibility
    with st.chat_message("assistant"):
        try:
            resource_name = os.environ.get("AGENT_RESOURCE_NAME")
            if not resource_name:
                st.markdown(":warning: AGENT_RESOURCE_NAME not set")
                st.session_state.is_sending = False
                return

            # Tool call tracking
            tool_icons = {
                "rag_retrieval": "📚",
                "list_repositories": "🔍",
                "get_repository_file": "📄",
                "list_medium_articles": "✍️",
            }

            # Create status widget for real-time updates
            with st.status("🤖 Processing your query...", expanded=True) as status:
                # Show initial thinking message
                thinking_placeholder = st.empty()
                thinking_placeholder.write("_Analyzing query and planning tool calls..._")
                
                # Call the agent (this is where the actual work happens)
                result = query_agent(
                    resource_name=resource_name,
                    message=prompt,
                    session_id=st.session_state.session_id,
                )

                # Clear the thinking message
                thinking_placeholder.empty()
                
                st.session_state.session_id = result.get("session_id")
                events = result.get("events", [])

                # Parse events for tool calls and responses
                tool_count = 0
                for event in events:
                    if isinstance(event, dict) and "content" in event:
                        content = event["content"]
                        
                        # Check for function_call (tool invocation)
                        if "parts" in content:
                            for part in content["parts"]:
                                if "function_call" in part:
                                    fc = part["function_call"]
                                    tool_name = fc.get("name", "unknown")
                                    icon = tool_icons.get(tool_name, "🔧")
                                    tool_count += 1
                                    st.write(f"{icon} **Called:** `{tool_name}`")
                                
                                # Check for function_response (tool completion)
                                if "function_response" in part:
                                    fr = part["function_response"]
                                    tool_name = fr.get("name", "unknown")
                                    icon = tool_icons.get(tool_name, "✅")
                                    
                                    # Extract result summary
                                    response_data = fr.get("response", {})
                                    if isinstance(response_data, dict):
                                        contexts = response_data.get("contexts", [])
                                        if contexts:
                                            st.write(f"  ↳ Retrieved **{len(contexts)}** results")
                                        else:
                                            st.write(f"  ↳ Completed")
                                    else:
                                        st.write(f"  ↳ Completed")

                # Update status with better message
                if tool_count > 0:
                    status.update(
                        label=f"✅ Used {tool_count} tool{'s' if tool_count != 1 else ''} — expand to see details", 
                        state="complete", 
                        expanded=False
                    )
                else:
                    status.update(
                        label="✅ Answered directly without tools", 
                        state="complete", 
                        expanded=False
                    )

            # Extract final response text and citations
            response_text = ""
            citations = []

            for event in events:
                if isinstance(event, dict) and "content" in event:
                    content = event["content"]
                    # Final model text
                    if content.get("role") == "model" and "parts" in content:
                        for part in content["parts"]:
                            if "text" in part:
                                response_text = part["text"]
                    # Citations from RAG
                    if content.get("role") == "user" and "parts" in content:
                        for part in content["parts"]:
                            if "function_response" in part:
                                fr = part["function_response"]
                                if "response" in fr and isinstance(fr["response"], dict):
                                    contexts = fr["response"].get("contexts", [])
                                    for ctx in contexts:
                                        if isinstance(ctx, dict) and "text" in ctx:
                                            citations.append(
                                                {
                                                    "text": ctx.get("text", "")[:200],
                                                    "source_uri": ctx.get("source_uri", "RAG Corpus"),
                                                    "distance": ctx.get("distance", ""),
                                                }
                                            )

            if not response_text:
                response_text = result.get("response", "No response received")

            # Display final answer
            st.markdown(response_text)

            if citations:
                with st.expander(f"📚 Sources ({len(citations)})"):
                    for i, cite in enumerate(citations, 1):
                        st.markdown(f"**Source {i}**")
                        if cite.get("text"):
                            st.caption(f"📄 _{cite['text'][:150]}..._")
                        if cite.get("distance"):
                            try:
                                st.caption(f"Relevance: {1 - float(cite['distance']):.2%}")
                            except Exception:
                                pass
                        st.markdown("---")

            # Persist assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text, "citations": citations}
            )

        except Exception as e:
            st.markdown(f":x: Error: {str(e)}")
        finally:
            st.session_state.is_sending = False

# ---------- Input orchestration ----------
pending = st.session_state.pending_prompt
if pending:
    st.session_state.pending_prompt = None
    send_prompt(pending)

prompt = st.chat_input("Ask about projects, articles, skills, or experience…")
if prompt:
    send_prompt(prompt)
