"""
Agent client for querying the deployed Vertex AI Agent Engine resource.
"""
import os
from typing import Optional
import vertexai
from vertexai import agent_engines


def init_vertexai():
    """Initialize Vertex AI with project and location from env."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
    
    vertexai.init(project=project, location=location)
    return project, location


def get_agent(resource_name: str):
    """
    Get a reference to the deployed Agent Engine resource.
    
    Args:
        resource_name: Full resource name or ID like 
                      "projects/PROJECT/locations/REGION/reasoningEngines/ID"
                      or short form "reasoningEngines/ID"
    
    Returns:
        AgentEngine instance
    """
    init_vertexai()
    # Use positional argument, not keyword 'name='
    return agent_engines.get(resource_name)

def query_agent(
    resource_name: str,
    message: str,
    user_id: str = "streamlit_user",
    session_id: Optional[str] = None
) -> dict:
    """
    Query the deployed agent and return the response with citations.
    
    Args:
        resource_name: Agent Engine resource name
        message: User message/query
        user_id: User identifier for session tracking
        session_id: Optional session ID to continue conversation
    
    Returns:
        dict with keys:
            - response: str, final text response
            - citations: list of dicts with source_uri and text
            - events: list of all events for debugging
            - session_id: str for continuing conversation
    """
    agent = get_agent(resource_name)
    
    # If no session exists, create one
    if not session_id:
        session = agent.create_session(user_id=user_id)
        session_id = session.get("id") or session.get("name", "").split("/")[-1]
    
    # Collect all events from streaming response
    events = []
    
    try:
        for event in agent.stream_query(
            user_id=user_id,
            session_id=session_id,
            message=message
        ):
            # Store raw event
            events.append(event)
    except Exception as e:
        return {
            "response": f"Error querying agent: {str(e)}",
            "citations": [],
            "events": events,
            "session_id": session_id
        }
    
    # Parse events to extract text and citations
        # Parse events to extract text and citations
    response_text = ""
    citations = []
    
    for event in events:
        if isinstance(event, dict):
            # Check if event has 'content' key (ADK response format)
            if "content" in event:
                content = event["content"]
                
                # Extract text from model responses
                if content.get("role") == "model" and "parts" in content:
                    for part in content["parts"]:
                        if "text" in part:
                            response_text = part["text"]  # Keep latest text
                
                # Extract citations from function responses
                if content.get("role") == "user" and "parts" in content:
                    for part in content["parts"]:
                        if "function_response" in part:
                            fr = part["function_response"]
                            if "response" in fr and isinstance(fr["response"], dict):
                                contexts = fr["response"].get("contexts", [])
                                for ctx in contexts:
                                    if isinstance(ctx, dict):
                                        citations.append({
                                            "source_uri": ctx.get("source_uri", "RAG Corpus"),
                                            "text": ctx.get("text", "")[:200],
                                            "distance": ctx.get("distance", 0)
                                        })
    
    return {
        "response": response_text or "No response text found",
        "citations": citations,
        "events": events,
        "session_id": session_id
    }


# Example usage for testing
# Example usage for testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    import json
    load_dotenv()
    
    resource = os.environ.get("AGENT_RESOURCE_NAME")
    if not resource:
        print("Set AGENT_RESOURCE_NAME in .env to test")
        exit(1)
    
    print("Testing agent query...")
    result = query_agent(
        resource_name=resource,
        message="What can you help me with?"
    )
    
    # Debug: print raw events
    print("\n=== DEBUG: Raw Events ===")
    for i, event in enumerate(result["events"]):
        print(f"\nEvent {i}:")
        print(json.dumps(event, indent=2, default=str))
    
    print("\n=== Response ===")
    print(result["response"])
    
    if result["citations"]:
        print("\n=== Citations ===")
        for i, cite in enumerate(result["citations"], 1):
            print(f"[{i}] {cite['source_uri']}")
            print(f"    {cite['text']}...")
