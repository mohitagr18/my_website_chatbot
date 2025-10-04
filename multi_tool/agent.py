from google.adk.agents import Agent
import vertexai
from vertexai.preview import rag
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
import datetime
from zoneinfo import ZoneInfo
import os
from dotenv import load_dotenv

load_dotenv()


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def rag_retrieval(query: str) -> dict:
    """Retrieve relevant information from the knowledge base.
    
    Args:
        query: The search query to find relevant documents
        
    Returns:
        dict: Retrieved contexts and sources
    """
    rag_corpus = os.getenv("RAG_CORPUS")
    
    if not rag_corpus:
        return {
            "status": "error",
            "error_message": "RAG corpus not configured"
        }
    
    try:
        # Don't reinitialize vertexai - use existing initialization
        # vertexai should already be initialized by the reasoning engine
        
        # Use RAG API to retrieve contexts
        rag_resource = rag.RagResource(rag_corpus=rag_corpus)
        response = rag.retrieval_query(
            rag_resources=[rag_resource],
            text=query,
            similarity_top_k=5,
        )
        
        contexts = []
        for context in response.contexts.contexts:
            contexts.append({
                "text": context.text,
                "distance": context.distance if hasattr(context, 'distance') else None
            })
        
        return {
            "status": "success",
            "contexts": contexts,
            "query": query
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "status": "error",
            "error_message": f"RAG retrieval failed: {str(e)}",
            "details": error_details
        }


root_agent = Agent(
    name="multi_tool_bot",
    model='gemini-2.0-flash',
    description="A multi-tool bot that can use multiple tools to perform tasks",
    instruction="""You are a helpful assistant with access to multiple tools.

        AVAILABLE TOOLS:
        1. get_current_time(city) - Get current time for a city
        2. get_weather(city) - Get weather information for a city  
        3. rag_retrieval(query) - Search the knowledge base for information about documents, articles, projects, or any content in the corpus

        IMPORTANT GUIDELINES:
        - When asked about articles, documents, papers, projects, or any content that might be in a knowledge base, ALWAYS use the rag_retrieval tool first
        - When asked to summarize, explain, or provide information about specific content/documents, use rag_retrieval
        - Use get_current_time for time queries
        - Use get_weather for weather queries
        - When using rag_retrieval, cite the sources from the retrieved contexts

        Examples of when to use rag_retrieval:
        - "Summarize the article about X"
        - "What does the document say about Y?"
        - "Tell me about the project Z"
        - "What's in the knowledge base about A?"

        Always try to use the appropriate tool before saying you cannot help.""",
    tools=[get_current_time, get_weather, rag_retrieval]
)
