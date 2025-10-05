from google.adk.agents import Agent
import vertexai
from vertexai.preview import rag
from google.adk.tools.mcp_tool import MCPToolSet, StdioConnectionParams
import os
from dotenv import load_dotenv

load_dotenv()

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

# GitHub MCP tool for live repository access
GITHUB_MCP = MCPToolSet(
    name="github_tools",
    connection_params=StdioConnectionParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN", "")}
    )
)

root_agent = Agent(
    name="multi_tool_bot",
    model='gemini-2.5-flash',
    description="A portfolio assistant with RAG search and GitHub repository access",
    instruction="""You are a helpful assistant for a portfolio chatbot with access to two main capabilities:

AVAILABLE TOOLS:
1. rag_retrieval(query) - Search the knowledge base for information about documents, articles, projects, or any content in the corpus
2. github_tools - Access GitHub repositories to list files, read code, check commits, issues, and pull requests

IMPORTANT GUIDELINES:
- When asked about articles, documents, papers, projects, or general portfolio content, use rag_retrieval
- When asked to summarize, explain, or provide information about specific written content, use rag_retrieval
- When asked about specific repository structure, files, or code, use GitHub tools
- When asked about recent commits, issues, or pull requests, use GitHub tools
- Always cite sources when using rag_retrieval
- Specify which repository you're querying when using GitHub tools

Examples of when to use rag_retrieval:
- "Summarize the style coach article"
- "What projects are in the portfolio?"
- "Tell me about the multi-agent hackathon project"

Examples of when to use GitHub tools:
- "What files are in the multi_tool_bot repository?"
- "Show me the README from repository X"
- "List all Python files in the src directory"
- "What are the recent commits in repository Y?"

Always try to use the appropriate tool before saying you cannot help.""",
    tools=[rag_retrieval, GITHUB_MCP]
)
