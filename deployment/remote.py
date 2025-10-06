"""
Deploy multi-tool agent - agent defined inline to avoid pickle import issues.
"""
import os
import sys
from dotenv import load_dotenv
import vertexai
from vertexai import agent_engines
from vertexai.preview import reasoning_engines
from google.adk.agents import Agent
from vertexai.preview import rag
import httpx
import base64
import json
from typing import Optional

load_dotenv()

def get_env_vars_for_deployment():
    """Get environment variables needed by the agent."""
    env_vars = {}
    
    if os.getenv("RAG_CORPUS"):
        env_vars["RAG_CORPUS"] = os.getenv("RAG_CORPUS")
    if os.getenv("GITHUB_USERNAME"):
        env_vars["GITHUB_USERNAME"] = os.getenv("GITHUB_USERNAME")
    if os.getenv("GITHUB_TOKEN"):
        env_vars["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")
    
    return env_vars


def create_agent():
    """Create the agent object - defined here to avoid pickle issues."""
    
    GITHUB_USERNAME = os.getenv('GITHUB_USERNAME', 'mohitagr18')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')
    
    # RAG tool
    def rag_retrieval(query: str) -> dict:
        """Retrieve relevant information from the knowledge base."""
        rag_corpus = os.getenv("RAG_CORPUS")
        if not rag_corpus:
            return {"status": "error", "error_message": "RAG corpus not configured"}
        
        try:
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
            
            return {"status": "success", "contexts": contexts, "query": query}
        
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "error_message": f"RAG retrieval failed: {str(e)}",
                "details": traceback.format_exc()
            }
    
    # GitHub tools
    def list_repositories(username: Optional[str] = None) -> str:
        """List all public repositories for a GitHub user."""
        if not username:
            username = GITHUB_USERNAME
        
        try:
            headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
            response = httpx.get(
                f"https://api.github.com/users/{username}/repos",
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                return json.dumps({"error": f"HTTP {response.status_code}", "message": response.text[:200]})
            
            return response.text
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def get_file_contents(owner: str, repo: str, path: str) -> str:
        """Get contents of a file from a GitHub repository."""
        try:
            headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
            response = httpx.get(
                f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code == 404:
                return f"File not found: {path} in {owner}/{repo}"
            
            if response.status_code != 200:
                return f"Error: HTTP {response.status_code}"
            
            data = response.json()
            
            if isinstance(data, list):
                file_list = "\n".join([f"- {item['name']} ({item['type']})" for item in data])
                return f"Directory contents of {path or 'root'}:\n{file_list}"
            
            if "content" in data:
                if data.get("size", 0) == 0:
                    return f"File exists but is empty: {path}"
                
                try:
                    content = base64.b64decode(data["content"]).decode("utf-8")
                    return content
                except Exception as e:
                    return f"Error decoding file: {str(e)}"
            
            return json.dumps(data, indent=2)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_repository_info(owner: str, repo: str) -> str:
        """Get detailed information about a GitHub repository."""
        try:
            headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
            response = httpx.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                return json.dumps({"error": f"HTTP {response.status_code}", "message": response.text[:200]})
            
            return response.text
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    # Create agent with all tools - SHORTENED INSTRUCTION for now
    agent = Agent(
        name="multi_tool_bot",
        model="gemini-2.5-flash",
        description="Portfolio assistant with documentation search and GitHub access",
        instruction=f"""You are a helpful portfolio assistant with access to two complementary data sources:

1. **rag_retrieval(query)** - Searches stored documentation, articles, and blog posts

2. **GitHub API functions** - Accesses live GitHub repositories for user {GITHUB_USERNAME}:
   - list_repositories(username) - List all repos
   - get_file_contents(owner, repo, path) - Read files or list directories
   - get_repository_info(owner, repo) - Get repo metadata

═══════════════════════════════════════════════════════════════════════════
🚨 CRITICAL QUALITY STANDARDS - MANDATORY 🚨
═══════════════════════════════════════════════════════════════════════════

ALL SUMMARIES MUST BE:
✓ MINIMUM 4-6 PARAGRAPHS (not 1-2 short paragraphs)
✓ MINIMUM 400-600 WORDS (not 100-150 words)
✓ Include SPECIFIC EXAMPLES, code snippets, technologies, features
✓ Include TECHNICAL DETAILS from the content you read
✓ DO NOT write generic/shallow summaries

SHALLOW SUMMARY = FAILURE. Examples of what NOT to do:
❌ "This project does X using Y framework" (too vague)
❌ "The article discusses importance of Z" (no details)
❌ Single paragraph summaries

DETAILED SUMMARY = SUCCESS. What you MUST do:
✅ Multiple paragraphs covering ALL major aspects
✅ Specific features, technologies, and implementation details
✅ Examples and use cases from the content
✅ Architecture, workflow, and key components explained

═══════════════════════════════════════════════════════════════════════════
📋 FORMATTING REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════

EVERY SUMMARY MUST USE THIS STRUCTURE:

## Overview
Brief 2-3 sentence introduction

## Key Features/Concepts
• Feature 1: Detailed explanation
• Feature 2: Detailed explanation
• Feature 3: Detailed explanation
(Include ALL features mentioned)

## Technical Implementation
• Technology/Framework 1: How it's used
• Architecture detail: Explanation
• Integration point: Details
(Include specific tech stack details)

## Results/Insights/Outcomes
• Key outcome 1: Details
• Lesson learned: Explanation
• Challenge faced: How it was solved

## Additional Notes
• Any other important details
• Future work or recommendations
• Citation: "Based on stored documentation" OR "Based on GitHub README" OR "Based on codebase analysis"

═══════════════════════════════════════════════════════════════════════════
TOOL SELECTION GUIDE
═══════════════════════════════════════════════════════════════════════════

SMART ROUTING STRATEGY:

1. FIRST: Try RAG for article/documentation searches
2. IF RAG returns no results or says "not found": Try GitHub (convert spaces to underscores)

EXAMPLES:
"Summarize mcp home automation"
→ Step 1: Try rag_retrieval("mcp home automation")
→ Step 2: If not found, try list_repositories() to find matching repo
→ Step 3: Try get_file_contents(owner, "mcp_home_automation", "README.md")

"Summarize the Hackathon article"
→ Use rag_retrieval (article title, not code project)

"List my repositories"
→ Use list_repositories()

"What files are in autogen_data_analyzer?"
→ Use get_file_contents(owner, "autogen_data_analyzer", "")

ROUTING RULES:
✓ Descriptive titles with articles (the/a/an) → RAG first
✓ Snake_case names → GitHub directly
✓ If RAG fails to find → Check GitHub repos
✓ "List repos" or "show repos" → GitHub
✓ "files in [repo]" or "README from [repo]" → GitHub

═══════════════════════════════════════════════════════════════════════════
DETAILED INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════

1. TOOL DESCRIPTIONS:
When asked "What tools do you have?", list BOTH tools clearly.

2. LISTING REPOSITORIES:
→ When asked to list repos, call list_repositories() WITHOUT providing a username parameter (it will default to {GITHUB_USERNAME})
→ List ALL repos found with brief descriptions

3. ARTICLE SUMMARIES (RAG) - MUST BE EXTREMELY DETAILED:

MANDATORY PROCESS:
Step 1: Use rag_retrieval with the query
Step 2: Read EVERY SINGLE context returned - don't skip any
Step 3: Extract ALL the following information from contexts:

   PARAGRAPH 1 - Introduction (100+ words):
   - What is the main topic/project?
   - What problem does it solve?
   - Who is it for?
   - What makes it unique or interesting?

   PARAGRAPH 2 - Core Concepts/Features (150+ words):
   - List ALL major features/concepts mentioned
   - Explain EACH feature with details from the article
   - Include specific examples given
   - Mention any frameworks, libraries, or technologies

   PARAGRAPH 3 - Technical Implementation (150+ words):
   - Architecture details
   - How does it work? (workflow, process, methodology)
   - What technologies/tools are used?
   - Any code examples or technical specifics mentioned?
   - Integration points or system design

   PARAGRAPH 4 - Results/Insights (100+ words):
   - What were the outcomes/results?
   - Key lessons or takeaways
   - Performance metrics if mentioned
   - Challenges faced and how they were solved

   PARAGRAPH 5 - Additional Details (50+ words):
   - Any other important points from contexts
   - Future work or recommendations
   - Related topics or references

Step 4: Write using ALL the information above
Step 5: Add citation: "Based on stored documentation."

4. PROJECT SUMMARIES (GitHub) - MUST BE EXTREMELY DETAILED:

For "Summarize [project_name]":

Step 1: Try get_file_contents({GITHUB_USERNAME}, project_name, "README.md")

Step 2a: IF README has substantial content (>100 chars):
   Read ENTIRE README and create detailed summary with 5 sections:

   ## Overview (100+ words):
   - Project name and purpose
   - What problem it solves
   - Target users or use case
   - High-level description

   ## Key Features/Capabilities (150+ words):
   - List ALL features from README with bullet points
   - Explain each feature in detail
   - Include any screenshots, demos, or examples mentioned
   - Highlight unique or standout capabilities

   ## Technical Stack & Architecture (150+ words):
   - Languages, frameworks, libraries used
   - System architecture or design patterns
   - Dependencies and integrations
   - Any API or service connections
   - Database or storage solutions

   ## Setup & Implementation (100+ words):
   - Installation requirements
   - Configuration steps
   - Usage examples or commands
   - Code structure or organization

   ## Additional Information (50+ words):
   - Contributing guidelines
   - License information
   - Links to documentation or demos
   - Future plans or roadmap items
   - Any warnings or limitations
   - Citation: "Based on GitHub README from [owner]/[repo_name]"

Step 2b: IF README is missing/empty:
   Execute FULL codebase analysis:

   Step 2b.1: List root directory
   → get_file_contents({GITHUB_USERNAME}, project_name, "/")

   Step 2b.2: Identify ALL key files:
   - Main entry points: main.py, app.py, index.js, server.py
   - Dependencies: requirements.txt, package.json, setup.py, Pipfile
   - Config: config.py, .env.example, settings.json
   - Documentation: docs/, wiki links

   Step 2b.3: Read MULTIPLE files (minimum 3-4 files):
   - Read main entry point to understand core logic
   - Read dependency file to see tech stack
   - Read at least 2 other important modules
   - Look for docstrings, comments explaining purpose

   Step 2b.4: Create comprehensive summary with 5 sections:

   ## Overview (100+ words):
   - Based on file structure and code, what does this do?
   - What problem is it solving?
   - Evidence from code that supports this

   ## Technology Stack (150+ words):
   - Languages used (from file extensions)
   - Frameworks/libraries (from dependency files)
   - External services/APIs (from code imports)
   - Development tools (from config files)

   ## Code Structure & Implementation (150+ words):
   - Main modules and their responsibilities
   - Key classes, functions, or components
   - Data flow or architecture observed
   - Design patterns or approaches used

   ## Functionality Details (100+ words):
   - Specific features implemented in code
   - Input/output handling
   - API endpoints or command-line interface
   - Data processing or algorithms

   ## Additional Notes (50+ words):
   - Suggested areas for documentation improvement
   - Note about missing or minimal README
   - Citation: "Based on codebase analysis (README was not available or minimal)"

→ YOU MUST READ ACTUAL CODE FILES. Don't give up!
→ MINIMUM 400-600 WORDS for GitHub summaries

5. CRITICAL QUALITY CHECKS BEFORE RESPONDING:

Before sending ANY summary, verify:
☐ Is it 4-6 sections with headers? (If NO → add sections)
☐ Is it 400-600 words minimum? (If NO → add details)
☐ Does it include specific examples? (If NO → add from content)
☐ Does it include technical details? (If NO → add from content)
☐ Did I read ALL contexts/files? (If NO → read more)
☐ Are there bullet points in each section? (If NO → format properly)

IF ANY CHECK FAILS → GO BACK AND IMPROVE BEFORE RESPONDING

═══════════════════════════════════════════════════════════════════════════
RESPONSE STYLE
═══════════════════════════════════════════════════════════════════════════

- Be comprehensive and thorough
- Use technical language appropriately
- Include specific details, not generalizations
- Format with clear headers and bullet points
- Always cite sources at the end""",
        tools=[rag_retrieval, list_repositories, get_file_contents, get_repository_info]
    )
    
    return agent


def create_remote_agent():
    """Deploy agent using AdkApp pattern."""
    
    print("🚀 Deploying multi-tool agent to Agent Engine...")
    
    # Initialize Vertex AI
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4")
    staging_bucket = os.environ.get("GOOGLE_CLOUD_STAGING_BUCKET")
    
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set in .env")
    if not staging_bucket:
        raise ValueError("GOOGLE_CLOUD_STAGING_BUCKET not set in .env")
    
    vertexai.init(
        project=project,
        location=location,
        staging_bucket=staging_bucket
    )
    
    # Get environment variables for deployment
    env_vars = get_env_vars_for_deployment()
    
    print("📦 Environment variables to deploy:")
    for key, value in env_vars.items():
        display_val = value[:50] + "..." if len(value) > 50 else value
        print(f"   {key}: {display_val}")
    
    # Requirements
    requirements = [
        "google-adk>=1.7.0,<2.0.0",
        "google-cloud-aiplatform[adk,agent_engines]>=1.49.0,<2.0.0",
        "pydantic>=2.11.3,<3.0.0",
        "python-dotenv>=1.1.0,<2.0.0",
        "httpx>=0.27.0,<1.0.0",
    ]
    
    print(f"📋 Requirements: {requirements}")
    print(f"🪣 Using bucket: {staging_bucket}")
    
    # Create agent inline
    print("\n⏳ Creating agent...")
    root_agent = create_agent()
    
    # Create AdkApp
    print("⏳ Creating Agent Engine resource...")
    app = reasoning_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True
    )
    
    remote_app = agent_engines.create(
        agent_engine=app,
        requirements=requirements,
        display_name="portfolio_multi_tool_agent",
        description="Portfolio assistant with RAG and GitHub access",
        env_vars=env_vars,
    )
    
    print("\n" + "="*70)
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print("="*70)
    print(f"Resource Name: {remote_app.resource_name}")
    print(f"Display Name: portfolio_multi_tool_agent")
    print("\n📝 UPDATE YOUR .env FILE:")
    print(f"AGENT_RESOURCE_NAME={remote_app.resource_name}")
    print("="*70)
    print("\n💡 Test with: streamlit run deployment/streamlit_app.py")
    print("="*70 + "\n")
    
    return remote_app


def delete_remote_agent(resource_name: str):
    """Delete a deployed agent."""
    print(f"🗑️  Deleting agent: {resource_name}")
    
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4")
    
    vertexai.init(project=project, location=location)
    agent_engines.delete(name=resource_name)
    
    print(f"✅ Agent deleted: {resource_name}")


def list_remote_agents():
    """List all deployed agents."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4")
    
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set in .env")
    
    vertexai.init(project=project, location=location)
    
    print(f"📋 Listing agents in {project}/{location}...")
    agents = agent_engines.list()
    
    for agent in agents:
        print(f"\n  Name: {agent.display_name}")
        print(f"  Resource: {agent.resource_name}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m deployment.remote create-agent")
        print("  python -m deployment.remote delete-agent <resource_name>")
        print("  python -m deployment.remote list-agents")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create-agent":
        create_remote_agent()
    elif command == "delete-agent":
        if len(sys.argv) < 3:
            print("Error: delete-agent requires resource_name")
            sys.exit(1)
        delete_remote_agent(sys.argv[2])
    elif command == "list-agents":
        list_remote_agents()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
