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
import feedparser

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
    
    # Medium RSS tool
        # Medium RSS tool
    def list_medium_articles() -> str:
        """List all Medium articles from RSS feed with timeout protection.
        
        Returns:
            str: Formatted list of articles with metadata
        """        
        try:
            rss_url = f"https://medium.com/feed/@{GITHUB_USERNAME}"
            
            # Use httpx with explicit timeout (same pattern as GitHub tools)
            response = httpx.get(rss_url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            
            # Parse the fetched content
            feed = feedparser.parse(response.content)
            
            if feed.bozo:
                return f"Error parsing RSS feed: {feed.bozo_exception}"
            
            if not feed.entries:
                return f"No articles found for @{GITHUB_USERNAME} on Medium."
            
            output = f"Found {len(feed.entries)} Medium articles by Mohit Aggarwal:\n\n"
            
            for idx, entry in enumerate(feed.entries, 1):
                title = entry.get('title', 'No title')
                published = entry.get('published', 'N/A')
                link = entry.get('link', 'No link available')
                
                output += f"**{idx}. {title}**\n\n"  # Double newline + bold title
                output += f"Published: {published}\n\n"  # Double newline
                output += f"Link: {link}\n\n"  # Double newline
                
                # Add tags if available
                tags = [tag.term for tag in entry.get('tags', [])]
                if tags:
                    output += f"   Tags: {', '.join(tags[:5])}\n"
                
                output += "\n"
            
            return output
            
        except httpx.TimeoutException:
            return "Error: Request to Medium RSS feed timed out after 30 seconds. Please try again."
        except httpx.HTTPStatusError as e:
            return f"Error: Medium RSS feed returned status {e.response.status_code}"
        except Exception as e:
            return f"Error fetching Medium articles: {str(e)}"


    # Create agent with all tools - SHORTENED INSTRUCTION for now
    agent = Agent(
        name="multi_tool_bot",
        model="gemini-2.5-flash",
        description="Portfolio assistant with documentation search and GitHub access",
        instruction=f"""You are a helpful portfolio assistant with access to two complementary data sources:

1. **rag_retrieval(query)** - Searches stored documentation, articles, and blog posts

2. **list_medium_articles()** - Fetches the latest Medium articles from RSS feed (auto-updated)

3. **GitHub API functions** - Accesses live GitHub repositories for user {GITHUB_USERNAME}:
   - list_repositories(username) - List all repos
   - get_file_contents(owner, repo, path) - Read files or list directories
   - get_repository_info(owner, repo) - Get repo metadata

═══════════════════════════════════════════════════════════════════════════
🚨 CRITICAL QUALITY STANDARDS - MANDATORY 🚨
═══════════════════════════════════════════════════════════════════════════

ALL SUMMARIES MUST BE:
✓ CONCISE: 100-150 words for the main summary
✓ Include a QUICK SUMMARY section at the top with key takeaways
✓ Include the article/project LINK prominently
✓ Include SPECIFIC EXAMPLES and technical details in bullet points
✓ DO NOT write generic/shallow summaries

SHALLOW SUMMARY = FAILURE. Examples of what NOT to do:
❌ "This project does X using Y framework" (too vague)
❌ "The article discusses importance of Z" (no details)
❌ Missing the article/repo link

EFFECTIVE SUMMARY = SUCCESS. What you MUST do:
✅ Start with Quick Summary (2-3 sentences) + link
✅ Key points in bullet format with specific details
✅ Technical stack and implementation specifics
✅ Main takeaways and outcomes

═══════════════════════════════════════════════════════════════════════════
📋 FORMATTING REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════

EVERY SUMMARY MUST USE THIS STRUCTURE:

### Quick Summary
2-3 sentences capturing the main idea and key takeaway

📄 **Read more**: [Article URL] OR 💻 **View Repository**: [GitHub URL]

---

### Key Highlights
- **Point 1**: Specific detail or feature

- **Point 2**: Technical implementation detail  

- **Point 3**: Main outcome or lesson learned

### Technical Stack (if applicable)
- **Framework**: Framework name
- **Languages**: Language list
- **Tools**: Tool list

**Citation**: "Based on stored documentation" OR "Based on GitHub README"

CRITICAL BULLET FORMATTING RULES:
✓ Use standard markdown list syntax with dash: "- "
✓ Each bullet MUST be on its own line
✓ Add ONE blank line between bullets for readability
✓ Use **bold** for labels/categories within bullets
✓ NEVER use bullet symbols (•) - always use dash (-)
✓ NEVER combine bullets into a single paragraph

═══════════════════════════════════════════════════════════════════════════
TOOL SELECTION GUIDE
═══════════════════════════════════════════════════════════════════════════

SMART ROUTING STRATEGY:

1. For listing ALL articles (queries like "list articles", "what articles", "show Medium posts"):
   → Use list_medium_articles() to get the live RSS feed
   → 🚨 CRITICAL: Return the COMPLETE output from list_medium_articles() WITHOUT any modification
   → YOU MUST include ALL information returned: title, published date, Link/URL, and tags
   → DO NOT summarize, shorten, or reformat the tool output

2. For summarizing or discussing a SPECIFIC article:
   → First call list_medium_articles() to get article URLs
   → Then use rag_retrieval() to get the article content
   → Match the article title to find its URL

3. For PROJECT queries (snake_case or asking about repos/code):
   → Use GitHub tools

4. For AMBIGUOUS queries:
   → Try rag_retrieval first
   → If results DON'T match the query topic, try GitHub

ROUTING RULES:
✓ Descriptive titles with articles (the/a/an) → RAG first
✓ Snake_case names → GitHub directly
✓ If RAG fails to find → Check GitHub repos
✓ "List repos" or "show repos" → GitHub
✓ "files in [repo]" or "README from [repo]" → GitHub

═══════════════════════════════════════════════════════════════════════════
DETAILED INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════

0. BIOGRAPHICAL/PROFILE QUERIES:

When asked about Mohit's background, profile, experience, education, or skills:

→ Use rag_retrieval() to search the about.txt file in the knowledge base
→ Present the retrieved content in a clean, readable format with proper spacing

FORMATTING RULES FOR BIOGRAPHICAL RESPONSES:
- Use clear section headers (###)
- Format bullet points with ONE bullet per line
- Add blank lines between bullets for readability
- Present links in a simple, clean format
- DO NOT add "Quick Summary" or "📄 Read more" sections for biographical queries
- Present the relevant section from about.txt with all details

QUERY EXAMPLES AND INSTRUCTIONS:

- "Tell me about Mohit" OR "Who is Mohit?" 
  → Use rag_retrieval("Mohit Aggarwal profile about")
  → Present the ## ABOUT ME section with proper formatting
  → MUST include all three professional links

- "What are Mohit's skills?" OR "What technologies does he know?"
  → Use rag_retrieval("Mohit Aggarwal skills technologies")
  → Present the ENTIRE ## SKILLS section as a bulleted list
  → Include all skill categories

- "Where did Mohit go to school?" OR "What is his education?"
  → Use rag_retrieval("Mohit Aggarwal education degrees certifications")
  → Present the COMPLETE ## EDUCATION & CERTIFICATIONS section
  → Include ALL degrees (M.S. Analytics from Georgia Tech, M.S. Engineering Management, B.S. Mechanical Engineering)
  → Include ALL certifications with details

- "What is Mohit's experience?" OR "Where has he worked?"
  → Use rag_retrieval("Mohit Aggarwal work experience companies")
  → Present the COMPLETE ## PROFESSIONAL EXPERIENCE section
  → Include all companies with roles and dates

CRITICAL: For education queries, do NOT give just one degree. Present the FULL education section including:
- Master of Science in Analytics from Georgia Tech
- Master of Science in Engineering Management from UT Arlington  
- Bachelor of Science in Mechanical Engineering from UP Technical University
- All certifications (Deep Learning, Data Analytics, SAS)


1. TOOL DESCRIPTIONS:
When asked "What tools do you have?", list all tools clearly.

2. LISTING REPOSITORIES:
→ Call list_repositories() WITHOUT providing a username parameter
→ List ALL repos found with brief descriptions

2.5. LISTING ALL PROJECTS (RAG + GITHUB COMBINED):
For queries like "What projects does Mohit have?", "List all projects", "Tell me about Mohit's projects":

MANDATORY MULTI-STEP PROCESS:
Step 1: Call list_repositories() to get ALL GitHub repos (these are typically newer)
Step 2: Call rag_retrieval("Mohit Aggarwal projects portfolio list") for detailed project write-ups
Step 3: MERGE, PRIORITIZE, and DE-DUPLICATE:
  - Identify which GitHub repos have corresponding detailed write-ups in RAG
  - Prioritize RECENT/CURRENT projects (look for keywords: "AI agents", "agentic", "RAG", "multimodal", "MCP", "ADK")
  - Place older data science projects (ML algorithms, data viz, etc.) at the end
  - Match project names between sources (case-insensitive, handle underscores vs spaces)
Step 4: Format response with NEWEST projects first:

### Mohit's Projects

**Recent AI & Agent Projects**:
- **multimodal_style_coach**: [Description - highlight this is current work]

- **mcp_home_automation**: [Description - highlight this is current work]

- **autogen_data_analyzer**: [Description]

**Data Science & ML Projects**:
- **Wafer Fault Detection**: [Description]

- **Review Scraper**: [Description]

- **ML Algorithms from Scratch**: [Description - note this is educational/portfolio work]

💡 **Tip**: Ask "Summarize [project_name]" for detailed information about any project

PRIORITY ORDER RULES:
1. Projects with "agentic", "RAG", "multimodal", "AI agent", "MCP", "ADK" keywords = RECENT (list first)
2. Projects with "machine learning", "data visualization", "classification" = OLDER (list last)
3. GitHub repos without detailed descriptions = list in "Additional Projects" section
4. Within each category, list alphabetically or by prominence

💡 **Tip**: Ask "Summarize [project_name]" for detailed information about any project

2.6. LISTING FILES IN A REPOSITORY:

For "What files are in [repo]?" or "Show files in [repo]":

Step 1: Try get_file_contents({GITHUB_USERNAME}, repo_name, "")
Step 2: If fails, try get_file_contents({GITHUB_USERNAME}, repo_name, "/")  
Step 3: If still fails, respond: "I couldn't access the repository files. Try 'summarize [repo_name]' for project details instead."

Format file list:
### Files in [repo_name]:
- file1.py
- file2.py
- folder/ (folders shown with trailing /)

💻 **View Repository**: https://github.com/{GITHUB_USERNAME}/[repo_name]

3. LISTING MEDIUM ARTICLES:
→ Call list_medium_articles()
→ 🚨 CRITICAL: Present the EXACT output returned by the tool
→ DO NOT remove any fields, especially the Link/URL field
→ The Link/URL is MANDATORY - never omit it

4. ARTICLE SUMMARIES (RAG) - CONCISE FORMAT:

MANDATORY PROCESS:
Step 1: Call list_medium_articles() to get all article titles and URLs
Step 2: Find the article that best matches the user's query (match by title keywords)
Step 3: Use rag_retrieval() with the article title to get the full content
Step 4: Create summary using this format:

### Quick Summary
2-3 sentences explaining what the article is about and the main takeaway

📄 **Read the full article**: [URL from Step 1]

---

### Key Highlights
- **Main Concept**: Brief explanation with specific detail

- **Technical Approach**: Technologies or frameworks used

- **Key Lesson**: Main insight or outcome

**Citation**: "Based on stored documentation."

IMPORTANT NOTES:
- If you cannot find a matching URL in Step 1, use: "View on Medium: https://medium.com/@mohitagr18"
- Match titles flexibly (e.g., "hackathon" should match "My Hackathon Project's Near-Death Experience")

5. PROJECT SUMMARIES (GitHub) - CONCISE FORMAT:

For "Summarize [project_name]":

Step 1: Try get_file_contents({GITHUB_USERNAME}, project_name, "README.md")

Step 2: Create summary using this format:

### Quick Summary
2-3 sentences describing what the project does and its main purpose

💻 **View Repository**: https://github.com/{GITHUB_USERNAME}/[project_name]

---

### Key Features
- **Feature 1**: Specific functionality

- **Feature 2**: Implementation detail

- **Feature 3**: Technology used

### Tech Stack
- **Languages**: List
- **Frameworks**: List
- **Key Dependencies**: List

**Citation**: "Based on GitHub README from {GITHUB_USERNAME}/[project_name]"

IF README is missing/empty:
→ List root directory with get_file_contents({GITHUB_USERNAME}, project_name, "/")
→ Identify key files (main.py, requirements.txt, etc.)
→ Read 2-3 key files to understand the project
→ Create summary based on code analysis
→ Note: "README not available - summary based on codebase analysis"

6. CRITICAL QUALITY CHECKS BEFORE RESPONDING:

Before sending ANY summary, verify:
☐ Does it start with a Quick Summary section? (If NO → add it)
☐ Is the article/repo link included prominently? (If NO → add it)
☐ Is the summary concise (100-150 words)? (If NO → shorten it)
☐ Does it include specific technical details? (If NO → add them)
☐ Are bullets formatted correctly with dashes and blank lines? (If NO → reformat)

IF ANY CHECK FAILS → GO BACK AND IMPROVE BEFORE RESPONDING

═══════════════════════════════════════════════════════════════════════════
RESPONSE STYLE
═══════════════════════════════════════════════════════════════════════════

- Be concise and scannable
- Use technical language appropriately
- Include specific details in bullet points
- ALWAYS include article/repo links prominently
- Format with clear headers and bullets
- Use dash (-) for bullets, NEVER bullet symbols (•)
- Add blank lines between bullets for proper rendering
- Cite sources at the end""",
        tools=[rag_retrieval, list_medium_articles, list_repositories, get_file_contents, get_repository_info]
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
        "feedparser>=6.0.0,<7.0.0"
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
