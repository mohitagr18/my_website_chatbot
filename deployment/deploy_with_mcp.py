# deployment/deploy_with_mcp.py
"""
Deploy agent with MCP using custom installation script (Google official pattern).
"""
import os
import sys
from dotenv import load_dotenv
import vertexai
from vertexai import agent_engines

load_dotenv()

def deploy():
    """Deploy agent with MCP using build_options."""
    
    # Get paths
    project_root = os.getcwd()
    multi_tool_path = os.path.join(project_root, "multi_tool")
    install_script_path = os.path.join(project_root, "installation_scripts", "install_mcp_server.sh")
    
    # Verify paths
    if not os.path.exists(multi_tool_path):
        raise FileNotFoundError(f"multi_tool not found at {multi_tool_path}")
    if not os.path.exists(install_script_path):
        raise FileNotFoundError(f"Install script not found at {install_script_path}")
    
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4")
    staging_bucket = os.environ.get("GOOGLE_CLOUD_STAGING_BUCKET")
    
    vertexai.init(
        project=project,
        location=location,
        staging_bucket=staging_bucket
    )
    
    # Environment variables
    env_vars = {}
    if os.getenv("RAG_CORPUS"):
        env_vars["RAG_CORPUS"] = os.getenv("RAG_CORPUS")
    if os.getenv("GITHUB_USERNAME"):
        env_vars["GITHUB_USERNAME"] = os.getenv("GITHUB_USERNAME")
    if os.getenv("GITHUB_TOKEN"):
        env_vars["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")
    
    print("🚀 Deploying agent with MCP using custom installation script...")
    print(f"   Project: {project}")
    print(f"   Location: {location}")
    print(f"   Package: {multi_tool_path}")
    print(f"   Install script: {install_script_path}")
    
    # Deploy using ModuleAgent with build_options
    # CRITICAL FIX: Pass the entire multi_tool directory, not individual files
    remote_app = agent_engines.create(
        display_name="portfolio_multi_tool_agent_mcp",
        description="Portfolio assistant with RAG and GitHub MCP",
        agent_engine=agent_engines.ModuleAgent(
            module_name="multi_tool.agent",
            agent_name="agent_app",
            register_operations={
                "": ["get_session", "query"],
                "stream": ["stream_query"],
            },
        ),
        requirements=[
            "google-adk>=1.7.0",
            "google-cloud-aiplatform[adk,agent_engines]>=1.101.0",
            "pydantic>=2.11.3",
            "python-dotenv>=1.1.0",
        ],
        extra_packages=[
            multi_tool_path,  # Pass directory, not individual files
        ],
        env_vars=env_vars,
        build_options={
            "installation": [
                install_script_path,  # Full path to install script
            ],
        },
    )
    
    print("\n" + "="*70)
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print("="*70)
    print(f"Resource Name: {remote_app.resource_name}")
    print("\n📝 UPDATE YOUR .env FILE:")
    print(f"AGENT_RESOURCE_NAME={remote_app.resource_name}")
    print("="*70 + "\n")
    
    return remote_app

if __name__ == "__main__":
    deploy()
