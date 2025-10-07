# 🤖 Portfolio Chatbot with Multi-Tool Agent

This project implements an AI-powered portfolio chatbot for Mohit Aggarwal. The chatbot can answer questions about Mohit's work by accessing two distinct data sources: technical articles via Vertex AI RAG Engine and live code repositories via the GitHub API. This provides a comprehensive, interactive way to explore his professional portfolio.

The agent is built with the Google Agent Development Kit (ADK) and deployed as a serverless application on Vertex AI Agent Engine, with a user-friendly Streamlit interface for interaction.

## ✨ Features

*   **Dual Data Sources**: The chatbot intelligently queries either a **Vertex AI RAG Engine** for published articles and documentation or the **GitHub API** for real-time information on code repositories.
*   **Smart Routing**: It uses query patterns to decide the best data source. For example, queries containing "article" are routed to the RAG engine, while queries with `snake_case` repository names are sent to GitHub.
*   **In-Depth Summaries**: The agent is instructed to provide detailed, structured summaries of 400+ words for both articles and projects, ensuring high-quality, comprehensive answers.
*   **Interactive UI**: A Streamlit application provides a simple chat interface for users to interact with the deployed agent.
*   **Serverless Deployment**: The entire agent is deployed on Vertex AI Agent Engine, which handles scaling and serving automatically.

## 🏗️ Architecture

The system is composed of three main parts:
1.  **Streamlit UI**: A simple front-end application that sends user queries to the agent.
2.  **Agent Client**: A helper script that handles communication with the deployed Vertex AI Agent Engine.
3.  **Vertex AI Agent Engine**: The serverless backend where the multi-tool agent is deployed. The agent uses the `gemini-2.5-flash` model and has access to two sets of tools:
    *   `rag_retrieval()`: Searches the document corpus in the RAG Engine.
    *   `list_repositories()`, `get_file_contents()`, `get_repository_info()`: A set of tools for interacting with the GitHub API.

```
User Query
    ↓
Streamlit UI → Agent Client
    ↓
Vertex AI Agent Engine (deployed)
    ├─→ RAG Retrieval Tool → Vertex AI RAG Engine → GCS Bucket
    ├─→ GitHub API Tools → GitHub REST API
    └─→ gemini-2.5-flash (LLM)
```

## 📁 Project Structure

```
my_website_chatbot/
├── multi_tool/
│   ├── agent.py              # Original agent (for local adk web testing)
│   └── requirements.txt
├── deployment/
│   ├── remote.py             # Agent defined INLINE for deployment
│   ├── streamlit_app.py      # UI with intro section
│   └── agent_client.py       # Query helper
├── .env                       # Contains API keys and resource names
└── adk.yaml
```

## 🚀 Getting Started

### ✅ Prerequisites

*   Python 3.9+
*   Access to Google Cloud Platform with a configured project
*   A GitHub account and a personal access token (optional but recommended)

### 📦 Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd my_website_chatbot
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### ⚙️ Configuration

Create a `.env` file in the root of the project and add the following environment variables:

```
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
GOOGLE_CLOUD_LOCATION="your-gcp-region"
GOOGLE_CLOUD_STAGING_BUCKET="your-gcs-bucket-name"
RAG_CORPUS="your-rag-corpus-resource-name"
GITHUB_USERNAME="your-github-username"
GITHUB_TOKEN="your-github-personal-access-token"
AGENT_RESOURCE_NAME="" # This will be populated after deployment
```

## 💻 Usage

### 🧪 Local Testing

You can test the agent locally using the `adk web` command. This is useful for debugging the agent's logic before deploying it.

```bash
adk web multi_tool
```

### ☁️ Deployment

To deploy the agent to Vertex AI Agent Engine, run the `remote.py` script:

```bash
python -m deployment.remote create-agent
```

After a successful deployment, the script will output the `AGENT_RESOURCE_NAME`. Copy this value and add it to your `.env` file.

### 🖥️ Running the Streamlit UI

Once the agent is deployed and the `AGENT_RESOURCE_NAME` is set in your `.env` file, you can run the Streamlit application:

```bash
streamlit run deployment/streamlit_app.py
```

This will launch a local web server with the chat interface.

## 💬 Sample Queries

Here are a few examples of queries you can try:

*   "List my repositories"
*   "Summarize mcp_home_automation"
*   "Summarize the hackathon article"
*   "What projects involve AI agents?"

## 🧠 Key Learnings

*   **Inline Agent Definition**: Defining the agent directly within the deployment script (`remote.py`) is crucial to avoid `cloudpickle` module import errors on Agent Engine.
*   **Smart Routing Logic**: Explicitly instructing the agent on how to choose between the RAG engine and GitHub tools based on the query prevents hallucinations and improves the accuracy of the responses.
*   **Detailed Instructions**: Providing very specific instructions on the desired length and structure of the summaries results in much higher quality output from the LLM.