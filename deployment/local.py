import os
import sys

import vertexai
from dotenv import load_dotenv
from vertexai.preview import reasoning_engines

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multi_tool.agent import root_agent

# from agent import root_agent


def main():
    # Load environment variables
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY2")

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    if not project_id:
        print("Missing required environment variable: GOOGLE_CLOUD_PROJECT")
        sys.exit(1)
    elif not location:
        print("Missing required environment variable: GOOGLE_CLOUD_LOCATION")
        sys.exit(1)

    vertexai.init(project=project_id, location=location)

    app = reasoning_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True,
    )

    session = app.create_session(user_id='test_user')
    print("Session created:")
    print(f"  Session ID: {session.id}")
    print(f"  User ID: {session.user_id}")
    print(f"  App name: {session.app_name}")

    sessions = app.list_sessions(user_id='test_user')

    if hasattr(sessions,'sessions'):
        print(f'Found {len(sessions.sessions)} sessions:')
    elif hasattr(sessions, "session_ids"):
        print(f"Found session IDs: {sessions.session_ids}")
    else:
        print(f"Sessions response: {sessions}")

    

    test_message = ("Hello! What is the weather in Glendale, CA?")

    for event in app.stream_query(
        user_id='test_user',
        session_id=session.id,
        message=test_message
    ):
        print(event)

if __name__ == "__main__":
    main()