This project is to implement a LangGraph Agent using Reflexion pattern.
Project will use Tavily as the search tool and the reflexion technique.

# Virtual Environment

Project uses uv package manager and a Python virtual environment.
source .venv/bin/activate : Run virtual environment

# Project Structure

schemas.py : Includes Pydantic objects for structured output
chains.py : Python code that implements the necessary chains
tool_executor.py : Tool Calling functionality implemented
main.py : Main LangGraph graph file