import os 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool # Build a toolkit of multiple tools
from langchain_core.messages import HumanMessage # We will be using HumanMessage to invoke the Agent. So this is going to be the input for the Agent's execution.
from langchain_openai import ChatOpenAI # This supplies the LLM for our Agent
"""
AGENT = LLM + ToolKit
Toolkit = list of Tools
Tool = any function that agent can execute ==> We are specifying the ACTIONs to perform 
        - Function with docstrings, typings, arguments it receive, what output.
"""


# Print the constructed path for debugging
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
print(f"Looking for .env at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")



# Load environment variables from .env file in project root
load_dotenv(env_path)

@tool
def search(query: str) -> str:
    """
    Search the internet for a given query and return relevant results.
    
    This tool enables the agent to search the internet and retrieve information
    based on the provided query string. It performs a web search and returns
    the most relevant search results related to the query.
    
    Args:
        query (str): The search query to look up on the internet.
                    Can be a single term, phrase, or complex question.
    
    Returns:
        str: A string containing the search results related to the query.
             Includes relevant information, summaries, or links found from
             the internet search.
    
    Example:
        >>> search("weather in Tokyo")
        "Tokyo weather is sunny and 22°C..."
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


llm = ChatOpenAI()

toolkit = [search]

agent = create_agent(model=llm, tools=toolkit)



def main():
    print("Hello from langchain")
    # Pass a messages dict
    # The messages can be a list, if not internally it converts to a list
    # HumanMessage invokes the agent to respond.
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
    print(result)


if __name__ == '__main__':
    main()


"""

Agent Response Flow:
├── 1. User Input
│   └── HumanMessage: "What is the weather in Tokyo?"
│
├── 2. Agent Decision
│   └── AIMessage (tool_calls triggered)
│       └── Tool Call: search
│           └── Args: {"query": "weather in Tokyo"}
│           └── Tokens Used: 189 input, 15 output
│
├── 3. Tool Execution
│   └── ToolMessage: "Tokyo weather is sunny"
│       └── Tool Name: search
│
└── 4. Final Response
    └── AIMessage: "The weather in Tokyo is currently sunny."
        └── Tokens Used: 216 input, 9 output
        └── Status: Completed (finish_reason: 'stop')

Total Token Usage: 225 tokens (completion)
Model: gpt-3.5-turbo-0125


========================== Raw response ==================================
{
  "messages": [
    {
      "type": "HumanMessage",
      "content": "What is the weather in Tokyo?",
      "id": "a1fe083d-a7c7-4cb1-b308-330f5ee89edd"
    },
    {
      "type": "AIMessage",
      "content": "",
      "finish_reason": "tool_calls",
      "model": "gpt-3.5-turbo-0125",
      "tokens": {
        "input": 189,
        "output": 15,
        "total": 204
      },
      "tool_calls": [
        {
          "name": "search",
          "args": {
            "query": "weather in Tokyo"
          },
          "id": "call_GFaGctoYELX8I7mhxfbubpj0"
        }
      ],
      "id": "lc_run--019ba21a-873e-7542-969f-3027937ea496-0"
    },
    {
      "type": "ToolMessage",
      "tool_name": "search",
      "content": "Tokyo weather is sunny",
      "tool_call_id": "call_GFaGctoYELX8I7mhxfbubpj0",
      "id": "55fae38b-98b2-4a92-9398-f4a4dfc776d0"
    },
    {
      "type": "AIMessage",
      "content": "The weather in Tokyo is currently sunny.",
      "finish_reason": "stop",
      "model": "gpt-3.5-turbo-0125",
      "tokens": {
        "input": 216,
        "output": 9,
        "total": 225
      },
      "id": "lc_run--019ba21a-8d92-75e3-a4b6-a090141f6efc-0"
    }
  ],
  "summary": {
    "total_tokens_used": 429,
    "status": "completed",
    "model": "gpt-3.5-turbo-0125"
  }
}

"""