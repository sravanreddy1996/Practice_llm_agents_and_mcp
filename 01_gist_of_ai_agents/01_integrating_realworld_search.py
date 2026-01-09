import os 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool # Build a toolkit of multiple tools
from langchain_core.messages import HumanMessage # We will be using HumanMessage to invoke the Agent. So this is going to be the input for the Agent's execution.
from langchain_openai import ChatOpenAI # This supplies the LLM for our Agent
from tavily import TavilyClient

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

tavily_search_client = TavilyClient()


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
    return tavily_search_client.search(query=query)


llm = ChatOpenAI(name='gpt-5-mini')

toolkit = [search]

agent = create_agent(model=llm, tools=toolkit)



def main():
    print("Hello from langchain")
    # Pass a messages dict
    # The messages can be a list, if not internally it converts to a list
    # HumanMessage invokes the agent to respond.
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details")})
    print(result)


if __name__ == '__main__':
    main()


"""

```markdown
## Agent Execution Flow

```
Agent Response Tree
│
├── 1️⃣ User Input
│   └── 💬 HumanMessage
│       ├── Content: "What is the weather in Tokyo?"
│       └── ID: 24fd33ca-75e3-4d1d-8ce1-3977bc284d6f
│
├── 2️⃣ Agent Decision Making
│   └── 🤖 AIMessage (Tool Selection)
│       ├── Finish Reason: tool_calls
│       ├── Model: gpt-3.5-turbo-0125
│       ├── 📊 Token Usage
│       │   ├── Input: 189
│       │   ├── Output: 15
│       │   └── Total: 204
│       └── 🔧 Tool Calls
│           └── Tool: search
│               ├── Query: "weather in Tokyo"
│               ├── Call ID: call_4GA8IhyjgIGdruLSa9Z8Pgib
│               └── Type: tool_call
│
├── 3️⃣ Tool Execution
│   └── 🛠️ ToolMessage (Search Results)
│       ├── Tool Name: search
│       ├── Tool Call ID: call_4GA8IhyjgIGdruLSa9Z8Pgib
│       └── Current Weather Data
│           ├── Location: Tokyo, Japan
│           ├── Temperature: 9.0°C (48.2°F)
│           ├── Condition: Partly cloudy
│           ├── Humidity: 53%
│           ├── Wind: 11.9 km/h (7.4 mph) - South
│           ├── Pressure: 1019.0 mb
│           ├── Visibility: 10.0 km
│           ├── Last Updated: 2026-01-09 19:15
│           └── Results Count: 5
│
└── 4️⃣ Final Response
    └── 🤖 AIMessage (Final Answer)
        ├── Content: "The current weather in Tokyo is partly cloudy with a temperature of 9.0°C (48.2°F). The wind is blowing at 11.9 km/h (7.4 mph) from the south. The humidity is 53%, and the visibility is 10.0 km (6.0 miles)."
        ├── Finish Reason: stop
        ├── Model: gpt-3.5-turbo-0125
        ├── 📊 Token Usage
        │   ├── Input: 2023
        │   ├── Output: 66
        │   └── Total: 2089
        └── ID: lc_run--019ba243-63cd-7293-904d-313e41ca1922-0

📈 Summary
├── Total Tokens Used: 4582
├── Status: ✅ Completed
├── Model: gpt-3.5-turbo-0125
└── Response Time: 1.97s
```
```

========================== Raw response ==================================
{
  "messages": [
    {
      "type": "HumanMessage",
      "content": "What is the weather in Tokyo?",
      "id": "24fd33ca-75e3-4d1d-8ce1-3977bc284d6f"
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
          "id": "call_4GA8IhyjgIGdruLSa9Z8Pgib"
        }
      ],
      "id": "lc_run--019ba243-530c-72e0-aac5-444efd224343-0"
    },
    {
      "type": "ToolMessage",
      "tool_name": "search",
      "content": {
        "query": "weather in Tokyo",
        "current_weather": {
          "location": "Tokyo, Japan",
          "temperature_c": 9.0,
          "temperature_f": 48.2,
          "condition": "Partly cloudy",
          "humidity": "53%",
          "wind_kph": 11.9,
          "wind_mph": 7.4,
          "wind_direction": "South",
          "pressure_mb": 1019.0,
          "visibility_km": 10.0,
          "last_updated": "2026-01-09 19:15"
        },
        "top_results": 5
      },
      "tool_call_id": "call_4GA8IhyjgIGdruLSa9Z8Pgib",
      "id": "1eb65ac8-8ce5-4b11-8c86-ade4c225a236"
    },
    {
      "type": "AIMessage",
      "content": "The current weather in Tokyo is partly cloudy with a temperature of 9.0°C (48.2°F). The wind is blowing at 11.9 km/h (7.4 mph) from the south. The humidity is 53%, and the visibility is 10.0 km (6.0 miles).",
      "finish_reason": "stop",
      "model": "gpt-3.5-turbo-0125",
      "tokens": {
        "input": 2023,
        "output": 66,
        "total": 2089
      },
      "id": "lc_run--019ba243-63cd-7293-904d-313e41ca1922-0"
    }
  ],
  "summary": {
    "total_tokens_used": 4582,
    "status": "completed",
    "model": "gpt-3.5-turbo-0125",
    "response_time": "1.97s"
  }
}


=====================================================================================
```markdown
## Agent Job Search Execution Flow

```
Agent Response Tree
│
├── 1️⃣ User Input
│   └── 💬 HumanMessage
│       ├── Content: "search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details"
│       └── ID: a2b4ca88-62f2-4cdd-8557-e50a5fea8e59
│
├── 2️⃣ Agent Decision Making
│   └── 🤖 AIMessage (Tool Selection)
│       ├── Finish Reason: tool_calls
│       ├── Model: gpt-3.5-turbo-0125
│       ├── 📊 Token Usage
│       │   ├── Input: 204
│       │   ├── Output: 25
│       │   └── Total: 229
│       └── 🔧 Tool Calls
│           └── Tool: search
│               ├── Query: "AI engineer job postings using Langchain in India site:linkedin.com"
│               ├── Call ID: call_npCUUV91htB2G9x46B7zqcm7
│               └── Type: tool_call
│
├── 3️⃣ Tool Execution
│   └── 🛠️ ToolMessage (LinkedIn Job Search Results)
│       ├── Tool Name: search
│       ├── Query: "AI engineer job postings using Langchain in India"
│       ├── Results Count: 3
│       │
│       └── 📋 Job Postings Found
│           │
│           ├── 🏢 Job #1: AI Engineer - LangChain & Agentic Systems
│           │   ├── Company: tvara
│           │   ├── Location: Bengaluru, Karnataka, India
│           │   ├── Posted: 1 month ago
│           │   ├── Role Focus: Python, LangChain & Agentic Systems
│           │   ├── Applicants: Among first 25
│           │   ├── Relevance Score: 0.9999971
│           │   └── URL: https://in.linkedin.com/jobs/view/ai-engineer-langchain-agentic-systems-python-at-tvara-4321717133
│           │
│           ├── 🏢 Job #2: AI Engineer - GPT / LangChain / RAG / Data Pipelines
│           │   ├── Company: Peak Trust Global Real Estate India
│           │   ├── Location: India
│           │   ├── Technologies: GPT, LangChain, RAG, Data Pipelines
│           │   ├── Skills Required: Python, Automation, Cloud & AI
│           │   ├── Relevance Score: 0.9999889
│           │   └── URL: https://in.linkedin.com/jobs/view/ai-engineer-gpt-langchain-rag-data-pipelines-at-peak-trust-4302171754
│           │
│           └── 🏢 Job #3: AI Engineer (LangChain, Azure Ecosystem)
│               ├── Company: TECHXLE CONSULTING SERVICES PVT LTD
│               ├── Location: Coimbatore, Tamil Nadu, India
│               ├── Technologies: LangChain, Azure Ecosystem
│               ├── Role Focus: AI Engineering with Azure Cloud Integration
│               ├── Relevance Score: High
│               └── URL: https://in.linkedin.com/jobs/view/ai-engineer-langchain-azure-ecosystem-at-techxle-consulting-services-pvt-ltd-4278496049
│
└── 4️⃣ Final Response
    └── 🤖 AIMessage (Formatted Job Summary)
        ├── Content: Structured list of 3 AI Engineer positions with:
        │   ├── Company names
        │   ├── Job titles
        │   ├── Locations
        │   ├── Technologies (LangChain, Python, RAG, Azure, etc.)
        │   └── LinkedIn job posting URLs
        ├── Finish Reason: stop
        ├── Model: gpt-3.5-turbo-0125
        ├── 📊 Token Usage
        │   ├── Input: 1433
        │   ├── Output: 238
        │   └── Total: 1671
        └── ID: lc_run--019ba263-1c60-7ae1-b811-8625e0ccec25-0

📈 Summary
├── Total Tokens Used: 1900
├── Status: ✅ Completed Successfully
├── Model: gpt-3.5-turbo-0125
├── Jobs Found: 3
├── Locations: Bengaluru, Coimbatore, Pan-India
└── Key Technology: LangChain, Python, AI/ML, Azure
```
```
"""