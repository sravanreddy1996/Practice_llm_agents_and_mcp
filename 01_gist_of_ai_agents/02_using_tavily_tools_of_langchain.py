import os 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool # Build a toolkit of multiple tools
from langchain_core.messages import HumanMessage # We will be using HumanMessage to invoke the Agent. So this is going to be the input for the Agent's execution.
from langchain_openai import ChatOpenAI # This supplies the LLM for our Agent
from langchain_tavily import TavilySearch # Langchain tools related to Tavily Web Search implemented by Tavily written in Langchain.

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

# tavily_search_client = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Search the internet for a given query and return relevant results.
    
#     This tool enables the agent to search the internet and retrieve information
#     based on the provided query string. It performs a web search and returns
#     the most relevant search results related to the query.
    
#     Args:
#         query (str): The search query to look up on the internet.
#                     Can be a single term, phrase, or complex question.
    
#     Returns:
#         str: A string containing the search results related to the query.
#              Includes relevant information, summaries, or links found from
#              the internet search.
    
#     Example:
#         >>> search("weather in Tokyo")
#         "Tokyo weather is sunny and 22°C..."
#     """
#     print(f"Searching for {query}")
#     return tavily_search_client.search(query=query)


llm = ChatOpenAI(name='gpt-5-mini')

toolkit = [TavilySearch()]

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
## Agent Job Search - Tavily Search Integration

```
Agent Response Tree
│
├── 1️⃣ User Input
│   └── 💬 HumanMessage
│       ├── Content: "search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details"
│       └── ID: 8203feb6-3621-424e-9670-afb6d1ddcf48
│
├── 2️⃣ Agent Decision Making
│   └── 🤖 AIMessage (Tool Selection)
│       ├── Finish Reason: tool_calls
│       ├── Model: gpt-3.5-turbo-0125
│       ├── 📊 Token Usage
│       │   ├── Input: 1273
│       │   ├── Output: 39
│       │   └── Total: 1312
│       └── 🔧 Tool Calls
│           └── Tool: tavily_search
│               ├── Query: "AI engineer job posting using Langchain in India site:linkedin.com"
│               ├── Include Domains: linkedin.com
│               ├── Search Depth: advanced
│               ├── Call ID: call_elL7MYgBVfcaWoOZyh460cX3
│               └── Type: tool_call
│
├── 3️⃣ Tool Execution
│   └── 🛠️ ToolMessage (LinkedIn Job Search Results)
│       ├── Query: "AI engineer job posting using Langchain in India"
│       ├── Results Count: 5 (Top 3 Selected)
│       │
│       └── 📋 Job Postings Found
│           │
│           ├── 🏢 Job #1: Agentic AI / LLM Engineer
│           │   ├── Company: Meraki TalentWorks
│           │   ├── Location: PAN India
│           │   ├── Experience: 8–12 Years (Hands-on)
│           │   ├── Joining: Immediate (within 15 days)
│           │   ├── Project: High-impact global project with Mastercard
│           │   ├── Key Technologies: LangGraph, CrewAI, AutoGen, LangChain, LlamaIndex
│           │   ├── Key Skills:
│           │   │   ├── Agentic AI frameworks
│           │   │   ├── Python (Async programming, FastAPI)
│           │   │   ├── LLM & RAG (Prompt engineering, embeddings)
│           │   │   └── ML & Statistics
│           │   ├── Relevance Score: 0.9053309
│           │   └── URL: https://www.linkedin.com/posts/meraki-talentworks_hiring-hiringnow-jobs-activity-7407355209200570368-DGE_
│           │
│           ├── 🏢 Job #2: AI/ML Engineer – Python (LLM, RAG, LangChain, LlamaIndex)
│           │   ├── Company: Futops Technologies India Private Limited
│           │   ├── Location: Pune, Maharashtra, India
│           │   ├── Experience: 1-2 Years
│           │   ├── Employment Type: Full-time
│           │   ├── Working Mode: Work from office (Monday to Friday)
│           │   ├── Key Responsibilities:
│           │   │   ├── Develop LLM-based applications
│           │   │   ├── Build RAG systems using LangChain and LlamaIndex
│           │   │   └── Create cutting-edge AI systems
│           │   ├── Required Skills:
│           │   │   ├── Strong Python expertise (hands-on)
│           │   │   ├── LLM & RAG proficiency
│           │   │   ├── LangChain & LlamaIndex experience
│           │   │   └── ML fundamentals
│           │   ├── Relevance Score: 0.8795718
│           │   └── URL: https://in.linkedin.com/jobs/view/ai-ml-engineer-%E2%80%93-python-llm-rag-langchain-llamaindex-at-futops-technologies-india-private-limited-4270260171
│           │
│           └── 🏢 Job #3: AI Engineer (Generative AI, AWS, LangChain)
│               ├── Company: Cuedo Analytics
│               ├── Location: Bengaluru, Karnataka, India
│               ├── Seniority Level: Mid-Senior level
│               ├── Employment Type: Full-time
│               ├── Posted: 1 month ago
│               ├── Key Focus: Generative AI, AWS, LangChain
│               ├── Key Responsibilities:
│               │   ├── Build AI/ML solutions using LangChain
│               │   ├── Work with Generative AI models
│               │   └── AWS infrastructure & deployment
│               ├── Benefits:
│               │   ├── Cutting-edge LLM and AI architectures
│               │   ├── Collaborative team environment
│               │   ├── Exposure to advanced agent systems & vector search
│               │   └── Growth opportunities in AI architecture & leadership
│               ├── Relevance Score: 0.8740022
│               └── URL: https://in.linkedin.com/jobs/view/ai-engineer-generative-ai-aws-langchain-at-cuedo-analytics-4339924545
│
└── 4️⃣ Final Response
    └── 🤖 AIMessage (Formatted Job Summary)
        ├── Content: Structured list of 3 AI Engineer positions with:
        │   ├── Company names
        │   ├── Job titles
        │   ├── Locations (PAN India, Pune, Bengaluru)
        │   ├── Experience levels (1-2 yrs to 8-12 yrs)
        │   ├── Technologies (LangChain, Python, RAG, LLM, AWS)
        │   ├── Key responsibilities
        │   ├── Benefits & opportunities
        │   └── LinkedIn job posting URLs
        ├── Finish Reason: stop
        ├── Model: gpt-3.5-turbo-0125
        ├── 📊 Token Usage
        │   ├── Input: 4085
        │   ├── Output: 464
        │   └── Total: 4549
        └── ID: lc_run--019ba354-5b07-7721-b724-1872a1ce85ba-0

📈 Summary
├── Total Tokens Used: 5861
├── Status: ✅ Completed Successfully
├── Tool Used: Tavily Search (Advanced Depth)
├── Model: gpt-3.5-turbo-0125
├── Jobs Found: 3+ (from 5 results)
├── Locations: PAN India, Pune, Bengaluru
├── Experience Range: 1-2 years to 8-12 years
├── Key Technologies: LangChain, RAG, LLM, Python, AWS, CrewAI, AutoGen
└── Key Skills: Agentic AI, Prompt Engineering, Vector Search, FastAPI
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

================================================================================================
{
  "messages": [
    {
      "type": "HumanMessage",
      "content": "search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details",
      "additional_kwargs": {},
      "response_metadata": {},
      "id": "8203feb6-3621-424e-9670-afb6d1ddcf48"
    },
    {
      "type": "AIMessage",
      "content": "",
      "additional_kwargs": {
        "refusal": null
      },
      "response_metadata": {
        "token_usage": {
          "completion_tokens": 39,
          "prompt_tokens": 1273,
          "total_tokens": 1312,
          "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
          },
          "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 0
          }
        },
        "model_provider": "openai",
        "model_name": "gpt-3.5-turbo-0125",
        "system_fingerprint": null,
        "id": "chatcmpl-Cw8VaKfMVKz8ocRgF7AsN0vo8gk0E",
        "service_tier": "default",
        "finish_reason": "tool_calls",
        "logprobs": null
      },
      "tool_calls": [
        {
          "name": "tavily_search",
          "args": {
            "query": "AI engineer job posting using Langchain in India site:linkedin.com",
            "include_domains": [
              "linkedin.com"
            ],
            "search_depth": "advanced"
          },
          "id": "call_elL7MYgBVfcaWoOZyh460cX3",
          "type": "tool_call"
        }
      ],
      "invalid_tool_calls": [],
      "usage_metadata": {
        "input_tokens": 1273,
        "output_tokens": 39,
        "total_tokens": 1312,
        "input_token_details": {
          "audio": 0,
          "cache_read": 0
        },
        "output_token_details": {
          "audio": 0,
          "reasoning": 0
        }
      },
      "id": "lc_run--019ba354-494c-75d3-95d4-ef080245be50-0"
    },
    {
      "type": "ToolMessage",
      "content": {
        "query": "AI engineer job posting using Langchain in India",
        "follow_up_questions": null,
        "answer": null,
        "images": [],
        "results": [
          {
            "url": "https://www.linkedin.com/posts/meraki-talentworks_hiring-hiringnow-jobs-activity-7407355209200570368-DGE_",
            "title": "Meraki TalentWorks' Post",
            "content": "Urgent Hiring | Agentic AI / LLM Engineer | PAN India | LangChain, LangGraph, CrewAI, AutoGen",
            "score": 0.9053309,
            "raw_content": null
          },
          {
            "url": "https://in.linkedin.com/jobs/view/ai-ml-engineer-%E2%80%93-python-llm-rag-langchain-llamaindex-at-futops-technologies-india-private-limited-4270260171",
            "title": "AI/ML Engineer – Python (LLM, RAG, LangChain, LlamaIndex)",
            "content": "Futops Technologies India Pvt Ltd | Pune | 1–2 Years | Full-time | Work from Office",
            "score": 0.8795718,
            "raw_content": null
          },
          {
            "url": "https://in.linkedin.com/jobs/view/ai-engineer-generative-ai-aws-langchain-at-cuedo-analytics-4339924545",
            "title": "AI Engineer (Generative AI, AWS, LangChain)",
            "content": "Cuedo Analytics | Bengaluru | Mid-Senior | Full-time",
            "score": 0.8740022,
            "raw_content": null
          },
          {
            "url": "https://in.linkedin.com/jobs/view/ai-ml-engineer-langchain-%2B-rag-%2B-multi-agent-at-cloudzent-technology-services-4328123821",
            "title": "AI/ML Engineer (LangChain + RAG + Multi-Agent)",
            "content": "CloudZent Technology Services | Gurugram | Mid-Senior | Full-time",
            "score": 0.8593928,
            "raw_content": null
          },
          {
            "url": "https://in.linkedin.com/jobs/view/ai-ml-engineer-langchain-llama-at-sampoorna-consultants-4346792517",
            "title": "AI/ML Engineer - LangChain/LLama",
            "content": "Sampoorna Consultants | Chennai | Entry Level | Full-time",
            "score": 0.0,
            "raw_content": null
          }
        ]
      },
      "additional_kwargs": {
        "refusal": null
      },
      "response_metadata": {
        "token_usage": {
          "completion_tokens": 464,
          "prompt_tokens": 4085,
          "total_tokens": 4549,
          "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
          },
          "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 0
          }
        },
        "model_provider": "openai",
        "model_name": "gpt-3.5-turbo-0125",
        "system_fingerprint": null,
        "id": "chatcmpl-Cw8VeZf9ugrpr9QqKzSdFPrs4eLy7",
        "service_tier": "default",
        "finish_reason": "stop",
        "logprobs": null
      },
      "tool_calls": [],
      "invalid_tool_calls": [],
      "usage_metadata": {
        "input_tokens": 4085,
        "output_tokens": 464,
        "total_tokens": 4549,
        "input_token_details": {
          "audio": 0,
          "cache_read": 0
        },
        "output_token_details": {
          "audio": 0,
          "reasoning": 0
        }
      },
      "id": "lc_run--019ba354-5b07-7721-b724-1872a1ce85ba-0"
    }
  ]
}

"""