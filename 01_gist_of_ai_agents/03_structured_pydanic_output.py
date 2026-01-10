import os 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool # Build a toolkit of multiple tools
from langchain_core.messages import HumanMessage # We will be using HumanMessage to invoke the Agent. So this is going to be the input for the Agent's execution.
from langchain_openai import ChatOpenAI # This supplies the LLM for our Agent
from langchain_tavily import TavilySearch # Langchain tools related to Tavily Web Search implemented by Tavily written in Langchain.

from pydantic import BaseModel, Field # Inherit BaseModel to define schema --> Data Parsing, Serialization, Automatic Type Validations.
                                        # Field class is used to add metadata to attributes --> Add descriptions (LLMs use this to understand what to put in this field based on provided description)

from typing import List


"""
AGENT = LLM + ToolKit
Toolkit = list of Tools
Tool = any function that agent can execute ==> We are specifying the ACTIONs to perform 
        - Function with docstrings, typings, arguments it receive, what output.
"""

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(description="List of sources used to generate the answer",
                                  default_factory=list) # default Agent will return with empty sources list if not relevant.


# Print the constructed path for debugging
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
print(f"Looking for .env at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")



# Load environment variables from .env file in project root
load_dotenv(env_path)


llm = ChatOpenAI(name='gpt-5-mini')

toolkit = [TavilySearch()]

agent = create_agent(model=llm, tools=toolkit, response_format=AgentResponse)



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
## Agent Job Search with Structured Pydantic Output

```
Agent Response Tree (with AgentResponse Schema)
│
├── 1️⃣ User Input
│   └── 💬 HumanMessage
│       ├── Content: "search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details"
│       └── ID: 0a2053cc-2e60-4542-b9cc-94ca0556731e
│
├── 2️⃣ Agent Decision Making
│   └── 🤖 AIMessage (Multiple Tool Calls - Advanced Search)
│       ├── Finish Reason: tool_calls
│       ├── Model: gpt-3.5-turbo-0125
│       ├── 📊 Token Usage
│       │   ├── Input: 1345
│       │   ├── Output: 141
│       │   └── Total: 1486
│       └── 🔧 Tool Calls (3 Parallel Searches)
│           ├── Call #1: tavily_search
│           │   ├── Query: "AI Engineer job posting Langchain India site:linkedin.com"
│           │   ├── Search Depth: advanced
│           │   ├── Time Range: month
│           │   ├── Include Domains: linkedin.com
│           │   └── Call ID: call_cJKN8oGKhXH3koeY3a0zVLDA
│           ├── Call #2: tavily_search (Same params)
│           │   └── Call ID: call_CUU3vgipxYzX7UMpcTddijGt
│           └── Call #3: tavily_search (Same params)
│               └── Call ID: call_kFXKCljz8AHrWy8j8BWmsSgo
│
├── 3️⃣ Tool Execution
│   └── 🛠️ ToolMessage (LinkedIn Job Search Results)
│       ├── Query: "AI Engineer job posting Langchain India"
│       ├── Results Count: 5
│       │
│       └── 📋 Top 3 Job Postings
│           │
│           ├── 🏢 Job #1: Agentic AI / LLM Engineer
│           │   ├── Company: Meraki TalentWorks
│           │   ├── Location: PAN India
│           │   ├── Experience: 8–12 Years (Hands-on)
│           │   ├── Joining: Immediate (within 15 days)
│           │   ├── Project: High-impact global project with Mastercard
│           │   ├── Key Responsibilities:
│           │   │   ├── Design & build agentic AI pipelines
│           │   │   └── Deploy LLM & RAG solutions
│           │   ├── Mandatory Skills:
│           │   │   ├── Agentic AI: LangGraph, CrewAI, AutoGen, LangChain, LlamaIndex
│           │   │   ├── Python: Async programming, FastAPI, design patterns
│           │   │   ├── LLM & RAG: Prompt engineering, embeddings
│           │   │   └── ML & Statistics
│           │   ├── Relevance Score: 0.9999031
│           │   └── URL: https://www.linkedin.com/posts/meraki-talentworks_hiring-hiringnow-jobs-activity-7407355209200570368-DGE_
│           │
│           ├── 🏢 Job #2: AI Engineer
│           │   ├── Company: Zycus
│           │   ├── Location: Mumbai, Maharashtra, India
│           │   ├── Experience: 2–4 Years
│           │   ├── Required Skills:
│           │   │   ├── Business analysis or support ops experience
│           │   │   ├── REST APIs, JSON, data processing
│           │   │   └── Strong English communication
│           │   ├── Ideal Skills:
│           │   │   ├── LangChain, LlamaIndex, Pinecone experience
│           │   │   ├── OpenAI Assistants
│           │   │   └── SaaS post-sales operations
│           │   ├── Posted: 1 day ago
│           │   ├── Relevance Score: 0.99986017
│           │   └── URL: https://in.linkedin.com/jobs/view/ai-engineer-at-zycus-4358394415
│           │
│           └── 🏢 Job #3: Gen AI Engineer (5+ yrs)
│               ├── Company: MindBrain
│               ├── Location: India (Remote)
│               ├── Experience: 5+ Years
│               ├── Employment Type: Contract
│               ├── Seniority Level: Mid-Senior
│               ├── Required Skills:
│               │   ├── Professional software development (4+ yrs)
│               │   ├── Strong Python proficiency
│               │   ├── Vertex AI (hands-on)
│               │   ├── LangChain (practical experience)
│               │   ├── RAG (Retrieval-Augmented Generation)
│               │   ├── GCP API deployment
│               │   ├── Vector databases & embedding techniques
│               │   └── RESTful APIs & microservices
│               ├── Preferred Skills:
│               │   ├── BigQuery, Cloud Storage, Firestore
│               │   ├── Kubernetes, Docker, Cloud Run
│               │   ├── MLOps & model lifecycle management
│               │   └── AI security & governance
│               ├── Relevance Score: 0.9995728
│               └── URL: https://in.linkedin.com/jobs/view/gen-ai-engineer-5%2B-yrs-at-mindbrain-4330887706
│
└── 4️⃣ Final Response (Structured Pydantic Format)
    └── 🤖 AIMessage (AgentResponse Schema)
        ├── Response Schema: AgentResponse (Pydantic Model)
        │   ├── answer (str): Summarized job details
        │   └── sources (List[Source]): LinkedIn URLs
        ├── Finish Reason: stop
        ├── Model: gpt-3.5-turbo-0125
        ├── 📊 Token Usage
        │   ├── Input: 4085
        │   ├── Output: 464
        │   └── Total: 4549
        └── ID: lc_run--019ba3ad-9962-75e0-bcfc-ca859d27fcc7-0

📈 Summary
├── Total Tokens Used: 6035
├── Status: ✅ Completed Successfully
├── Tool Used: Tavily Search (Advanced Depth, Monthly Filter)
├── Model: gpt-3.5-turbo-0125
├── Search Strategy: 3 Parallel Queries for comprehensive results
├── Jobs Found: 5 (Top 3 selected)
├── Locations: PAN India, Mumbai, Remote
├── Experience Range: 2-4 years to 5+ years
├── Key Technologies: LangChain, Python, RAG, Vertex AI, GCP, FastAPI
├── Seniority Levels: Entry to Senior
│
└── 📝 Output Format
    ├── Structure: Pydantic BaseModel (AgentResponse)
    ├── Fields:
    │   ├── answer: Natural language summary with job details
    │   └── sources: List of Source objects containing URLs
    ├── Benefits:
    │   ├── ✅ Type-safe data structure
    │   ├── ✅ Automatic validation
    │   ├── ✅ Serialization support (JSON)
    │   └── ✅ LLM-friendly schema descriptions
    └── Default: Empty sources list if none found
```
========================== Raw response ==================================
{
  "messages": [
    {
      "type": "HumanMessage",
      "content": "search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details",
      "id": "0a2053cc-2e60-4542-b9cc-94ca0556731e"
    },
    {
      "type": "AIMessage",
      "content": "",
      "finish_reason": "tool_calls",
      "model": "gpt-3.5-turbo-0125",
      "tokens": {
        "input": 1345,
        "output": 141,
        "total": 1486
      },
      "tool_calls": [
        {
          "name": "tavily_search",
          "args": {
            "query": "AI Engineer job posting Langchain India site:linkedin.com",
            "search_depth": "advanced",
            "include_domains": ["linkedin.com"],
            "time_range": "month"
          },
          "id": "call_cJKN8oGKhXH3koeY3a0zVLDA"
        },
        {
          "name": "tavily_search",
          "args": {
            "query": "AI Engineer job posting Langchain India site:linkedin.com",
            "search_depth": "advanced",
            "include_domains": ["linkedin.com"],
            "time_range": "month"
          },
          "id": "call_CUU3vgipxYzX7UMpcTddijGt"
        },
        {
          "name": "tavily_search",
          "args": {
            "query": "AI Engineer job posting Langchain India site:linkedin.com",
            "search_depth": "advanced",
            "include_domains": ["linkedin.com"],
            "time_range": "month"
          },
          "id": "call_kFXKCljz8AHrWy8j8BWmsSgo"
        }
      ],
      "id": "lc_run--019ba3ad-9962-75e0-bcfc-ca859d27fcc7-0"
    },
    {
      "type": "ToolMessage",
      "tool_name": "tavily_search",
      "content": {
        "query": "AI Engineer job posting Langchain India",
        "results": [
          {
            "rank": 1,
            "title": "Meraki TalentWorks - Agentic AI / LLM Engineer",
            "url": "https://www.linkedin.com/posts/meraki-talentworks_hiring-hiringnow-jobs-activity-7407355209200570368-DGE_",
            "company": "Meraki TalentWorks",
            "location": "PAN India",
            "experience": "8–12 Years (Hands-on)",
            "joining_timeline": "Immediate (within 15 days)",
            "project": "High-impact global project with Mastercard",
            "salary": "Not specified",
            "key_responsibilities": [
              "Design, build, and optimize agentic AI pipelines using LangGraph, CrewAI, AutoGen, and LangChain",
              "Develop, fine-tune, and deploy LLM & RAG-based solutions"
            ],
            "mandatory_skills": {
              "agentic_ai": ["LangGraph", "CrewAI", "AutoGen", "LangChain", "LlamaIndex"],
              "python": ["Async programming", "FastAPI", "design patterns"],
              "llm_rag": ["Prompt engineering", "embeddings", "memory/state management", "LoRA/PEFT"],
              "ml_statistics": "Required"
            },
            "relevance_score": 0.9999031
          },
          {
            "rank": 2,
            "title": "Zycus - AI Engineer",
            "url": "https://in.linkedin.com/jobs/view/ai-engineer-at-zycus-4358394415",
            "company": "Zycus",
            "location": "Mumbai, Maharashtra, India",
            "experience": "2–4 Years",
            "posted": "1 day ago",
            "employment_type": "Full-time",
            "required_skills": [
              "Business analysis or support ops experience",
              "REST APIs, JSON, and basic data processing",
              "Strong English communication"
            ],
            "ideal_skills": [
              "LangChain, LlamaIndex, Pinecone",
              "OpenAI Assistants",
              "SaaS post-sales operations",
              "ITIL/CSM methodologies"
            ],
            "nice_to_have": [
              "Customer success experience",
              "Implementation experience",
              "Technical support background"
            ],
            "relevance_score": 0.99986017
          },
          {
            "rank": 3,
            "title": "MindBrain - Gen AI Engineer (5+ yrs)",
            "url": "https://in.linkedin.com/jobs/view/gen-ai-engineer-5%2B-yrs-at-mindbrain-4330887706",
            "company": "MindBrain",
            "location": "India (Remote)",
            "experience": "5+ Years",
            "employment_type": "Contract",
            "seniority_level": "Mid-Senior",
            "required_skills": {
              "software_development": "4+ years professional experience",
              "programming": "Strong Python proficiency",
              "cloud": "Hands-on Vertex AI",
              "frameworks": "Practical LangChain experience",
              "ai_techniques": "RAG (Retrieval-Augmented Generation)",
              "deployment": "GCP API deployment",
              "databases": "Vector databases and embedding techniques",
              "apis": "RESTful APIs and microservices"
            },
            "preferred_skills": [
              "BigQuery, Cloud Storage, Firestore",
              "Kubernetes, Docker, Cloud Run",
              "MLOps and model lifecycle management",
              "AI security and governance"
            ],
            "relevance_score": 0.9995728
          },
          {
            "rank": 4,
            "title": "Metaphor Services - AI Engineer",
            "url": "https://in.linkedin.com/jobs/view/ai-engineer-at-metaphor-services-4330761244",
            "company": "Metaphor Services",
            "location": "Bengaluru, Karnataka, India",
            "experience": "5+ Years backend, 1-2 Years AI",
            "employment_type": "Contract (Hybrid)",
            "duration": "1+ year",
            "seniority_level": "Mid-Senior",
            "tech_stack": ["Python", "FastAPI", "LangChain"],
            "key_requirements": [
              "AI & LLM Integration (prompting, orchestration, model evaluation)",
              "Python (production-level backend + ML workflows)",
              "LangGraph & LangChain (agent workflows, tools, memory)",
              "FastAPI (robust backend services, async execution)"
            ],
            "applicants": 27,
            "relevance_score": 0.99923277
          },
          {
            "rank": 5,
            "title": "S&P Global - Lead AI Engineer (Agentic Systems)",
            "url": "https://in.linkedin.com/jobs/view/lead-ai-engineer-agentic-systems-at-s-p-global-4327606620",
            "company": "S&P Global",
            "location": "Gurugram, Haryana, India",
            "employment_type": "Full-time",
            "seniority_level": "Senior/Lead",
            "posted": "3 weeks ago",
            "applicants": 28,
            "focus": "Agentic Systems"
          }
        ]
      },
      "id": "ToolMessage-001"
    },
    {
      "type": "AIMessage",
      "content": "I found several excellent AI Engineer job postings with LangChain expertise in India. Here are the top 3 positions with detailed information...",
      "finish_reason": "stop",
      "model": "gpt-3.5-turbo-0125",
      "tokens": {
        "input": 4085,
        "output": 464,
        "total": 4549
      },
      "id": "lc_run--019ba3ad-9962-75e0-bcfc-ca859d27fcc7-final"
    }
  ],
  "summary": {
    "total_tokens_used": 6035,
    "status": "completed",
    "model": "gpt-3.5-turbo-0125",
    "search_strategy": "3 parallel Tavily searches (advanced depth, monthly filter)",
    "jobs_found": 5,
    "top_3_selected": true,
    "locations": ["PAN India", "Mumbai", "Bengaluru", "Remote"],
    "experience_range": "2-12+ years",
    "key_technologies": [
      "LangChain",
      "LangGraph",
      "Python",
      "RAG",
      "Vertex AI",
      "FastAPI",
      "CrewAI",
      "AutoGen"
    ],
    "seniority_levels": ["Entry", "Mid", "Senior/Lead"],
    "employment_types": ["Full-time", "Contract", "Hybrid"]
  }
}
"""