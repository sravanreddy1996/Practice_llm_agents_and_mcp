"""
Docstring for 02_original_langchain_ReAct_agent.00_main
- this method of react agent creation is not used currently.
"""
import os 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool # Build a toolkit of multiple tools
from langchain_core.messages import HumanMessage # We will be using HumanMessage to invoke the Agent. So this is going to be the input for the Agent's execution.
from langchain_openai import ChatOpenAI # This supplies the LLM for our Agent
from langchain_classic import hub # Hub is designed for sharing and exploring prompts, chains and agents created by the community.
                            # It serves as developer's resource for writing prompts

# Langsmith > Prompts saver > Hub ----> Browse and look into hub resources.


from langchain_classic.agents.react.agent import create_react_agent # Built-in function to create a Runnable. It is a component of Chain. It is used for building Reasoning engine.
                                                                    # Overall a special prompt is sent to LLM to make it a Reasoning engine. LLM does Reasoning.

from langchain_classic.agents import AgentExecutor # AgentExecutor is the Runtime for React Agent. 
                                                    # If the reasoning engine is going to tell us what we need to run, if we need to call the search tool with which arguments, Then the Agent Executor is going to make the actual calls. So its going to be the runtime.
                                                    # AE is going to be the thing that is going to execute everything. ACTING (call tool)
                                                    # In simplest understanding, it is just a LOOP.

from langchain_tavily import TavilySearch # Tavily team wrapped tavily tools into Langchain.

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

llm = ChatOpenAI(model='gpt-4')
toolkit = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = toolkit,
    prompt = react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True) # Agent Runtime
# A continuous While LOOP where each time, it selects action (tool), does action (tool call), with output it runs next iteration if answwer not yet found, else FINISH the loop.

chain = agent_executor

def main():
    print("Hi")

    result = chain.invoke(
        input = {
            "input": "search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details"
        }
    )
    # Input will be pulled and given to the Agent (LLM) as 1st HumanMessage.
    print(result)

if __name__ == '__main__':
    main()

    """
AgentExecutor
├── User Query
│   └── "search for job postings for an AI engineer using LangChain in India on LinkedIn"
│
├── Thought
│   └── Need to use tavily_search to find relevant LinkedIn job postings
│
├── Action
│   ├── Tool: tavily_search
│   └── Action Input
│       ├── query
│       │   └── "AI engineer langchain job postings india linkedin"
│       ├── follow_up_questions: None
│       ├── answer: None
│       └── images: []
│
├── Tool Response (tavily_search)
│   ├── response_time: 0.66s
│   ├── request_id: be7489aa-345f-4a5b-88a0-328c11076667
│   └── results
│       ├── Result 1
│       │   ├── Title: "1,000+ Generative AI Engineer jobs in India"
│       │   ├── URL: in.linkedin.com/jobs/generative-ai-engineer-jobs
│       │   └── Score: 0.9999739
│       │
│       ├── Result 2
│       │   ├── Title: "AI Engineer - GPT / LangChain / RAG / Data Pipelines"
│       │   ├── Company: Peak Trust Global Real Estate
│       │   ├── URL: in.linkedin.com/jobs/view/ai-engineer-gpt-langchain-rag-data-pipelines-at-peak-trust-4302171754
│       │   └── Score: 0.99996495
│       │
│       ├── Result 3
│       │   ├── Title: "AI/ML Engineer (LangChain + RAG + Multi-Agent)"
│       │   ├── Company: CloudZent Technology Services
│       │   ├── URL: in.linkedin.com/jobs/view/ai-ml-engineer-langchain-%2B-rag-%2B-multi-agent-at-cloudzent-technology-services-4328123821
│       │   └── Score: 0.99995995
│       │
│       ├── Result 4 (Ignored)
│       │   ├── Title: "Senior AI Engineer (India)"
│       │   └── Reason: Not LangChain-specific
│       │
│       └── Result 5
│           ├── Title: "Agentic AI Engineer | Google ADK, LangChain & Langraph"
│           ├── Company: AI-Data Value Info Com-Tech Alliance
│           ├── URL: in.linkedin.com/jobs/view/agentic-ai-engineer-google-adk-langchain-langraph-at-ai-data-value-info-com-tech-alliance-4323204373
│           └── Score: 0.9998876
│
├── Observation
│   └── Relevant job listings identified (LangChain-specific)
│
├── Final Output
│   └── Job Postings
│       ├── 1. AI Engineer - GPT / LangChain / RAG / Data Pipelines
│       │   ├── Company: Peak Trust Global Real Estate
│       │   └── Link: https://in.linkedin.com/jobs/view/ai-engineer-gpt-langchain-rag-data-pipelines-at-peak-trust-4302171754
│       │
│       ├── 2. AI/ML Engineer (LangChain + RAG + Multi-Agent)
│       │   ├── Company: CloudZent Technology Services
│       │   └── Link: https://in.linkedin.com/jobs/view/ai-ml-engineer-langchain-%2B-rag-%2B-multi-agent-at-cloudzent-technology-services-4328123821
│       │
│       └── 3. Agentic AI Engineer | Google ADK, LangChain & Langraph
│           ├── Company: AI-Data Value Info Com-Tech Alliance
│           └── Link: https://in.linkedin.com/jobs/view/agentic-ai-engineer-google-adk-langchain-langraph-at-ai-data-value-info-com-tech-alliance-4323204373


======================================================= Final result ================================================================
root
├── input
│   └── "search for 3 job postings for an ai engineer using langchain in india on linkedin and list their details"
└── output
    ├── Job Postings
    │   ├── 1. AI Engineer - GPT / LangChain / RAG / Data Pipelines
    │   │   ├── Company: Peak Trust Global Real Estate
    │   │   ├── Location: India
    │   │   └── Link: https://in.linkedin.com/jobs/view/ai-engineer-gpt-langchain-rag-data-pipelines-at-peak-trust-4302171754
    │   ├── 2. AI/ML Engineer (LangChain + RAG + Multi-Agent)
    │   │   ├── Company: CloudZent Technology Services
    │   │   └── Link: https://in.linkedin.com/jobs/view/ai-ml-engineer-langchain-%2B-rag-%2B-multi-agent-at-cloudzent-technology-services-4328123821
    │   └── 3. Agentic AI Engineer | Google ADK, LangChain & Langraph
    │       ├── Company: AI-Data Value Info Com-Tech Alliance
    │       └── Link: https://in.linkedin.com/jobs/view/agentic-ai-engineer-google-adk-langchain-langraph-at-ai-data-value-info-com-tech-alliance-4323204373

    """