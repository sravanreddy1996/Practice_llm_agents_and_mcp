import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Print the constructed path for debugging
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
print(f"Looking for .env at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")



# Load environment variables from .env file in project root
load_dotenv(env_path)

text_path = os.path.join(os.path.dirname(__file__), "..", "data", "00_text_on_roman_empire.txt")
response_path = os.path.join(os.path.dirname(__file__), "..", "data", "00_result_summary_on_roman_empire.txt")
text = open(text_path, 'r').read()


# Prompt template
summary_template = """
Given the information {information} about a topic/person I want to you to create:
1. A short summary
2. two interesting facts from the provided content.
"""

## Build a prompt template from above string template
summary_prompt_template = PromptTemplate(
    input_variables=["information"],
    template=summary_template,
)


print(summary_prompt_template)

# llm = ChatOpenAI(temperature=0, model='gpt-5-mini') # Underlying uses OpenAI SDK to make llm call.
llm = ChatOllama(temperature=0, model='gemma3:270m')
chain = summary_prompt_template | llm 
response = chain.invoke(input={"information": text}) # input: Prompt fillers input_variables' values.


print("Response from LLM:")
print(response) ## Type: AIMessage -- Wrapper class of what LLm wants to reply us
# response.content ---> Answer that LLM generated
# AIMessage object contains other info like tool calling, how many tokens did we consume, how much did it cost... etc.
"""
Here's the structure of AIMessage response converted to key-value pairs:

response_dict = {
    "content": "Here's a summary of the provided information about the Roman Empire:\n\n**Summary:**\n\nThe Roman Empire, a vast and influential empire spanning from the Roman Republic to the fall of Constantinople in 1453, was characterized by its remarkable stability and prosperity. Its vast territories were organized into senatorial provinces, governed by proconsuls, and imperial provinces, which were governed by legates. The empire's rise was marked by periods of unprecedented stability and prosperity, culminating in the Pax Romana (Roman Peace). The empire's institutions and culture had a lasting impact on the development of language, religion, art, architecture, literature, philosophy, law, and forms of government across its territories. Latin evolved into the Romance languages, while Medieval Greek became the language of the East. The Empire's adoption of Christianity resulted in the formation of medieval Christendom. Roman and Greek art had a profound impact on the Italian Renaissance. The rediscovery of classical science and technology contributed",
    "response_metadata": {
        "total_duration": 2664053900,
        "load_duration": 95487400,
        "prompt_eval_count": 837,
        "prompt_eval_duration": 236493200,
        "eval_count": 250,
        "eval_duration": 2171769100,
        "logprobs": None,
        "model_name": "gemma3:270m",
        "model_provider": "ollama"
    },
    "id": "lc_run--019b9d06-4780-7bf3-ad97-85524328934e-0",
    "tool_calls": [],
    "invalid_tool_calls": [],
    "usage_metadata": {
        "input_tokens": 837,
        "output_tokens": 250,
        "total_tokens": 1087
    }
}
"""
# Write response to output file
with open(response_path, 'w', encoding='utf-8') as output_file:
    output_file.write(str(response))

print(f"\nSummary written to: {response_path}")