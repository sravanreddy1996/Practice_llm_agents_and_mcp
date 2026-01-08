import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
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

llm = ChatOpenAI(temperature=0, model='gpt-5-mini') # Underlying uses OpenAI SDK to make llm call.
chain = summary_prompt_template | llm 
response = chain.invoke(input={"information": text}) # input: Prompt fillers input_variables' values.


print("Response from LLM:")
print(response) ## Type: AIMessage -- Wrapper class of what LLm wants to reply us
# response.content ---> Answer that LLM generated
# AIMessage object contains other info like tool calling, how many tokens did we consume, how much did it cost... etc.
"""
Here's the structure of AIMessage response converted to key-value pairs:

**Main Keys:**

- **content**: "Short summary\nThe Eastern Roman Empire..." (the actual LLM-generated summary text)
- **additional_kwargs**: {'refusal': None}
- **response_metadata**: Contains model and token details
- **usage_metadata**: Token usage breakdown

**Nested under response_metadata:**
- **token_usage**: 
  - completion_tokens: 728
  - prompt_tokens: 797
  - total_tokens: 1525
  - completion_tokens_details: {accepted_prediction_tokens, audio_tokens, reasoning_tokens, rejected_prediction_tokens}
  - prompt_tokens_details: {audio_tokens, cached_tokens}
- **model_provider**: "openai"
- **model_name**: "gpt-5-mini-2025-08-07"
- **system_fingerprint**: None
- **id**: "chatcmpl-CvdRIS8e58F0zNbxLgYvVycLN9gfa"
- **service_tier**: "default"
- **finish_reason**: "stop"
- **logprobs**: None

**Nested under usage_metadata:**
- **input_tokens**: 797
- **output_tokens**: 728
- **total_tokens**: 1525
- **input_token_details**: {audio: 0, cache_read: 0}
- **output_token_details**: {audio: 0, reasoning: 512}

**Additional top-level keys:**
- **id**: "lc_run--019b9c35-e757-7ad2-9005-f0528bacac83-0"
- **tool_calls**: []
- **invalid_tool_calls**: []
"""
# Write response to output file
with open(response_path, 'w', encoding='utf-8') as output_file:
    output_file.write(str(response))

print(f"\nSummary written to: {response_path}")