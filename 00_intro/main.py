import os

from dotenv import load_dotenv

# Print the constructed path for debugging
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
print(f"Looking for .env at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")


# Load environment variables from .env file in project root
load_dotenv(env_path)


# Access your API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
print(openai_api_key)
print(google_api_key)


def main():
    print("hello from langchain")


if __name__ == "__main__":
    main()
