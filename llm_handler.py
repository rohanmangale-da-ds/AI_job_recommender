# src/llm_handler.py
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv()
groq_llama3_key = os.getenv("groq_llama3_key")

# Initialize Groq client
client = Groq(api_key=groq_llama3_key)

def get_llm_response(user_messages):
    """
    Generates a response from the Llama 3 model based on user input.
    `user_messages` should be a list of dicts with 'role' and 'content'.
    """
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=user_messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    print(get_llm_response(test_messages))
