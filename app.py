from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


response = client.embeddings.create(
    input="Your text string goes here",
    model="gemini-embedding-001"
)

print(response.data[0].embedding)

messages=[{"role": "system", "content": "You are a helpful assistant."}]

def chat_with_gemini(query):
    # sliding window to limit context length
    MAX_HISTORY_TURNS = 10 # Keep the last 10 pairs of (user, assistant) messages
    MAX_MESSAGES = MAX_HISTORY_TURNS * 2

    if len(messages) > MAX_MESSAGES:
        # Keep the system prompt plus the most recent messages
        messages = [messages[0]] + messages[-MAX_MESSAGES:]

    messages.append({"role": "user", "content": query})
    chat = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        reasoning_effort="medium",
    )
    response = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    return response


print(chat_with_gemini("Briefly explain to me how AI works"))
print(chat_with_gemini("How does it differ from machine learning?"))
print(chat_with_gemini("summarize each one in one sentence"))
