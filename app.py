import os
import requests
import streamlit as st
from dotenv import load_dotenv
import time

load_dotenv()

API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("HF_MODEL")

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def response_generator(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

total_tokens = 0
CONTEXT_LIMIT = 32768

# Streamlit User Interface
st.title("Hugging Face AI Chat App")
st.write("Model:", MODEL_NAME)

with st.sidebar:
    usage_percent = 100 - int((total_tokens / CONTEXT_LIMIT) * 100)
    token_bar = st.progress(usage_percent, text="Token Usage")
    st.write(f"Tokens left: {usage_percent}%")

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Prompt input template
prompt = st.chat_input('Type your prompt here')

if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role':'user', 'content':prompt})

    # Load Hugging Face model
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user", "content": prompt
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    answer = result["choices"][0]["message"]["content"]

    # Show the LLM response
    with st.chat_message('assistant'):
        st.write_stream(response_generator(answer))
    # Store the LLM response in state
    st.session_state.messages.append({'role':'assistant', 'content':answer})

    # Check token usage
    usage = result.get("usage", {})
    total_tokens = usage.get("total_tokens")
