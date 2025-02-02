import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.title("🚗 AI Chatbot for Vehicle Data Analysis")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.text_input("Ask about vehicle population trends:", key="user_input")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    response = requests.post(API_URL, json={"user_id": "123", "query": user_input})
    
    if response.status_code == 200:
        bot_reply = response.json()["response"]
    else:
        bot_reply = "Error: Unable to fetch response."

    st.session_state["messages"].append({"role": "bot", "content": bot_reply})

    with st.chat_message("bot"):
        st.markdown(bot_reply)
