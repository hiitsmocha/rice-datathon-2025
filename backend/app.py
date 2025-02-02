from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# Load local dataset
DATA_FILE = "vehicle_data.json"

def load_local_data():
    """Reads vehicle population data from local JSON file."""
    with open(DATA_FILE, "r") as f:
        return json.load(f)

class QueryRequest(BaseModel):
    user_id: str
    query: str

conversation_context = {}

def query_database(user_id, natural_language_query):
    """Searches local dataset, sends results to Gemini for analysis."""
    
    global conversation_context
    previous_context = conversation_context.get(user_id, "")

    # Load local data
    data = load_local_data()

    # Simulate filtering (Replace with MongoDB query later)
    relevant_data = [entry for entry in data if str(entry).lower() in natural_language_query.lower()]

    if not relevant_data:
        return "I couldn't find relevant data in the local dataset."

    # Convert filtered data into a summary
    data_summary = "\n".join([str(entry) for entry in relevant_data])

    # Ask Gemini to analyze the data
    analysis_prompt = f"""
    You are analyzing vehicle population data. Here is the relevant data:
    {data_summary}

    Based on this data, generate a detailed response to the user's question: '{natural_language_query}'
    """
    analysis_response = genai.ChatModel("gemini-pro").generate_content(analysis_prompt)

    # Save conversation context for follow-ups
    conversation_context[user_id] = natural_language_query

    return analysis_response.text.strip()

@app.post("/query")
async def chatbot_query(request: QueryRequest):
    """Handles chatbot queries"""
    return {"response": query_database(request.user_id, request.query)}
