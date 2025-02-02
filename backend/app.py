from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import re

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

import re

def extract_filters(natural_language_query):
    """Extracts filters like year, fuel type, and vehicle category from the query."""
    
    # Define regex patterns to extract numbers (years) and keywords
    year_match = re.search(r"\b(20\d{2})\b", natural_language_query)
    fuel_match = re.search(r"\b(gasoline|diesel|electric|hybrid)\b", natural_language_query, re.IGNORECASE)
    category_match = re.search(r"\b(car|truck|bus|motorcycle|suv)\b", natural_language_query, re.IGNORECASE)

    year = year_match.group(1) if year_match else None
    fuel_type = fuel_match.group(1).capitalize() if fuel_match else None
    category = category_match.group(1).capitalize() if category_match else None

    return {"year": year, "fuel_type": fuel_type, "category": category}

def query_database(user_id, natural_language_query):
    """Filters dataset based on extracted query parameters."""
    
    global conversation_context
    previous_context = conversation_context.get(user_id, "")

    # Load local data
    data = load_local_data()
    
    # Extract filters from the query
    filters = extract_filters(natural_language_query)
    year, fuel_type, category = filters["year"], filters["fuel_type"], filters["category"]

    # Apply filtering
    relevant_data = [
        entry for entry in data
        if (not year or str(entry["Date"]) == year)
        and (not fuel_type or entry["Fuel Type"].lower() == fuel_type.lower())
        and (not category or entry["Vehicle Category"].lower() == category.lower())
    ]

    if not relevant_data:
        return "I couldn't find relevant data matching your query."

    # Summarize filtered data
    data_summary = "\n".join([str(entry) for entry in relevant_data[:5]])  # Limit results

    # Ask Gemini to analyze the data
    analysis_prompt = f"""
    You are analyzing vehicle population data. Here is the relevant data:
    {data_summary}

    Based on this data, generate a detailed response to the user's question: '{natural_language_query}'
    """
    analysis_response = genai.ChatModel("gemini-pro").generate_content(analysis_prompt)

    # Save context
    conversation_context[user_id] = natural_language_query

    return analysis_response.text.strip()



@app.post("/query")
async def chatbot_query(request: QueryRequest):
    """Handles chatbot queries"""
    return {"response": query_database(request.user_id, request.query)}
