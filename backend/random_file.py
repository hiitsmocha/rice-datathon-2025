import pandas as pd

df = pd.read_csv("/Users/nam/Desktop/git/rice-datathon/rice-datathon-2025/backend/scoring.csv")
df.to_json("vehicle_data.json", orient = "records", indent = 4)


import google.generativeai as genai

genai.configure(api_key="AIzaSyDUZVf2L1nDjQ2Z9iLM12EWNROQnjuX-qU")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("")
print(response.text)


