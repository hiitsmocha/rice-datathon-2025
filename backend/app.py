import os
from dotenv import load_dotenv
import pymongo

load_dotenv()

mongo_url = os.getenv("MONGO_URI")
hf_token = os.getenv("HF_TOKEN")  
embedding_url = os.getenv("EMBEDDING_URL")

# Connect to MongoDB
client = pymongo.MongoClient(mongo_url)
db = client['emp_data']
collection = db['emp']
