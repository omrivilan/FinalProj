from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os

# Load from env or hardcode (NOT recommended for production)
MONGO_URI = "mongodb+srv://idanhefe:idan1234@roboadvisor.omthsqt.mongodb.net/?retryWrites=true&w=majority&appName=roboadvisor"
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Connected to MongoDB Atlas!")
except Exception as e:
    print("Failed to connect:", e)

# Use your desired database and collection
db = client["roboadvisor"]
users_col = db["users"]
reset_tokens_col = db["reset_tokens"]
users_chat_col   = db["users_chat"]
