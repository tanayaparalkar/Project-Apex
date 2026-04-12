import os
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

# Get from environment variable
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("❌ MONGO_URI not set in environment variables")

client = MongoClient(MONGO_URI)

db = client["eventoracle"]

collections = [
    "prices",
    "news",
    "events",
    "flows",
    "features",
    "labels",
    "predictions"
]

for col in collections:
    if col not in db.list_collection_names():
        db.create_collection(col)
        print(f"✅ Created collection: {col}")
    else:
        print(f"⚠️ Already exists: {col}")

print("🎉 Database setup complete!")
