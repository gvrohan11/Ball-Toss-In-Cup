"""Teammate 3: log every attempt to MongoDB Atlas.

log_attempt(doc) - doc has ball, cup, release_pose, hit (bool), ...
Until MONGODB_URI is set it just prints, so the main loop works without it.
"""
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def log_attempt(doc):
    doc = {**doc, "timestamp": datetime.now(timezone.utc)}
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("  [log, no MONGODB_URI yet]", doc)
        return
    # TODO: pip install pymongo, then:
    # from pymongo import MongoClient
    # MongoClient(uri)["balltoss"]["attempts"].insert_one(doc)
    raise NotImplementedError("wire up pymongo here")
