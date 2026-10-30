"""MongoDB teammate: log every attempt to MongoDB Atlas.

    from logger import log_attempt, recent_attempts, suggest_offset
    log_attempt({"phase": "drop", "ball": {...}, "cup": {...}, "release": {...},
                 "result": {"hit": True}})

Setup:
    pip install pymongo
    put MONGODB_URI=mongodb+srv://... in .env   (the event's Atlas cluster)

Without MONGODB_URI (or if Atlas is down) each attempt is saved as readable JSON instead
(observations/detections/<attempt>/attempt.json, or observations/attempts/), so the robot
loop never breaks because of logging.
Schema: see TASKS.md (collection balltoss.attempts).

    python logger.py            # insert one test doc and print the stats
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("MONGODB_DB", "balltoss")
COLLECTION = "attempts"
SESSION = os.getenv("BALLTOSS_SESSION", datetime.now().strftime("%a-%H%M").lower())

_collection = None


def _get_collection():
    """Lazily connect once. Returns None if there's no URI or the connection fails."""
    global _collection
    if _collection is not None:
        return _collection
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None
    try:
        from pymongo import DESCENDING, MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        col = client[DB_NAME][COLLECTION]
        col.create_index([("timestamp", DESCENDING)])
        col.create_index("session")
        _collection = col
    except Exception as e:  # never let logging crash the robot
        print(f"  [log] MongoDB unavailable ({str(e)[:80]}); saving locally instead")
        return None
    return _collection


def _jsonable(v):
    """Tuples -> lists, numpy floats -> float, int keys -> str, so pymongo can store them."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "item"):
        return v.item()
    return v


LOCAL_DIR = Path(__file__).resolve().parent / "observations" / "attempts"


def _readable(v):
    """For the local JSON copy: binary thumbnails -> a size note, floats rounded."""
    if isinstance(v, dict):
        return {k: _readable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_readable(x) for x in v]
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes jpeg>"
    if isinstance(v, float):
        return round(v, 1)
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def save_local(doc):
    """Write the attempt as readable JSON: into its detection-log folder when it has one,
    else observations/attempts/. Returns the path, or None if it can't be written."""
    try:
        folder = doc.get("detection_log")
        if folder and Path(folder).is_dir():
            path = Path(folder) / "attempt.json"
        else:
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            path = LOCAL_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{doc['attempt_id'][:8]}.json"
        path.write_text(json.dumps(_readable(doc), indent=2, default=str) + "\n")
        return path
    except Exception as e:
        print(f"  [log] couldn't save locally either ({e})")
        return None


def log_attempt(doc):
    """Store one attempt. Adds attempt_id, timestamp and session. Returns the stored doc.
    Without Atlas (no MONGODB_URI, or it's down) the attempt is saved as a local JSON file."""
    doc = _jsonable(dict(doc))
    doc.setdefault("attempt_id", str(uuid.uuid4()))
    doc.setdefault("timestamp", datetime.now(timezone.utc))
    doc.setdefault("session", SESSION)
    hit = doc.get("result", {}).get("hit")
    col = _get_collection()
    if col is not None:
        try:
            col.insert_one(dict(doc))
            print(f"  [log] saved attempt {doc['attempt_id'][:8]} to Atlas, hit={hit}")
            return doc
        except Exception as e:
            print(f"  [log] Atlas insert failed ({str(e)[:80]})")
    path = save_local(doc)
    if path:
        why = "no MONGODB_URI" if not os.getenv("MONGODB_URI") else "Atlas unavailable"
        print(f"  [log] {why}: saved attempt locally, hit={hit} -> {path}")
    return doc


def recent_attempts(n=20, session=None):
    col = _get_collection()
    if col is None:
        return []
    q = {"session": session} if session else {}
    return list(col.find(q, {"_id": 0}).sort("timestamp", -1).limit(n))


def hit_rate(session=None):
    """(hits, total) over attempts that reached the drop and recorded a result.
    Failures before the drop (no ball, missed pick, no cup) are not counted here."""
    col = _get_collection()
    if col is None:
        return (0, 0)
    q = {"phase": "drop", "result.hit": {"$in": [True, False]}}
    if session:
        q["session"] = session
    total = col.count_documents(q)
    hits = col.count_documents({**q, "result.hit": True})
    return (hits, total)


def suggest_offset(n=10):
    """Auto-tune: average the recorded miss vectors of the last n misses.
    Returns (dx, dy) in mm to ADD to the release offset, or (0, 0) if no data.
    Needs result.miss_mm = {"x": .., "y": ..} = where the ball landed minus the cup center."""
    col = _get_collection()
    if col is None:
        return (0, 0)
    misses = list(col.find({"result.hit": False, "result.miss_mm": {"$exists": True}},
                           {"result.miss_mm": 1}).sort("timestamp", -1).limit(n))
    if not misses:
        return (0, 0)
    dx = sum(m["result"]["miss_mm"]["x"] for m in misses) / len(misses)
    dy = sum(m["result"]["miss_mm"]["y"] for m in misses) / len(misses)
    return (-dx, -dy)


if __name__ == "__main__":
    log_attempt({"phase": "test", "result": {"hit": True}, "notes": "logger.py self-test"})
    print("hit rate:", hit_rate())
    print("suggested offset:", suggest_offset())
