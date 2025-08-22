import json
import os
import logging

logger = logging.getLogger(__name__)

BLACKLIST_FILE = 'blacklist.json'

blacklist = []

def load_blacklist():
    global blacklist
    if os.path.exists(BLACKLIST_FILE):
        try:
            with open(BLACKLIST_FILE, 'r') as f:
                blacklist = json.load(f)
        except json.JSONDecodeError:
            logger.error("Failed to load blacklist.json: invalid JSON")
            blacklist = []
    else:
        blacklist = []

def save_blacklist():
    with open(BLACKLIST_FILE, 'w') as f:
        json.dump(blacklist, f)

def add_to_blacklist(user_id):
    if user_id not in blacklist:
        blacklist.append(user_id)
        save_blacklist()

def remove_from_blacklist(user_id):
    if user_id in blacklist:
        blacklist.remove(user_id)
        save_blacklist()

load_blacklist()