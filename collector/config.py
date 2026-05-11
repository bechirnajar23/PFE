# config.py
import os

from dotenv import load_dotenv

load_dotenv()


def _int_env(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# HGW connection
hostname = os.getenv("HGW_HOST", "192.168.1.1")
username = os.getenv("HGW_USER", "root")
password = os.getenv("HGW_PASSWORD", "sah")
port = _int_env("HGW_PORT", 23)

# Target delay between two snapshots. If the collection itself takes longer
# than this value, the next cycle starts immediately.
interval = max(1, _int_env("COLLECTION_INTERVAL", 5))

# HTTP receiver kept for compatibility with older scripts.
RECEIVER_HOST = os.getenv("RECEIVER_HOST", "0.0.0.0")
RECEIVER_PORT = _int_env("RECEIVER_PORT", 8080)

# Event window settings.
WINDOW_SECONDS = _int_env("WINDOW_SECONDS", 60)
NORMAL_SEND_INTERVAL = _int_env("NORMAL_SEND_INTERVAL", 120)
