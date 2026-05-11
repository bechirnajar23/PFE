import telnetlib
import os
from dotenv import load_dotenv
import time

load_dotenv()

HOST = os.getenv("HGW_HOST")
USER = os.getenv("HGW_USER")
PASSWORD = os.getenv("HGW_PASSWORD")
try:
    COMMAND_DELAY = float(os.getenv("TELNET_COMMAND_DELAY", "0.4"))
except (TypeError, ValueError):
    COMMAND_DELAY = 0.4

def create_telnet_client():
    try:
        tn = telnetlib.Telnet(HOST, timeout=5)

        tn.read_until(b"login: ")
        tn.write(USER.encode() + b"\n")

        tn.read_until(b"Password: ")
        tn.write(PASSWORD.encode() + b"\n")

        print(f"[INFO] Connected to HGW {HOST} ✅")
        return tn

    except Exception as e:
        print(f"[ERROR] Telnet connection failed: {e}")
        return None


def send_command(tn, command, timeout=None):
    if timeout is None:
        timeout = COMMAND_DELAY
    tn.write((command + "\n").encode())

    time.sleep(timeout)

    output = tn.read_very_eager().decode(errors="ignore")

    # 🔥 nettoyage IMPORTANT
    output = output.replace("\r", "")
    output = output.replace("\x00", "")

    # enlever prompt HGW
    lines = output.split("\n")
    clean_lines = []

    for line in lines:
        if line.strip().endswith("#"):
            continue
        if command in line:
            continue
        clean_lines.append(line)
    
    
    return "\n".join(clean_lines).strip()



def close_telnet(tn):
    try:
        tn.close()
    except:
        pass
