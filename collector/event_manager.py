# event_manager.py
import os
import time
import threading
from datetime import datetime
from collections import deque

HGW_LOG = "/tmp/hgw_monitor.log"
STATE_FILE = "/tmp/event_comm.state"

CHECK_INTERVAL = 5
NORMAL_SEND_INTERVAL = 120
WINDOW_SECONDS = 60

_lock = threading.Lock()

# ─────────────────────────────────────────────
# État équivalent au script shell
# ─────────────────────────────────────────────
state = {
    "WINDOW_OPEN": False,
    "WINDOW_START_EPOCH": None,
    "WINDOW_END_EPOCH": None,
    "EVENT_COUNT": 0,
    "LAST_RECORDED_URGENT_LINE": "",
    "LAST_PROCESSED_LINE": 0,
    "LAST_SENT_NORMAL_LINE": 0,
    "LAST_NORMAL_SEND_EPOCH": 0,
}

# ─────────────────────────────────────────────
# Structures exposées à app.py
# ─────────────────────────────────────────────
urgent_window = {
    "open": False,
    "start_timestamp": "",
    "start_epoch": "",
    "end_epoch": "",
    "duration": "",
    "event_count": 0,
    "last_update": "",
    "events": [],
}

urgent_history = deque(maxlen=200)
normal_history = deque(maxlen=500)


# ═════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")


def get_epoch():
    return int(time.time())


def log_msg(msg: str):
    print(f"[EVENT] {get_timestamp()} {msg}")


def load_state():
    if not os.path.exists(STATE_FILE):
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")

                if k in (
                    "WINDOW_START_EPOCH",
                    "WINDOW_END_EPOCH",
                    "EVENT_COUNT",
                    "LAST_PROCESSED_LINE",
                    "LAST_SENT_NORMAL_LINE",
                    "LAST_NORMAL_SEND_EPOCH",
                ):
                    try:
                        state[k] = int(v) if v else 0
                    except ValueError:
                        state[k] = 0
                elif k == "WINDOW_OPEN":
                    state[k] = str(v) == "1"
                else:
                    state[k] = v
    except Exception as e:
        log_msg(f"LOAD_STATE_ERROR {e}")


def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write(f'WINDOW_OPEN="{1 if state["WINDOW_OPEN"] else 0}"\n')
            f.write(f'WINDOW_START_EPOCH="{state["WINDOW_START_EPOCH"] or ""}"\n')
            f.write(f'WINDOW_END_EPOCH="{state["WINDOW_END_EPOCH"] or ""}"\n')
            f.write(f'EVENT_COUNT="{state["EVENT_COUNT"]}"\n')
            f.write(
                f'LAST_RECORDED_URGENT_LINE='
                f'"{state["LAST_RECORDED_URGENT_LINE"]}"\n'
            )
            f.write(f'LAST_PROCESSED_LINE="{state["LAST_PROCESSED_LINE"]}"\n')
            f.write(f'LAST_SENT_NORMAL_LINE="{state["LAST_SENT_NORMAL_LINE"]}"\n')
            f.write(f'LAST_NORMAL_SEND_EPOCH="{state["LAST_NORMAL_SEND_EPOCH"]}"\n')
    except Exception as e:
        log_msg(f"SAVE_STATE_ERROR {e}")


# ═════════════════════════════════════════════
# Fenêtre urgente
# ═════════════════════════════════════════════
def create_new_urgent_window(current_ts: str, current_epoch: int, first_line: str):
    state["WINDOW_OPEN"] = True
    state["WINDOW_START_EPOCH"] = current_epoch
    state["WINDOW_END_EPOCH"] = current_epoch + WINDOW_SECONDS
    state["EVENT_COUNT"] = 1
    state["LAST_RECORDED_URGENT_LINE"] = first_line

    with _lock:
        urgent_window["open"] = True
        urgent_window["start_timestamp"] = current_ts
        urgent_window["start_epoch"] = current_epoch
        urgent_window["end_epoch"] = current_epoch + WINDOW_SECONDS
        urgent_window["duration"] = f"{WINDOW_SECONDS}s"
        urgent_window["event_count"] = 1
        urgent_window["last_update"] = current_ts
        urgent_window["events"] = [first_line]

    log_msg(
        f"URGENT_WINDOW_OPENED start={state['WINDOW_START_EPOCH']} "
        f"end={state['WINDOW_END_EPOCH']}"
    )


def append_to_urgent_window(current_ts: str, new_line: str):
    state["EVENT_COUNT"] += 1
    state["LAST_RECORDED_URGENT_LINE"] = new_line

    with _lock:
        urgent_window["event_count"] = state["EVENT_COUNT"]
        urgent_window["last_update"] = current_ts
        urgent_window["events"].append(new_line)

    log_msg(f"URGENT_WINDOW_UPDATED event_count={state['EVENT_COUNT']}")


def close_urgent_window_if_expired(now: int):
    if (
        state["WINDOW_OPEN"]
        and state["WINDOW_END_EPOCH"]
        and now >= state["WINDOW_END_EPOCH"]
    ):
        with _lock:
            urgent_history.append({
                "received_at": get_timestamp(),
                "event_count": urgent_window["event_count"],
                "window_start": urgent_window["start_timestamp"],
                "window_duration": urgent_window["duration"],
                "events": list(urgent_window["events"]),
            })

            urgent_window["open"] = False
            urgent_window["start_timestamp"] = ""
            urgent_window["start_epoch"] = ""
            urgent_window["end_epoch"] = ""
            urgent_window["duration"] = ""
            urgent_window["event_count"] = 0
            urgent_window["last_update"] = ""
            urgent_window["events"] = []

        log_msg(
            f"URGENT_WINDOW_CLOSED start={state['WINDOW_START_EPOCH']} "
            f"end={state['WINDOW_END_EPOCH']} total_events={state['EVENT_COUNT']}"
        )

        state["WINDOW_OPEN"] = False
        state["WINDOW_START_EPOCH"] = None
        state["WINDOW_END_EPOCH"] = None
        state["EVENT_COUNT"] = 0
        state["LAST_RECORDED_URGENT_LINE"] = ""


# ═════════════════════════════════════════════
# Lecture fichier log
# ═════════════════════════════════════════════
def process_new_lines_for_urgent():
    if not os.path.exists(HGW_LOG):
        return

    try:
        with open(HGW_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        log_msg(f"READ_LOG_ERROR {e}")
        return

    total_lines = len(lines)
    if total_lines <= state["LAST_PROCESSED_LINE"]:
        return

    new_lines = lines[state["LAST_PROCESSED_LINE"]:]
    for line in new_lines:
        line = line.strip()
        if not line:
            continue

        local_status = ""
        parts = line.split()
        for p in parts:
            if p.startswith("LOCAL_STATUS="):
                local_status = p.split("=", 1)[1]
                break

        if not local_status:
            continue

        if local_status == "URGENT":
            current_ts = get_timestamp()
            current_epoch = get_epoch()

            if not state["WINDOW_OPEN"]:
                create_new_urgent_window(current_ts, current_epoch, line)
            else:
                if line != state["LAST_RECORDED_URGENT_LINE"]:
                    append_to_urgent_window(current_ts, line)

    state["LAST_PROCESSED_LINE"] = total_lines


# ═════════════════════════════════════════════
# Historique NORMAL
# ═════════════════════════════════════════════
def process_normal_history_if_due():
    now = get_epoch()

    if not state["LAST_NORMAL_SEND_EPOCH"]:
        state["LAST_NORMAL_SEND_EPOCH"] = now
        return

    elapsed = now - state["LAST_NORMAL_SEND_EPOCH"]
    if elapsed < NORMAL_SEND_INTERVAL:
        return

    if not os.path.exists(HGW_LOG):
        state["LAST_NORMAL_SEND_EPOCH"] = now
        return

    try:
        with open(HGW_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        log_msg(f"READ_NORMAL_ERROR {e}")
        state["LAST_NORMAL_SEND_EPOCH"] = now
        return

    total_lines = len(lines)
    if total_lines <= state["LAST_SENT_NORMAL_LINE"]:
        state["LAST_NORMAL_SEND_EPOCH"] = now
        return

    new_lines = lines[state["LAST_SENT_NORMAL_LINE"]:]
    normal_lines = [l.strip() for l in new_lines if "LOCAL_STATUS=NORMAL" in l]

    if not normal_lines:
        state["LAST_SENT_NORMAL_LINE"] = total_lines
        state["LAST_NORMAL_SEND_EPOCH"] = now
        return

    last_status = ""
    if lines:
        last_line = lines[-1].strip()
        for p in last_line.split():
            if p.startswith("LOCAL_STATUS="):
                last_status = p.split("=", 1)[1]
                break

    if last_status == "URGENT":
        log_msg("NORMAL_SEND_SKIPPED reason=urgent_active")
        return

    ts = get_timestamp()

    with _lock:
        normal_history.append({
            "received_at": ts,
            "timestamp": ts,
            "from_line": state["LAST_SENT_NORMAL_LINE"] + 1,
            "to_line": total_lines,
            "lines": normal_lines,
        })

    log_msg(
        f"NORMAL_HISTORY_STORED "
        f"from_line={state['LAST_SENT_NORMAL_LINE'] + 1} "
        f"to_line={total_lines}"
    )

    state["LAST_SENT_NORMAL_LINE"] = total_lines
    state["LAST_NORMAL_SEND_EPOCH"] = now


# ═════════════════════════════════════════════
# Boucle principale
# ═════════════════════════════════════════════
def event_loop():
    load_state()

    if not state["LAST_NORMAL_SEND_EPOCH"]:
        state["LAST_NORMAL_SEND_EPOCH"] = get_epoch()

    log_msg(
        f"EVENT_MANAGER_STARTED "
        f"check_interval={CHECK_INTERVAL}s "
        f"normal_interval={NORMAL_SEND_INTERVAL}s"
    )

    while True:
        try:
            now = get_epoch()
            close_urgent_window_if_expired(now)
            process_new_lines_for_urgent()
            process_normal_history_if_due()
            save_state()
        except Exception as e:
            log_msg(f"EVENT_LOOP_ERROR {e}")

        time.sleep(CHECK_INTERVAL)


def start_event_manager_thread():
    thread = threading.Thread(target=event_loop, daemon=True)
    thread.start()
    log_msg("EVENT_MANAGER_THREAD_STARTED")
    return thread
