# data_collection.py
from ast import main
import os
import time
import threading
from collections import deque
from datetime import datetime

from config import interval
from telnet_client import create_telnet_client, send_command, close_telnet
from data_logger import save_snapshot

PROBE_FILE = "/tmp/probe_module_output.txt"
WAN_IFACE = "eth0"
DHCP_PROCESS = "dnsmasq"
PING_TARGET = "8.8.8.8"
PING_ENABLE = True
try:
    PING_COMMAND_DELAY = float(os.getenv("PING_COMMAND_DELAY", "1.2"))
except (TypeError, ValueError):
    PING_COMMAND_DELAY = 1.2

memory_usage = deque(maxlen=200)
time_points = deque(maxlen=200)
cpu_usage = deque(maxlen=200)
cpu_time_points = deque(maxlen=200)
latency_history = deque(maxlen=200)
latency_times = deque(maxlen=200)
wan_rx_history = deque(maxlen=200)
wan_tx_history = deque(maxlen=200)
wan_rate_times = deque(maxlen=200)

snapshots_history = deque(maxlen=500)

current_snapshot = {
    "timestamp": "N/A",
    "LOCAL_STATUS": "N/A",
    "STATUS_REASON": "N/A",
    "CPU_USAGE_PERCENT": 0,
    "CPU_USER_PERCENT": 0,
    "CPU_SYSTEM_PERCENT": 0,
    "CPU_IDLE_PERCENT": 0,
    "MEM_TOTAL_MB": 0,
    "MEM_FREE_MB": 0,
    "MEM_USED_MB": 0,
    "BUFFERS_MB": 0,
    "CACHED_MB": 0,
    "MEM_USAGE_PERCENT": 0,
    "DHCP_PROCESS_STATUS": "N/A",
    "DHCP_DATA_STATE": "N/A",
    "DHCP_V6_STATE": "N/A",
    "WAN_STATE": "N/A",
    "WAN_IPV4_ENABLE": "N/A",
    "WAN_IPV6_ENABLE": "N/A",
    "WAN_RX_RATE_KBPS": "N/A",
    "WAN_TX_RATE_KBPS": "N/A",
    "NET_LATENCY_MS": "N/A",
    "NET_LATENCY_AVG_5": "N/A",
    "NET_PING_STATUS": "N/A",
}

_lock = threading.Lock()

_state = {
    "LAST_DHCP_DATA_STATE": "NA",
    "LAST_DHCP_V6_STATE": "NA",
    "LAST_WAN_IPV4_ENABLE": "NA",
    "LAST_WAN_IPV6_ENABLE": "NA",
    "LAST_WAN_RX_BYTES": "NA",
    "LAST_WAN_TX_BYTES": "NA",
    "LAST_LAT_1": "NA",
    "LAST_LAT_2": "NA",
    "LAST_LAT_3": "NA",
    "LAST_LAT_4": "NA",
    "LAST_LAT_5": "NA",
}


def kb_to_mb(v):
    try:
        return int(v) // 1024
    except Exception:
        return 0


def keep_last_value(current, last):
    if current not in (None, "", "NA"):
        return str(current)
    if last not in (None, "", "NA"):
        return str(last)
    return "NA"


def read_cpu_usage(tn):
    import re

    # ---------- MÉTHODE 1 : /proc/stat ----------
    def parse_cpu_stat(out):
        for line in out.splitlines():
            if line.startswith("cpu "):
                p = line.split()
                return {
                    "user": int(p[1]),
                    "nice": int(p[2]),
                    "sys": int(p[3]),
                    "idle": int(p[4]),
                    "iowait": int(p[5]) if len(p) > 5 else 0,
                }
        return None

    try:
        c1 = parse_cpu_stat(send_command(tn, "cat /proc/stat"))
        time.sleep(1)
        c2 = parse_cpu_stat(send_command(tn, "cat /proc/stat"))

        if c1 and c2:
            total1 = sum(c1.values())
            total2 = sum(c2.values())

            total_delta = total2 - total1
            idle_delta = c2["idle"] - c1["idle"]
            iowait_delta = c2["iowait"] - c1["iowait"]

            if total_delta > 0:
                usage = (total_delta - idle_delta - iowait_delta) * 100 // total_delta

                return {
                    "CPU_USAGE_PERCENT": usage,
                    "CPU_USER_PERCENT": c2["user"] * 100 // total2,
                    "CPU_SYSTEM_PERCENT": c2["sys"] * 100 // total2,
                    "CPU_IDLE_PERCENT": (idle_delta + iowait_delta) * 100 // total_delta,
                }
    except:
        pass

    # ---------- MÉTHODE 2 : fallback TOP ----------
    try:
        out = send_command(tn, "top -n 1 | head -n 5")
        line = [l for l in out.split("\n") if "CPU:" in l][0]

        usr = int(re.search(r'(\d+)% usr', line).group(1))
        sys = int(re.search(r'(\d+)% sys', line).group(1))
        idle = int(re.search(r'(\d+)% idle', line).group(1))

        return {
            "CPU_USAGE_PERCENT": 100 - idle,
            "CPU_USER_PERCENT": usr,
            "CPU_SYSTEM_PERCENT": sys,
            "CPU_IDLE_PERCENT": idle,
        }
    except:
        pass

    return {
        "CPU_USAGE_PERCENT": 0,
        "CPU_USER_PERCENT": 0,
        "CPU_SYSTEM_PERCENT": 0,
        "CPU_IDLE_PERCENT": 0,
    }


def read_memory_usage(tn):
    import re

    # -------- MÉTHODE 1 : meminfo --------
    try:
        out = send_command(tn, "cat /proc/meminfo")

        mem_total = mem_free = buffers = cached = 0

        for line in out.splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1])
            elif line.startswith("MemFree:"):
                mem_free = int(line.split()[1])
            elif line.startswith("Buffers:"):
                buffers = int(line.split()[1])
            elif line.startswith("Cached:"):
                cached = int(line.split()[1])

        if mem_total > 0:
            mem_used = mem_total - mem_free
            mem_pct = mem_used * 100 // mem_total

            return {
                "MEM_TOTAL_MB": kb_to_mb(mem_total),
                "MEM_FREE_MB": kb_to_mb(mem_free),
                "MEM_USED_MB": kb_to_mb(mem_used),
                "BUFFERS_MB": kb_to_mb(buffers),
                "CACHED_MB": kb_to_mb(cached),
                "MEM_USAGE_PERCENT": mem_pct,
            }
    except:
        pass

    # -------- FALLBACK : top --------
    try:
        out = send_command(tn, "top -n 1 | head -n 3")
        line = [l for l in out.split("\n") if "Mem:" in l][0]

        used = int(re.search(r'(\d+)K used', line).group(1))
        free = int(re.search(r'(\d+)K free', line).group(1))

        total = used + free
        pct = used * 100 // total if total > 0 else 0

        return {
            "MEM_TOTAL_MB": kb_to_mb(total),
            "MEM_FREE_MB": kb_to_mb(free),
            "MEM_USED_MB": kb_to_mb(used),
            "BUFFERS_MB": 0,
            "CACHED_MB": 0,
            "MEM_USAGE_PERCENT": pct,
        }
    except:
        pass

    return {
        "MEM_TOTAL_MB": 0,
        "MEM_FREE_MB": 0,
        "MEM_USED_MB": 0,
        "BUFFERS_MB": 0,
        "CACHED_MB": 0,
        "MEM_USAGE_PERCENT": 0,
    }
#raise alerte 

def read_dhcp_status(tn):
    result = {
        "DHCP_PROCESS_STATUS": "STOPPED",
        "DHCP_DATA_STATE": "NA",
        "DHCP_V6_STATE": "NA",
    }

    out_ps = send_command(tn, "ps")
    if DHCP_PROCESS in out_ps:
        result["DHCP_PROCESS_STATUS"] = "RUNNING"

    out_data = send_command(
        tn,
        f"grep 'DHCP, Type = Status, Intf = dhcp_data' {PROBE_FILE} 2>/dev/null | tail -n 1",
    )
    m = None
    if out_data.strip():
        import re
        m = re.search(r"State = ([^,]+)", out_data)
    if m:
        result["DHCP_DATA_STATE"] = m.group(1).strip()

    out_v6 = send_command(
        tn,
        f"grep 'DHCP, Type = Status, Intf = dhcpv6_data' {PROBE_FILE} 2>/dev/null | tail -n 1",
    )
    m = None
    if out_v6.strip():
        import re
        m = re.search(r"State = ([^,]+)", out_v6)
    if m:
        result["DHCP_V6_STATE"] = m.group(1).strip()

    result["DHCP_DATA_STATE"] = keep_last_value(
        result["DHCP_DATA_STATE"], _state["LAST_DHCP_DATA_STATE"]
    )
    result["DHCP_V6_STATE"] = keep_last_value(
        result["DHCP_V6_STATE"], _state["LAST_DHCP_V6_STATE"]
    )
    return result


def read_wan_status_and_rate(tn):
    result = {
        "WAN_STATE": "UNKNOWN",
        "WAN_IPV4_ENABLE": "NA",
        "WAN_IPV6_ENABLE": "NA",
        "WAN_RX_BYTES": "NA",
        "WAN_TX_BYTES": "NA",
        "WAN_RX_RATE_KBPS": "NA",
        "WAN_TX_RATE_KBPS": "NA",
    }

    out_link = send_command(tn, f"ip link show {WAN_IFACE} 2>/dev/null")
    if "state UP" in out_link:
        result["WAN_STATE"] = "UP"
    elif "state DOWN" in out_link:
        result["WAN_STATE"] = "DOWN"

    out_ip = send_command(
        tn,
        f"grep 'WAN, Type = IPStatus' {PROBE_FILE} 2>/dev/null | tail -n 1",
    )
    import re
    if out_ip.strip():
        m4 = re.search(r"IPv4Enable = (\d+)", out_ip)
        m6 = re.search(r"IPv6Enable = (\d+)", out_ip)
        if m4:
            result["WAN_IPV4_ENABLE"] = m4.group(1)
        if m6:
            result["WAN_IPV6_ENABLE"] = m6.group(1)

    result["WAN_IPV4_ENABLE"] = keep_last_value(
        result["WAN_IPV4_ENABLE"], _state["LAST_WAN_IPV4_ENABLE"]
    )
    result["WAN_IPV6_ENABLE"] = keep_last_value(
        result["WAN_IPV6_ENABLE"], _state["LAST_WAN_IPV6_ENABLE"]
    )

    out_rx = send_command(
        tn,
        f"cat /sys/class/net/{WAN_IFACE}/statistics/rx_bytes 2>/dev/null",
    ).strip()
    out_tx = send_command(
        tn,
        f"cat /sys/class/net/{WAN_IFACE}/statistics/tx_bytes 2>/dev/null",
    ).strip()

    try:
        result["WAN_RX_BYTES"] = int(out_rx)
    except Exception:
        result["WAN_RX_BYTES"] = "NA"

    try:
        result["WAN_TX_BYTES"] = int(out_tx)
    except Exception:
        result["WAN_TX_BYTES"] = "NA"

    if (
        result["WAN_RX_BYTES"] != "NA"
        and _state["LAST_WAN_RX_BYTES"] not in ("NA", "", None)
    ):
        rx_delta = int(result["WAN_RX_BYTES"]) - int(_state["LAST_WAN_RX_BYTES"])
        if rx_delta < 0:
            rx_delta = 0
        result["WAN_RX_RATE_KBPS"] = rx_delta // 1024 // interval

    if (
        result["WAN_TX_BYTES"] != "NA"
        and _state["LAST_WAN_TX_BYTES"] not in ("NA", "", None)
    ):
        tx_delta = int(result["WAN_TX_BYTES"]) - int(_state["LAST_WAN_TX_BYTES"])
        if tx_delta < 0:
            tx_delta = 0
        result["WAN_TX_RATE_KBPS"] = tx_delta // 1024 // interval

    return result


def read_ping_status(tn):
    result = {
        "NET_LATENCY_MS": "NA",
        "NET_PING_STATUS": "DISABLED",
    }

    if not PING_ENABLE:
        return result

    out = send_command(
        tn,
        f"ping -c 1 -W 1 {PING_TARGET} 2>/dev/null",
        timeout=PING_COMMAND_DELAY,
    )
    import re
    if "time=" in out:
        m = re.search(r"time=([\d.]+)", out)
        if m:
            result["NET_LATENCY_MS"] = m.group(1)
            result["NET_PING_STATUS"] = "OK"
    else:
        result["NET_PING_STATUS"] = "FAIL"

    return result


def compute_latency_average(current):
    vals = [
        current,
        _state["LAST_LAT_1"],
        _state["LAST_LAT_2"],
        _state["LAST_LAT_3"],
        _state["LAST_LAT_4"],
    ]
    total = 0
    count = 0
    for v in vals:
        if v not in ("", None, "NA"):
            try:
                total += int(float(v))
                count += 1
            except Exception:
                pass
    return str(total // count) if count > 0 else "NA"


def compute_local_status(snap):
    local_status = "NORMAL"
    status_reason = "healthy"

    if snap["DHCP_PROCESS_STATUS"] == "STOPPED":
        return "URGENT", "dhcp_process_stopped"

    if snap["WAN_STATE"] != "UP":
        local_status = "WARNING"
        status_reason = "wan_not_up"

    if snap["DHCP_DATA_STATE"] != "Bound" and snap["WAN_STATE"] == "UP":
        local_status = "WARNING"
        status_reason = "dhcp_not_bound"

    if int(snap["CPU_USAGE_PERCENT"]) >= 85:
        return "URGENT", "high_cpu"

    if int(snap["CPU_USAGE_PERCENT"]) >= 70 and local_status == "NORMAL":
        local_status = "WARNING"
        status_reason = "cpu_elevated"

    if int(snap["MEM_USAGE_PERCENT"]) >= 90:
        return "URGENT", "high_memory"

    if int(snap["MEM_USAGE_PERCENT"]) >= 80 and local_status == "NORMAL":
        local_status = "WARNING"
        status_reason = "memory_elevated"

    if snap["NET_PING_STATUS"] == "FAIL" and snap["WAN_STATE"] == "UP":
        local_status = "WARNING"
        status_reason = "ping_failed"

    if snap["NET_LATENCY_AVG_5"] != "NA":
        try:
            if int(snap["NET_LATENCY_AVG_5"]) >= 120 and local_status == "NORMAL":
                local_status = "WARNING"
                status_reason = "latency_elevated"
        except Exception:
            pass

    return local_status, status_reason


def save_internal_state(snap):
    _state["LAST_DHCP_DATA_STATE"] = snap["DHCP_DATA_STATE"]
    _state["LAST_DHCP_V6_STATE"] = snap["DHCP_V6_STATE"]
    _state["LAST_WAN_IPV4_ENABLE"] = snap["WAN_IPV4_ENABLE"]
    _state["LAST_WAN_IPV6_ENABLE"] = snap["WAN_IPV6_ENABLE"]
    _state["LAST_WAN_RX_BYTES"] = snap["WAN_RX_BYTES"]
    _state["LAST_WAN_TX_BYTES"] = snap["WAN_TX_BYTES"]
    _state["LAST_LAT_5"] = _state["LAST_LAT_4"]
    _state["LAST_LAT_4"] = _state["LAST_LAT_3"]
    _state["LAST_LAT_3"] = _state["LAST_LAT_2"]
    _state["LAST_LAT_2"] = _state["LAST_LAT_1"]
    _state["LAST_LAT_1"] = snap["NET_LATENCY_MS"]


def collect_data():
    while True:
        tn = None
        try:
            tn = create_telnet_client()
            if tn is None:
                print("[ERROR] Connexion Telnet échouée")
                time.sleep(interval)
                continue

            print("[INFO] Collecte HGW démarrée ✅")

            while True:
                cycle_started = time.monotonic()
                ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
                now = datetime.now().strftime("%H:%M:%S")

                cpu = read_cpu_usage(tn)
                mem = read_memory_usage(tn)
                dhcp = read_dhcp_status(tn)
                wan = read_wan_status_and_rate(tn)
                ping = read_ping_status(tn)
                lat_avg = compute_latency_average(ping["NET_LATENCY_MS"])

                snap = {
                    "timestamp": ts,
                    **cpu,
                    **mem,
                    **dhcp,
                    **wan,
                    **ping,
                    "NET_LATENCY_AVG_5": lat_avg,
                }

                local_status, status_reason = compute_local_status(snap)
                snap["LOCAL_STATUS"] = local_status
                snap["STATUS_REASON"] = status_reason

                save_internal_state(snap)

                with _lock:
                    current_snapshot.update(snap)
                    snapshots_history.append(dict(snap))

                    memory_usage.append(snap["MEM_USAGE_PERCENT"])
                    time_points.append(now)

                    cpu_usage.append(snap["CPU_USAGE_PERCENT"])
                    cpu_time_points.append(now)

                    if snap["NET_LATENCY_MS"] != "NA":
                        try:
                            latency_history.append(float(snap["NET_LATENCY_MS"]))
                            latency_times.append(now)
                        except Exception:
                            pass

                    if snap["WAN_RX_RATE_KBPS"] != "NA":
                        try:
                            wan_rx_history.append(int(snap["WAN_RX_RATE_KBPS"]))
                            wan_tx_history.append(
                                int(snap["WAN_TX_RATE_KBPS"])
                                if snap["WAN_TX_RATE_KBPS"] != "NA" else 0
                            )
                            wan_rate_times.append(now)
                        except Exception:
                            pass

                save_snapshot(snap)

                print(
                    f"[SNAPSHOT] timestamp={snap['timestamp']} "
                    f"LOCAL_STATUS={snap['LOCAL_STATUS']} "
                    f"STATUS_REASON={snap['STATUS_REASON']} "
                    f"CPU_USAGE_PERCENT={snap['CPU_USAGE_PERCENT']} "
                    f"MEM_USAGE_PERCENT={snap['MEM_USAGE_PERCENT']} "
                    f"DHCP_PROCESS_STATUS={snap['DHCP_PROCESS_STATUS']} "
                    f"DHCP_DATA_STATE={snap['DHCP_DATA_STATE']} "
                    f"DHCP_V6_STATE={snap['DHCP_V6_STATE']} "
                    f"WAN_STATE={snap['WAN_STATE']} "
                    f"WAN_IPV4_ENABLE={snap['WAN_IPV4_ENABLE']} "
                    f"WAN_IPV6_ENABLE={snap['WAN_IPV6_ENABLE']} "
                    f"WAN_RX_RATE_KBPS={snap['WAN_RX_RATE_KBPS']} "
                    f"WAN_TX_RATE_KBPS={snap['WAN_TX_RATE_KBPS']} "
                    f"NET_LATENCY_MS={snap['NET_LATENCY_MS']} "
                    f"NET_LATENCY_AVG_5={snap['NET_LATENCY_AVG_5']} "
                    f"NET_PING_STATUS={snap['NET_PING_STATUS']}"
                )

                elapsed = time.monotonic() - cycle_started
                time.sleep(max(0, interval - elapsed))

        except Exception as e:
            print(f"[ERROR] collect_data : {e}")
        finally:
            if tn:
                close_telnet(tn)

        time.sleep(interval)
import time

if __name__ == "__main__":
    collect_data()
