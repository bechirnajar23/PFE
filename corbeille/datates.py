import pandas as pd
import numpy as np

np.random.seed(42)

# =========================
# CONFIG
# =========================
start = pd.Timestamp("2026-04-25 00:00:00")
end = pd.Timestamp.now()
timestamps = pd.date_range(start, end, freq="5min")

rows = []

cpu = 30
mem = 40

scenario = "normal"

for i, ts in enumerate(timestamps):

    # =========================
    # SCENARIO SWITCH
    # =========================
    if i % 500 == 0:
        scenario = np.random.choice([
            "normal", "drift", "spike", "critical",
            "wan_down", "latency", "recovery", "noise"
        ])

    # =========================
    # NORMAL
    # =========================
    if scenario == "normal":
        cpu += np.random.normal(0, 2)
        mem += np.random.normal(0, 1)

    # =========================
    # DRIFT (memory leak)
    # =========================
    elif scenario == "drift":
        cpu += np.random.uniform(0.1, 0.5)
        mem += np.random.uniform(0.3, 0.8)

    # =========================
    # SPIKE BRUTAL
    # =========================
    elif scenario == "spike":
        if np.random.rand() < 0.1:
            cpu = np.random.uniform(85, 98)
        else:
            cpu += np.random.normal(0, 3)

        mem += np.random.normal(0, 1)

    # =========================
    # CRITICAL (pre-crash)
    # =========================
    elif scenario == "critical":
        cpu += np.random.uniform(1, 3)
        mem += np.random.uniform(1, 2)

    # =========================
    # WAN DOWN
    # =========================
    elif scenario == "wan_down":
        cpu += np.random.normal(0, 1)
        mem += np.random.normal(0, 1)
        wan = 0
    else:
        wan = 1

    # =========================
    # LATENCY EXPLOSION
    # =========================
    if scenario == "latency":
        latency = np.random.uniform(150, 600)
    else:
        latency = np.random.normal(50, 10)

    # =========================
    # RECOVERY
    # =========================
    if scenario == "recovery":
        cpu -= np.random.uniform(2, 5)
        mem -= np.random.uniform(2, 4)

    # =========================
    # NOISE
    # =========================
    if scenario == "noise":
        cpu += np.random.normal(0, 10)
        mem += np.random.normal(0, 8)

    # =========================
    # CLAMP
    # =========================
    cpu = np.clip(cpu, 5, 98)
    mem = np.clip(mem, 20, 95)

    # =========================
    # STATUS
    # =========================
    if cpu > 90 or mem > 90:
        status = "CRITICAL"
        reason = "pre_crash"
    elif cpu > 70 or mem > 75:
        status = "WARNING"
        reason = "resource_pressure"
    elif latency > 120:
        status = "DEGRADED"
        reason = "high_latency"
    else:
        status = "NORMAL"
        reason = "healthy"

    # =========================
    # NETWORK
    # =========================
    rx = np.random.uniform(50, 1500)
    tx = np.random.uniform(20, 600)

    # =========================
    # SAVE ROW
    # =========================
    rows.append([
        ts, status, reason,
        cpu, cpu*0.7, cpu*0.3, 100-cpu,
        936, 936*(1-mem/100), 936*(mem/100),
        25, 120, mem,
        "RUNNING", "Bound", "Bound",
        "UP", 1, 1,
        rx, tx,
        latency, int(latency/2),
        "OK" if latency < 120 else "FAIL",
        ts
    ])

# =========================
# DATAFRAME
# =========================
columns = [
    "timestamp","LOCAL_STATUS","STATUS_REASON",
    "CPU_USAGE_PERCENT","CPU_USER_PERCENT","CPU_SYSTEM_PERCENT","CPU_IDLE_PERCENT",
    "MEM_TOTAL_MB","MEM_FREE_MB","MEM_USED_MB","BUFFERS_MB","CACHED_MB","MEM_USAGE_PERCENT",
    "DHCP_PROCESS_STATUS","DHCP_DATA_STATE","DHCP_V6_STATE",
    "WAN_STATE","WAN_IPV4_ENABLE","WAN_IPV6_ENABLE",
    "WAN_RX_RATE_KBPS","WAN_TX_RATE_KBPS",
    "NET_LATENCY_MS","NET_LATENCY_AVG_5","NET_PING_STATUS","logged_at"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv("data/test_full_scenarios.csv", index=False)

print("✅ Dataset généré:", df.shape)