import pandas as pd
import numpy as np

np.random.seed(42)

# =========================
# TIME RANGE
# =========================
start = pd.Timestamp("2026-04-03 00:00:00")
end = pd.Timestamp.now()

timestamps = pd.date_range(start, end, freq="h")

# =========================
# INIT
# =========================
rows = []

cpu = 25
mem = 35

daily_growth = 0
cycle_day = 0

# =========================
# LOOP
# =========================
for ts in timestamps:

    hour = ts.hour

    # =========================
    # RESET CHAQUE JOUR
    # =========================
    if hour == 0:
        cycle_day += 1

        # croissance journalière (4% → 10%)
        daily_growth = np.random.uniform(4, 10)

    # =========================
    # PROGRESSION DANS LA JOURNÉE
    # =========================
    cpu += (daily_growth / 24) + np.random.normal(0, 0.3)
    mem += (daily_growth / 24) + np.random.normal(0, 0.2)

    # =========================
    # BRUIT RÉALISTE
    # =========================
    cpu += np.sin(hour / 24 * 2 * np.pi) * 2
    mem += np.sin(hour / 24 * 2 * np.pi) * 1.5

    # =========================
    # CRASH CONDITION
    # =========================
    crash = False

    if cpu >= 90 or mem >= 90:
        crash = True

    # =========================
    # REBOOT APRÈS CRASH
    # =========================
    if crash:
        status = "CRITICAL"
        reason = "pre_crash"

        # enregistrer point critique
        rows.append([
            ts, status, reason, cpu, mem, 500, "UP"
        ])

        # reboot (reset brutal)
        cpu = np.random.uniform(10, 25)
        mem = np.random.uniform(20, 40)

        cycle_day = 0
        continue

    # =========================
    # LATENCE
    # =========================
    latency = 40 + cpu * 0.5 + np.random.normal(0, 5)

    # =========================
    # WAN DOWN (rare)
    # =========================
    if np.random.rand() < 0.01:
        wan = "DOWN"
        status = "CRITICAL"
        reason = "wan_failure"
    else:
        wan = "UP"

        # =========================
        # STATUS LOGIQUE
        # =========================
        if cpu > 80 or mem > 80:
            status = "WARNING"
            reason = "resource_pressure"
        else:
            status = "NORMAL"
            reason = "healthy"

    # =========================
    # SAVE
    # =========================
    rows.append([
        ts, status, reason, cpu, mem, latency, wan
    ])

# =========================
# DATAFRAME
# =========================
df = pd.DataFrame(rows, columns=[
    "timestamp",
    "LOCAL_STATUS",
    "STATUS_REASON",
    "CPU_USAGE_PERCENT",
    "MEM_USAGE_PERCENT",
    "NET_LATENCY_MS",
    "WAN_STATE"
])

df.to_csv(r"C:\Users\bnajjar\Desktop\projet conda\data\test.csv", index=False)

print("✅ Dataset généré :", df.shape)
print("📅 From:", df["timestamp"].min())
print("📅 To  :", df["timestamp"].max())