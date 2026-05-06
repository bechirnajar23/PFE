# config.py

# ── Telnet ──────────────────────────────────
hostname = '192.168.1.1'
username = 'root'
password = 'sah'
port     = 23
interval = 5

# ── Serveur HTTP récepteur ──────────────────
RECEIVER_HOST = '0.0.0.0'
RECEIVER_PORT = 8080          # Port écouté par le serveur Python
                               # (correspond à SERVER_URL dans le script bash)

# ── Fenêtre temporelle ──────────────────────
WINDOW_SECONDS      = 60      # Durée fenêtre urgente (secondes)
NORMAL_SEND_INTERVAL = 120    # Intervalle envoi normal (secondes)
