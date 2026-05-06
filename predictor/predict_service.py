import time
import pandas as pd
from sqlalchemy import create_engine

from predict_multi_horizon import predict_all_horizons

DB_URL = "postgresql://hgw_user:hgw_password@timescaledb:5432/hgw_monitoring"

engine = create_engine(DB_URL)

print("🚀 Predictor started...")

while True:
    try:
        # lire dernières données
        df = pd.read_sql("""
            SELECT * FROM monitor_snapshots
            ORDER BY timestamp DESC
            LIMIT 200
        """, engine)

        if len(df) < 50:
            print("⏳ waiting data...")
            time.sleep(10)
            continue

        preds = predict_all_horizons(df)

        # sauvegarder
        for horizon, prob in preds.items():
            engine.execute(
                f"""
                INSERT INTO predictions_log (timestamp, horizon, probability)
                VALUES (NOW(), '{horizon}', {prob})
                """
            )

        print("✅ predictions inserted")

    except Exception as e:
        print("❌ error:", e)

    time.sleep(30)