# utils/durability.py
import os, json, time, pandas as pd
from datetime import datetime

def new_job_dir(base="runs"):
    os.makedirs(base, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join(base, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, ensure_ascii=False, indent=2, fp=f)

def append_csv(path, df: pd.DataFrame):
    header = not os.path.exists(path)
    df.to_csv(path, index=False, mode="a", header=header, encoding="utf-8")

def read_existing_results(path) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            pass
    return pd.DataFrame()

def heartbeat(path):
    with open(path, "w") as f:
        f.write(str(time.time()))
