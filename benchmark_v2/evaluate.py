import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v1_pipeline import run_v1_pipeline

def run_benchmark(test_dir="dataset_v2/test"):
    df = pd.read_csv(os.path.join(test_dir, "metadata.csv"))
    records = []
    latencies = []
    
    for _, row in df.iterrows():
        s_id = row['sample_id']
        true_x, true_y = row['true_x'], row['true_y']
        
        search = cv2.imread(os.path.join(test_dir, f"{s_id}_search.png"), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(os.path.join(test_dir, f"{s_id}_ref.png"), cv2.IMREAD_GRAYSCALE)
        
        t0 = time.perf_counter()
        pred_x, pred_y, _ = run_v1_pipeline(search, ref)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
        err = float(np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2))
        records.append({"error_px": err, "c1": int(err <= 1.0), "c3": int(err <= 3.0)})
        
    res_df = pd.DataFrame(records)
    summary = {
        "samples": len(res_df),
        "mean_error_px": float(res_df["error_px"].mean()),
        "top1_acc_3px": float(res_df["c3"].mean()),
        "avg_latency_ms": float(np.mean(latencies))
    }
    print("Phase 1 Benchmark Result:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_benchmark()