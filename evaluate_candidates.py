import os
import time
import cv2
import numpy as np
import pandas as pd
from candidate_generation_v2 import generate_candidates

def evaluate_recall(test_dir="dataset_v2/test"):
    df = pd.read_csv(os.path.join(test_dir, "metadata.csv"))
    k_list = [25, 50, 100, 250]
    results = []
    
    for k in k_list:
        hits_3px = 0
        latencies = []
        for _, row in df.iterrows():
            s_id = row['sample_id']
            true_x, true_y = row['true_x'], row['true_y']
            
            search = cv2.imread(os.path.join(test_dir, f"{s_id}_search.png"), cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(os.path.join(test_dir, f"{s_id}_ref.png"), cv2.IMREAD_GRAYSCALE)
            
            t0 = time.perf_counter()
            cands = generate_candidates(search, ref, top_k=k)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
            min_d = min([np.sqrt((cx - true_x)**2 + (cy - true_y)**2) for cx, cy, _ in cands])
            if min_d <= 3.0:
                hits_3px += 1
                
        results.append({
            "K": k,
            "Latency_ms": float(np.mean(latencies)),
            "Recall@3px": hits_3px / len(df)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("candidate_recall_results.csv", index=False)
    print("Candidate Recall Results:")
    print(res_df)

if __name__ == "__main__":
    evaluate_recall()