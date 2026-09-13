import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

# Main project folder ko import path me add karein
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from candidate_generation_v2 import generate_candidates

def extract_features(patch, ref):
    p_f = patch.astype(np.float32)
    r_f = ref.astype(np.float32)
    mse = np.mean((p_f - r_f)**2)
    mean_diff = abs(np.mean(p_f) - np.mean(r_f))
    std_diff = abs(np.std(p_f) - np.std(r_f))
    res = cv2.matchTemplate(patch, ref, cv2.TM_CCOEFF_NORMED)
    corr = float(res[0][0]) if res.size > 0 else 0.0
    return [mse, mean_diff, std_diff, corr]

def mine_and_train(train_dir="dataset_v2/train", test_dir="dataset_v2/test"):
    df_train = pd.read_csv(os.path.join(train_dir, "metadata.csv"))
    rows = []
    
    print("Mining hard negatives from training set...")
    for _, row in df_train.iterrows():
        s_id = row['sample_id']
        true_x, true_y = row['true_x'], row['true_y']
        
        search = cv2.imread(os.path.join(train_dir, f"{s_id}_search.png"), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(os.path.join(train_dir, f"{s_id}_ref.png"), cv2.IMREAD_GRAYSCALE)
        
        cands = generate_candidates(search, ref, top_k=50)
        h, w = ref.shape
        
        for cx, cy, score in cands:
            dist = np.sqrt((cx - true_x)**2 + (cy - true_y)**2)
            x1, y1 = int(cx - w//2), int(cy - h//2)
            x2, y2 = int(cx + w//2), int(cy + h//2)
            if x1 < 0 or y1 < 0 or x2 > search.shape[1] or y2 > search.shape[0]:
                continue
                
            patch = search[y1:y2, x1:x2]
            feats = extract_features(patch, ref)
            
            if dist <= 3.0:
                rows.append(feats + [score, 1])
            elif dist > 10.0 and score > 0.35:
                rows.append(feats + [score, 0])
                
    cols = ["mse", "mean_diff", "std_diff", "corr", "cand_score", "label"]
    dataset = pd.DataFrame(rows, columns=cols)
    dataset.to_csv("hard_negative_mining/hard_negatives.csv", index=False)
    
    # Train XGBoost
    X = dataset.drop(columns=["label"])
    y = dataset["label"]
    scale_pos = float((y == 0).sum()) / max(1, (y == 1).sum())
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42)
    model.fit(X, y)
    
    with open("hard_negative_mining/model/xgboost_hard_neg.pkl", "wb") as f:
        pickle.dump(model, f)
    print("XGBoost model successfully trained and saved!")

    # Test Evaluation
    df_test = pd.read_csv(os.path.join(test_dir, "metadata.csv"))
    v1_errs, v2_errs = [], []
    
    for _, row in df_test.iterrows():
        s_id = row['sample_id']
        true_x, true_y = row['true_x'], row['true_y']
        
        search = cv2.imread(os.path.join(test_dir, f"{s_id}_search.png"), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(os.path.join(test_dir, f"{s_id}_ref.png"), cv2.IMREAD_GRAYSCALE)
        cands = generate_candidates(search, ref, top_k=50)
        h, w = ref.shape
        
        # V1 baseline
        v1_errs.append(np.sqrt((cands[0][0] - true_x)**2 + (cands[0][1] - true_y)**2))
        
        # V2 XGBoost
        feat_list, valid_pts = [], []
        for cx, cy, score in cands:
            x1, y1 = int(cx - w//2), int(cy - h//2)
            x2, y2 = int(cx + w//2), int(cy + h//2)
            if x1 >= 0 and y1 >= 0 and x2 <= search.shape[1] and y2 <= search.shape[0]:
                feat_list.append(extract_features(search[y1:y2, x1:x2], ref) + [score])
                valid_pts.append((cx, cy))
                
        if feat_list:
            probs = model.predict_proba(feat_list)[:, 1]
            best_pt = valid_pts[np.argmax(probs)]
            v2_errs.append(np.sqrt((best_pt[0] - true_x)**2 + (best_pt[1] - true_y)**2))
        else:
            v2_errs.append(v1_errs[-1])
            
    report = f"""# Phase 3 Hard-Negative Mining Results
- **Baseline Mean Error**: {np.mean(v1_errs):.2f} px
- **Hard-Negative XGBoost Mean Error**: {np.mean(v2_errs):.2f} px
- **Baseline Top-1 (<=3px)**: {np.mean(np.array(v1_errs) <= 3.0)*100:.1f}%
- **Hard-Negative Top-1 (<=3px)**: {np.mean(np.array(v2_errs) <= 3.0)*100:.1f}%
"""
    with open("hard_negative_mining/report.md", "w") as f:
        f.write(report)
    print("\n" + report)

if __name__ == "__main__":
    mine_and_train()