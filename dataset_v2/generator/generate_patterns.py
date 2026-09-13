import cv2
import numpy as np
import os
import pandas as pd

def add_sem_noise_and_artifacts(img, blur_ksize=3, noise_sigma=15, contrast_factor=0.8):
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.convertScaleAbs(noisy, alpha=contrast_factor, beta=10)

def generate_sample(sample_id, pattern_type="periodic", img_size=256, patch_size=32):
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    pitch = np.random.choice([8, 12, 16])
    
    if pattern_type == "periodic":
        for x in range(0, img_size, pitch):
            cv2.line(canvas, (x, 0), (x, img_size), 180, 1)
        for y in range(0, img_size, pitch):
            cv2.line(canvas, (0, y), (img_size, y), 180, 1)
    else:
        for _ in range(20):
            pt1 = tuple(np.random.randint(0, img_size, size=2))
            pt2 = tuple(np.random.randint(0, img_size, size=2))
            cv2.line(canvas, pt1, pt2, 200, 2)
            
    true_x = float(np.random.randint(patch_size, img_size - patch_size))
    true_y = float(np.random.randint(patch_size, img_size - patch_size))
    
    # Target feature draw karein
    cv2.circle(canvas, (int(true_x), int(true_y)), 3, 255, -1)
    
    half_p = patch_size // 2
    ref_patch = canvas[int(true_y - half_p):int(true_y + half_p), int(true_x - half_p):int(true_x + half_p)].copy()
    search_img = add_sem_noise_and_artifacts(canvas)
    
    metadata = {
        "sample_id": sample_id,
        "pattern_type": pattern_type,
        "pitch": int(pitch),
        "true_x": true_x,
        "true_y": true_y,
        "difficulty": "hard" if pattern_type == "periodic" else "medium"
    }
    return search_img, ref_patch, metadata

def main():
    splits = {"train": 40, "validation": 10, "test": 20}
    for split, count in splits.items():
        meta_list = []
        out_dir = os.path.join("dataset_v2", split)
        for i in range(count):
            s_id = f"{split}_{i:04d}"
            p_type = "periodic" if i % 2 == 0 else "aperiodic"
            search, ref, meta = generate_sample(s_id, p_type)
            cv2.imwrite(os.path.join(out_dir, f"{s_id}_search.png"), search)
            cv2.imwrite(os.path.join(out_dir, f"{s_id}_ref.png"), ref)
            meta_list.append(meta)
        pd.DataFrame(meta_list).to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
        print(f"Generated {count} samples in dataset_v2/{split}/")

if __name__ == "__main__":
    main()