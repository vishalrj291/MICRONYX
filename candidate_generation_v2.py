import cv2
import numpy as np

def generate_candidates(search_img, ref_img, top_k=100, min_distance=8):
    h_ref, w_ref = ref_img.shape
    corr_map = cv2.matchTemplate(search_img, ref_img, cv2.TM_CCOEFF_NORMED)
    
    candidates = []
    temp_map = corr_map.copy()
    
    for _ in range(top_k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_map)
        if max_val <= -1.0:
            break
        peak_x, peak_y = max_loc
        cx = peak_x + (w_ref / 2.0)
        cy = peak_y + (h_ref / 2.0)
        candidates.append((cx, cy, float(max_val)))
        
        # NMS suppression
        x1 = max(0, peak_x - min_distance)
        y1 = max(0, peak_y - min_distance)
        x2 = min(temp_map.shape[1], peak_x + min_distance + 1)
        y2 = min(temp_map.shape[0], peak_y + min_distance + 1)
        temp_map[y1:y2, x1:x2] = -1.0
        
    return candidates