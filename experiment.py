"""
experiment.py — Lane Detection Hyperparameter Experiment
AGENT: Modify the hyperparameters in the CONFIG section below.
Do NOT modify the pipeline functions or scoring logic.
Run this file, record the score, iterate.
"""

import cv2  # pip install opencv-python
import numpy as np
import glob
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — AGENT MODIFIES THESE HYPERPARAMETERS
# ============================================================
EXPERIMENT_NAME = "best_final"
EXPERIMENT_NOTES = "Final best config: CANNY_LOW=60, CANNY_HIGH=180 (exp6), HT_MAX_LINE_GAP=250 (exp13), HT_MIN_LINE_LEN=30 (exp19), RANSAC_TRIALS=500 (exp3), SLOPE_EPS=0.4 (exp4)"

# Preprocessing
GAUSS_KERNEL      = 5       # Gaussian blur kernel size (odd: 3,5,7,9)
HLS_WHITE_L       = 200     # HLS white lane threshold (L channel, 0-255)
HLS_YELLOW_H_LOW  = 15      # HLS yellow hue lower bound (0-180)
HLS_YELLOW_H_HIGH = 35      # HLS yellow hue upper bound (0-180)
HLS_YELLOW_S      = 100     # HLS yellow saturation minimum (0-255)

# Canny edge detection
CANNY_LOW         = 60      # Lower threshold (should be ~1/3 of CANNY_HIGH)
CANNY_HIGH        = 180     # Upper threshold

# ROI trapezoid (fractions of image W/H)
ROI_BL_X          = 0.10   # bottom-left x
ROI_TL_X          = 0.45   # top-left x
ROI_TR_X          = 0.55   # top-right x
ROI_BR_X          = 0.95   # bottom-right x
ROI_TOP_Y         = 0.60   # top y (vanishing point fraction)

# Hough Transform
HT_RHO            = 1       # Distance resolution (pixels)
HT_THETA_DEG      = 1.0     # Angle resolution (degrees)
HT_THRESHOLD      = 20      # Minimum votes
HT_MIN_LINE_LEN   = 30      # Minimum line length (pixels)
HT_MAX_LINE_GAP   = 250     # Maximum gap to bridge (pixels)

# Slope classification
SLOPE_EPS         = 0.4     # |slope| threshold to discard horizontal segments

# RANSAC fitting
RANSAC_THRESHOLD  = 15      # Inlier residual threshold (pixels)
RANSAC_TRIALS     = 500     # Maximum iterations

# ============================================================
# PIPELINE (do not modify)
# ============================================================

def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hls  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    H_ch, L_ch, S_ch = hls[:,:,0], hls[:,:,1], hls[:,:,2]
    white  = (L_ch > HLS_WHITE_L).astype(np.uint8) * 255
    yellow = ((H_ch >= HLS_YELLOW_H_LOW) & (H_ch <= HLS_YELLOW_H_HIGH) &
              (S_ch > HLS_YELLOW_S)).astype(np.uint8) * 255
    mask    = cv2.bitwise_or(white, yellow)
    masked  = cv2.bitwise_and(gray, gray, mask=mask)
    k = GAUSS_KERNEL if GAUSS_KERNEL % 2 == 1 else GAUSS_KERNEL + 1
    return cv2.GaussianBlur(masked, (k, k), 0)


def roi_mask(img_shape):
    H, W = img_shape[:2]
    return np.array([[
        (int(ROI_BL_X * W), H),
        (int(ROI_TL_X * W), int(ROI_TOP_Y * H)),
        (int(ROI_TR_X * W), int(ROI_TOP_Y * H)),
        (int(ROI_BR_X * W), H),
    ]], dtype=np.int32)


def run_pipeline(img_bgr):
    """Returns (left_slope, left_int, right_slope, right_int, n_left, n_right)."""
    blurred = preprocess(img_bgr)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    verts   = roi_mask(edges.shape)
    mask    = np.zeros_like(edges)
    cv2.fillPoly(mask, verts, 255)
    masked  = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, HT_RHO, HT_THETA_DEG * np.pi / 180,
                            HT_THRESHOLD, minLineLength=HT_MIN_LINE_LEN,
                            maxLineGap=HT_MAX_LINE_GAP)
    if lines is None or len(lines) == 0:
        return None, None, None, None, 0, 0

    left, right = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        length = np.hypot(x2 - x1, y2 - y1)
        if m < -SLOPE_EPS:
            left.append((m, b, length))
        elif m > SLOPE_EPS:
            right.append((m, b, length))

    def fit(group, H):
        if len(group) < 2:
            if group:
                return group[0][0], group[0][1]
            return None, None
        pts_x, pts_y = [], []
        for m, b, length in group:
            if abs(m) < 1e-6:
                continue
            y1_r, y2_r = H, int(ROI_TOP_Y * H)
            x1_r = int((y1_r - b) / m)
            x2_r = int((y2_r - b) / m)
            for t in np.linspace(0, 1, max(2, int(length // 8))):
                pts_x.append(int(x1_r + t * (x2_r - x1_r)))
                pts_y.append(int(y1_r + t * (y2_r - y1_r)))
        if len(pts_x) < 4:
            return np.mean([g[0] for g in group]), np.mean([g[1] for g in group])
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
            ransac = RANSACRegressor(LinearRegression(),
                                     residual_threshold=RANSAC_THRESHOLD,
                                     max_trials=RANSAC_TRIALS, random_state=42)
            X = np.array(pts_x).reshape(-1, 1)
            y_arr = np.array(pts_y)
            ransac.fit(X, y_arr)
            return float(ransac.estimator_.coef_[0]), float(ransac.estimator_.intercept_)
        except Exception:
            return np.mean([g[0] for g in group]), np.mean([g[1] for g in group])

    H = img_bgr.shape[0]
    lm, lb = fit(left, H)
    rm, rb = fit(right, H)
    return lm, lb, rm, rb, len(left), len(right)


# ============================================================
# SCORING (do not modify)
# ============================================================

def compute_score(image_paths):
    """
    Score = weighted combination of:
    - detection_rate: fraction of images with both lanes detected (weight 60)
    - coverage_score: average span of detected lanes over ROI height (weight 25)
    - consistency:    low slope variance across images (weight 15)

    Higher is better. Maximum possible = 100.
    """
    results = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        H = img.shape[0]
        lm, lb, rm, rb, nl, nr = run_pipeline(img)
        detected = lm is not None and rm is not None and nl > 0 and nr > 0

        coverage = 0.0
        if detected:
            y_bot = H
            y_top = int(ROI_TOP_Y * H)
            roi_height = y_bot - y_top
            if abs(lm) > 1e-6:
                lx_bot = (y_bot - lb) / lm
                lx_top = (y_top - lb) / lm
                l_span = abs(lx_bot - lx_top)
            else:
                l_span = 0
            if abs(rm) > 1e-6:
                rx_bot = (y_bot - rb) / rm
                rx_top = (y_top - rb) / rm
                r_span = abs(rx_bot - rx_top)
            else:
                r_span = 0
            # Normalize: expected span ~300-400px for 960px wide image
            coverage = min(1.0, (l_span + r_span) / (2 * roi_height * 1.5))

        results.append({
            'path': path,
            'detected': detected,
            'coverage': coverage,
            'left_slope': lm,
            'right_slope': rm,
            'n_left': nl,
            'n_right': nr,
        })

    n = len(results)
    if n == 0:
        return 0.0, {}

    detection_rate = sum(r['detected'] for r in results) / n
    avg_coverage   = np.mean([r['coverage'] for r in results])

    detected_slopes_l = [r['left_slope'] for r in results if r['detected'] and r['left_slope'] is not None]
    detected_slopes_r = [r['right_slope'] for r in results if r['detected'] and r['right_slope'] is not None]
    slope_std_l = np.std(detected_slopes_l) if len(detected_slopes_l) > 1 else 0.5
    slope_std_r = np.std(detected_slopes_r) if len(detected_slopes_r) > 1 else 0.5
    consistency = max(0.0, 1.0 - (slope_std_l + slope_std_r))  # higher std = lower score

    score = (detection_rate * 60.0) + (avg_coverage * 25.0) + (consistency * 15.0)

    metrics = {
        'detection_rate': round(detection_rate, 4),
        'avg_coverage':   round(float(avg_coverage), 4),
        'consistency':    round(float(consistency), 4),
        'score':          round(float(score), 4),
        'n_images':       n,
        'n_detected':     int(sum(r['detected'] for r in results)),
    }
    return score, metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    image_paths = sorted(
        glob.glob('data/udacity_lanes/test_images/*.jpg') +
        glob.glob('data/udacity_lanes/test_images/*.png') +
        glob.glob('test_images/extra/*.jpg') +
        glob.glob('test_images/extra/*.png')
    )

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Notes: {EXPERIMENT_NOTES}")
    print(f"Images: {len(image_paths)}")
    print()

    t0 = time.time()
    score, metrics = compute_score(image_paths)
    elapsed = time.time() - t0

    print(f"SCORE: {score:.4f}")
    print(f"  detection_rate : {metrics['detection_rate']:.4f}  ({metrics['n_detected']}/{metrics['n_images']})")
    print(f"  avg_coverage   : {metrics['avg_coverage']:.4f}")
    print(f"  consistency    : {metrics['consistency']:.4f}")
    print(f"  elapsed        : {elapsed:.2f}s")

    # Log to JSONL
    log_entry = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'name': EXPERIMENT_NAME,
        'notes': EXPERIMENT_NOTES,
        'elapsed_s': round(elapsed, 2),
        'params': {
            'GAUSS_KERNEL': GAUSS_KERNEL,
            'HLS_WHITE_L': HLS_WHITE_L,
            'HLS_YELLOW_H_LOW': HLS_YELLOW_H_LOW,
            'HLS_YELLOW_H_HIGH': HLS_YELLOW_H_HIGH,
            'HLS_YELLOW_S': HLS_YELLOW_S,
            'CANNY_LOW': CANNY_LOW,
            'CANNY_HIGH': CANNY_HIGH,
            'ROI_BL_X': ROI_BL_X, 'ROI_TL_X': ROI_TL_X,
            'ROI_TR_X': ROI_TR_X, 'ROI_BR_X': ROI_BR_X,
            'ROI_TOP_Y': ROI_TOP_Y,
            'HT_THRESHOLD': HT_THRESHOLD,
            'HT_MIN_LINE_LEN': HT_MIN_LINE_LEN,
            'HT_MAX_LINE_GAP': HT_MAX_LINE_GAP,
            'SLOPE_EPS': SLOPE_EPS,
            'RANSAC_THRESHOLD': RANSAC_THRESHOLD,
            'RANSAC_TRIALS': RANSAC_TRIALS,
        },
        **metrics,
    }

    log_path = 'experiments_log.jsonl'
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    print(f"\nLogged to {log_path}")
