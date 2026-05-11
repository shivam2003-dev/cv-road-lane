"""
video_pipeline.py — Lane Detection on Video Files
Usage:
  python3 video_pipeline.py --input data/udacity_lanes/test_videos/solidWhiteRight.mp4
  python3 video_pipeline.py --input data/udacity_lanes/test_videos/challenge.mp4
  python3 video_pipeline.py --input all   (process all videos in test_videos/)
"""

import cv2
import numpy as np
import glob
import os
import time
import argparse
import json
from pathlib import Path

# ── Best hyperparameters from autoresearch ────────────────────────────────────
GAUSS_KERNEL      = 5
HLS_WHITE_L       = 200
HLS_YELLOW_H_LOW  = 15
HLS_YELLOW_H_HIGH = 35
HLS_YELLOW_S      = 100
CANNY_LOW         = 60
CANNY_HIGH        = 180
ROI_BL_X, ROI_BR_X = 0.10, 0.95
ROI_TL_X, ROI_TR_X = 0.45, 0.55
ROI_TOP_Y         = 0.60
HT_RHO            = 1
HT_THETA          = np.pi / 180
HT_THRESHOLD      = 20
HT_MIN_LINE_LEN   = 30
HT_MAX_LINE_GAP   = 250
SLOPE_EPS         = 0.4
RANSAC_THRESHOLD  = 15
RANSAC_TRIALS     = 500

# ── Temporal smoothing ────────────────────────────────────────────────────────
EMA_ALPHA = 0.85   # how much previous frame influences current (0=no smooth, 1=frozen)


# ── Pipeline functions ────────────────────────────────────────────────────────

def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hls  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    H_ch, L_ch, S_ch = hls[:,:,0], hls[:,:,1], hls[:,:,2]
    white  = (L_ch > HLS_WHITE_L).astype(np.uint8) * 255
    yellow = ((H_ch >= HLS_YELLOW_H_LOW) & (H_ch <= HLS_YELLOW_H_HIGH) &
              (S_ch > HLS_YELLOW_S)).astype(np.uint8) * 255
    mask   = cv2.bitwise_or(white, yellow)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    k = GAUSS_KERNEL if GAUSS_KERNEL % 2 == 1 else GAUSS_KERNEL + 1
    return cv2.GaussianBlur(masked, (k, k), 0)


def get_roi_verts(img_shape):
    H, W = img_shape[:2]
    return np.array([[
        (int(ROI_BL_X * W), H),
        (int(ROI_TL_X * W), int(ROI_TOP_Y * H)),
        (int(ROI_TR_X * W), int(ROI_TOP_Y * H)),
        (int(ROI_BR_X * W), H),
    ]], dtype=np.int32)


def fit_ransac(pts_x, pts_y):
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    X = np.array(pts_x).reshape(-1, 1)
    y = np.array(pts_y)
    ransac = RANSACRegressor(LinearRegression(),
                             residual_threshold=RANSAC_THRESHOLD,
                             max_trials=RANSAC_TRIALS, random_state=42)
    ransac.fit(X, y)
    return float(ransac.estimator_.coef_[0]), float(ransac.estimator_.intercept_)


def detect_lanes(img_bgr):
    """Raw lane detection — returns (lm, lb, rm, rb) or Nones."""
    blurred = preprocess(img_bgr)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    verts   = get_roi_verts(edges.shape)
    roi_m   = np.zeros_like(edges)
    cv2.fillPoly(roi_m, verts, 255)
    masked  = cv2.bitwise_and(edges, roi_m)

    lines = cv2.HoughLinesP(masked, HT_RHO, HT_THETA, HT_THRESHOLD,
                            minLineLength=HT_MIN_LINE_LEN,
                            maxLineGap=HT_MAX_LINE_GAP)
    if lines is None:
        return None, None, None, None

    left_g, right_g = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        length = np.hypot(x2 - x1, y2 - y1)
        if m < -SLOPE_EPS:
            left_g.append((m, b, length))
        elif m > SLOPE_EPS:
            right_g.append((m, b, length))

    def group_to_pts(group, H):
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
        return pts_x, pts_y

    H = img_bgr.shape[0]
    lm, lb, rm, rb = None, None, None, None

    if len(left_g) >= 2:
        try:
            px, py = group_to_pts(left_g, H)
            if len(px) >= 4:
                lm, lb = fit_ransac(px, py)
        except Exception:
            lm, lb = np.mean([g[0] for g in left_g]), np.mean([g[1] for g in left_g])
    elif left_g:
        lm, lb = left_g[0][0], left_g[0][1]

    if len(right_g) >= 2:
        try:
            px, py = group_to_pts(right_g, H)
            if len(px) >= 4:
                rm, rb = fit_ransac(px, py)
        except Exception:
            rm, rb = np.mean([g[0] for g in right_g]), np.mean([g[1] for g in right_g])
    elif right_g:
        rm, rb = right_g[0][0], right_g[0][1]

    return lm, lb, rm, rb


def draw_lanes_on_frame(frame_bgr, lm, lb, rm, rb):
    """Draw smoothed lane lines on BGR frame."""
    H, W = frame_bgr.shape[:2]
    overlay = np.zeros_like(frame_bgr)

    for m, b, color in [(lm, lb, (255, 80, 80)), (rm, rb, (80, 165, 255))]:
        if m is None or abs(m) < 1e-6:
            continue
        y_bot = H
        y_top = int(ROI_TOP_Y * H)
        x_bot = int((y_bot - b) / m)
        x_top = int((y_top - b) / m)
        # Clamp to image bounds
        x_bot = max(0, min(W - 1, x_bot))
        x_top = max(0, min(W - 1, x_top))
        cv2.line(overlay, (x_bot, y_bot), (x_top, y_top), color, 10)

    return cv2.addWeighted(frame_bgr, 0.8, overlay, 1.0, 0)


def add_hud(frame, frame_no, fps, detected, lm, rm, ema_alpha):
    """Overlay HUD info on frame."""
    H, W = frame.shape[:2]
    info = [
        f"Frame: {frame_no}",
        f"FPS: {fps:.1f}",
        f"Detected: {'YES' if detected else 'NO'}",
        f"EMA alpha: {ema_alpha}",
    ]
    if detected and lm is not None and rm is not None:
        info.append(f"L slope: {lm:.3f}")
        info.append(f"R slope: {rm:.3f}")

    color = (0, 255, 0) if detected else (0, 0, 255)
    for i, line in enumerate(info):
        cv2.putText(frame, line, (15, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


class LaneTracker:
    """Maintains EMA-smoothed lane parameters across frames."""
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.lm = self.lb = self.rm = self.rb = None
        self.miss_count = 0
        self.MAX_MISS = 15  # frames before resetting

    def update(self, lm, lb, rm, rb):
        detected = lm is not None and rm is not None
        if detected:
            self.miss_count = 0
            if self.lm is None:
                self.lm, self.lb = lm, lb
                self.rm, self.rb = rm, rb
            else:
                self.lm = self.alpha * self.lm + (1 - self.alpha) * lm
                self.lb = self.alpha * self.lb + (1 - self.alpha) * lb
                self.rm = self.alpha * self.rm + (1 - self.alpha) * rm
                self.rb = self.alpha * self.rb + (1 - self.alpha) * rb
        else:
            self.miss_count += 1
            if self.miss_count > self.MAX_MISS:
                self.lm = self.lb = self.rm = self.rb = None

        return self.lm, self.lb, self.rm, self.rb, detected


def process_video(input_path, output_path=None, show_progress=True):
    """Process a single video file. Returns metrics dict."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_path}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (W, H))

    tracker     = LaneTracker(alpha=EMA_ALPHA)
    frame_no    = 0
    detected_ct = 0
    t_start     = time.time()
    slopes_l, slopes_r = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect raw
        lm_raw, lb_raw, rm_raw, rb_raw = detect_lanes(frame)

        # Smooth with EMA
        lm, lb, rm, rb, detected = tracker.update(lm_raw, lb_raw, rm_raw, rb_raw)

        if detected:
            detected_ct += 1
            if lm is not None:
                slopes_l.append(lm)
            if rm is not None:
                slopes_r.append(rm)

        # Draw
        annotated = draw_lanes_on_frame(frame, lm, lb, rm, rb)
        elapsed   = time.time() - t_start
        fps_live  = frame_no / elapsed if elapsed > 0 else 0
        annotated = add_hud(annotated, frame_no, fps_live, detected, lm, rm, EMA_ALPHA)

        if writer:
            writer.write(annotated)

        frame_no += 1
        if show_progress and frame_no % 50 == 0:
            pct = frame_no / total * 100 if total > 0 else 0
            print(f"  [{pct:5.1f}%] frame {frame_no}/{total}  "
                  f"detected={detected_ct}  fps={fps_live:.1f}")

    cap.release()
    if writer:
        writer.release()

    elapsed_total = time.time() - t_start
    detection_rate = detected_ct / frame_no if frame_no > 0 else 0
    slope_std_l = float(np.std(slopes_l)) if len(slopes_l) > 1 else 0
    slope_std_r = float(np.std(slopes_r)) if len(slopes_r) > 1 else 0
    temporal_consistency = max(0.0, 1.0 - (slope_std_l + slope_std_r))

    metrics = {
        'input': input_path,
        'output': output_path,
        'total_frames': frame_no,
        'detected_frames': detected_ct,
        'detection_rate': round(detection_rate, 4),
        'temporal_consistency': round(temporal_consistency, 4),
        'slope_std_left': round(slope_std_l, 5),
        'slope_std_right': round(slope_std_r, 5),
        'fps_avg': round(frame_no / elapsed_total, 1),
        'duration_s': round(elapsed_total, 2),
    }
    return metrics


def main():
    global EMA_ALPHA
    parser = argparse.ArgumentParser(description='Lane detection on video files')
    parser.add_argument('--input', default='all',
                        help='Video path or "all" for all test videos')
    parser.add_argument('--output_dir', default='data/udacity_lanes/test_videos_output',
                        help='Directory for annotated output videos')
    parser.add_argument('--ema', type=float, default=EMA_ALPHA,
                        help=f'EMA smoothing alpha (default {EMA_ALPHA})')
    args = parser.parse_args()

    EMA_ALPHA = args.ema

    if args.input == 'all':
        video_paths = sorted(
            glob.glob('data/udacity_lanes/test_videos/*.mp4') +
            glob.glob('data/udacity_lanes/test_videos/*.avi')
        )
    else:
        video_paths = [args.input]

    all_metrics = []
    for vpath in video_paths:
        name = Path(vpath).stem
        out  = os.path.join(args.output_dir, f'{name}_annotated.mp4')
        print(f"\nProcessing: {vpath}")
        print(f"Output:     {out}")
        m = process_video(vpath, output_path=out)
        all_metrics.append(m)

        print(f"\nResults for {name}:")
        print(f"  Frames           : {m['total_frames']}")
        print(f"  Detection rate   : {m['detection_rate']:.4f}  ({m['detected_frames']}/{m['total_frames']})")
        print(f"  Temporal consist : {m['temporal_consistency']:.4f}")
        print(f"  Slope std (L/R)  : {m['slope_std_left']:.5f} / {m['slope_std_right']:.5f}")
        print(f"  Avg FPS          : {m['fps_avg']}")

    # Save metrics
    with open('video_results.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to video_results.json")

    # Summary table
    print(f"\n{'Video':<30} {'DetRate':>9} {'Consistency':>13} {'FPS':>6}")
    print("-" * 65)
    for m in all_metrics:
        name = Path(m['input']).stem
        print(f"{name:<30} {m['detection_rate']:>9.4f} {m['temporal_consistency']:>13.4f} {m['fps_avg']:>6.1f}")


if __name__ == '__main__':
    main()
