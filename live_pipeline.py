"""
live_pipeline.py — Real-time Lane Detection on Webcam / Live Feed
Usage:
  python3 live_pipeline.py               # webcam (default)
  python3 live_pipeline.py --source 0    # explicit webcam index
  python3 live_pipeline.py --source data/udacity_lanes/test_videos/challenge.mp4

Controls:
  Q — quit
  S — save current frame
  + / - — increase/decrease EMA alpha
"""

import cv2
import numpy as np
import time
import argparse
import os

# Best hyperparameters from autoresearch
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
EMA_ALPHA         = 0.85


class LaneTracker:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.lm = self.lb = self.rm = self.rb = None
        self.miss = 0

    def update(self, lm, lb, rm, rb):
        if lm is not None and rm is not None:
            self.miss = 0
            if self.lm is None:
                self.lm, self.lb, self.rm, self.rb = lm, lb, rm, rb
            else:
                a = self.alpha
                self.lm = a * self.lm + (1-a) * lm
                self.lb = a * self.lb + (1-a) * lb
                self.rm = a * self.rm + (1-a) * rm
                self.rb = a * self.rb + (1-a) * rb
            return self.lm, self.lb, self.rm, self.rb, True
        else:
            self.miss += 1
            if self.miss > 15:
                self.lm = self.lb = self.rm = self.rb = None
            return self.lm, self.lb, self.rm, self.rb, False


def process_frame(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hls  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    H_ch, L_ch, S_ch = hls[:,:,0], hls[:,:,1], hls[:,:,2]
    white  = (L_ch > HLS_WHITE_L).astype(np.uint8) * 255
    yellow = ((H_ch >= HLS_YELLOW_H_LOW) & (H_ch <= HLS_YELLOW_H_HIGH) &
              (S_ch > HLS_YELLOW_S)).astype(np.uint8) * 255
    mask   = cv2.bitwise_or(white, yellow)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.GaussianBlur(masked, (GAUSS_KERNEL, GAUSS_KERNEL), 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    H, W = img_bgr.shape[:2]
    verts = np.array([[
        (int(ROI_BL_X*W), H), (int(ROI_TL_X*W), int(ROI_TOP_Y*H)),
        (int(ROI_TR_X*W), int(ROI_TOP_Y*H)), (int(ROI_BR_X*W), H),
    ]], dtype=np.int32)
    roi_m = np.zeros_like(edges)
    cv2.fillPoly(roi_m, verts, 255)
    masked_e = cv2.bitwise_and(edges, roi_m)

    lines = cv2.HoughLinesP(masked_e, HT_RHO, HT_THETA, HT_THRESHOLD,
                            minLineLength=HT_MIN_LINE_LEN, maxLineGap=HT_MAX_LINE_GAP)
    if lines is None:
        return None, None, None, None

    left_g, right_g = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        m = (y2-y1)/(x2-x1)
        b = y1 - m*x1
        lg = np.hypot(x2-x1, y2-y1)
        if m < -SLOPE_EPS:
            left_g.append((m, b, lg))
        elif m > SLOPE_EPS:
            right_g.append((m, b, lg))

    def fit(group):
        if not group:
            return None, None
        if len(group) < 2:
            return group[0][0], group[0][1]
        pts_x, pts_y = [], []
        for m, b, length in group:
            if abs(m) < 1e-6:
                continue
            y1r, y2r = H, int(ROI_TOP_Y*H)
            x1r, x2r = int((y1r-b)/m), int((y2r-b)/m)
            for t in np.linspace(0, 1, max(2, int(length//8))):
                pts_x.append(int(x1r+t*(x2r-x1r)))
                pts_y.append(int(y1r+t*(y2r-y1r)))
        if len(pts_x) < 4:
            return np.mean([g[0] for g in group]), np.mean([g[1] for g in group])
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
            r = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESHOLD,
                                max_trials=RANSAC_TRIALS, random_state=42)
            r.fit(np.array(pts_x).reshape(-1,1), np.array(pts_y))
            return float(r.estimator_.coef_[0]), float(r.estimator_.intercept_)
        except Exception:
            return np.mean([g[0] for g in group]), np.mean([g[1] for g in group])

    return *fit(left_g), *fit(right_g)


def draw(frame, lm, lb, rm, rb, fps, detected, alpha):
    H, W = frame.shape[:2]
    overlay = np.zeros_like(frame)
    for m, b, color in [(lm, lb, (255,80,80)), (rm, rb, (80,165,255))]:
        if m and abs(m) > 1e-6:
            yb, yt = H, int(ROI_TOP_Y*H)
            xb = max(0, min(W-1, int((yb-b)/m)))
            xt = max(0, min(W-1, int((yt-b)/m)))
            cv2.line(overlay, (xb,yb), (xt,yt), color, 10)
    out = cv2.addWeighted(frame, 0.8, overlay, 1.0, 0)
    status = "DETECTED" if detected else "SEARCHING"
    c = (0,255,0) if detected else (0,0,255)
    cv2.putText(out, status + "  " + str(int(fps)) + "fps  EMA=" + str(round(alpha, 2)),
                (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    cv2.putText(out, "Q=quit  S=save  +/-=EMA", (12, H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    return out


def main():
    global EMA_ALPHA
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0')
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Cannot open source: " + str(src))
        return

    tracker = LaneTracker(alpha=EMA_ALPHA)
    frame_ct = 0
    t_start = time.time()
    os.makedirs('output_frames', exist_ok=True)
    print("Lane detection running on: " + str(src) + "  |  Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm_r, lb_r, rm_r, rb_r = process_frame(frame)
        lm, lb, rm, rb, det = tracker.update(lm_r, lb_r, rm_r, rb_r)
        tracker.alpha = EMA_ALPHA

        frame_ct += 1
        elapsed = time.time() - t_start
        fps_live = frame_ct / elapsed if elapsed > 0.1 else 0

        out = draw(frame, lm, lb, rm, rb, fps_live, det, EMA_ALPHA)
        cv2.imshow('Lane Detection — Live', out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = 'output_frames/frame_' + str(frame_ct).zfill(5) + '.jpg'
            cv2.imwrite(fname, out)
            print("Saved: " + fname)
        elif key == ord('+'):
            EMA_ALPHA = min(0.99, EMA_ALPHA + 0.05)
            print("EMA alpha: " + str(round(EMA_ALPHA, 2)))
        elif key == ord('-'):
            EMA_ALPHA = max(0.0, EMA_ALPHA - 0.05)
            print("EMA alpha: " + str(round(EMA_ALPHA, 2)))

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    print("\nProcessed " + str(frame_ct) + " frames in " + str(round(elapsed, 1)) + "s  (" + str(int(frame_ct/elapsed)) + " fps avg)")


if __name__ == '__main__':
    main()
