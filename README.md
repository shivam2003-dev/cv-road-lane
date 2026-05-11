# Problem 2 — Straight Lane Line Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-green?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20fitting-orange)
![Assignment](https://img.shields.io/badge/Course-AIMLCZG525-purple)

**Course**: Computer Vision (AIMLCZG525) | **Assignment 1**

Classical Computer Vision pipeline for detecting straight lane lines in dashcam images, combining Canny edge detection, Hough Transform, and three ML-based line fitting methods (Simple Averaging, RANSAC, K-Means).

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LANE LINE DETECTION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

  Input Image (960×540 px, BGR)
       │
       ▼
  ┌─────────────────────────────────────┐
  │  PREPROCESSING                      │
  │  ├─ BGR → HLS color space           │
  │  ├─ White mask  (L > 200)           │
  │  ├─ Yellow mask (H∈[15°,35°],S>100) │
  │  └─ Gaussian Blur (kernel=5×5, σ=0) │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  CANNY EDGE DETECTION            │
  │  τ_low = 50 │ τ_high = 150 (1:3)│
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  ROI MASKING                     │
  │  Trapezoid: (0.1W,H)→(0.45W,    │
  │  0.6H)→(0.55W,0.6H)→(0.95W,H)  │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  HOUGH TRANSFORM (Probabilistic) │
  │  ρ=1px │ θ=1° │ thresh=20       │
  │  minLen=20px │ maxGap=300px      │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  SLOPE CLASSIFICATION (ε = 0.3)  │
  │  m < -0.3  →  Left lane group    │
  │  m > +0.3  →  Right lane group   │
  │  |m| ≤ 0.3 →  Discarded (noise) │
  └──────────────┬───────────────────┘
                 │
          ┌──────┴───────┐
          ▼              ▼
      Left group    Right group
          │              │
          └──────┬────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────┐
  │  ML LINE FITTING                                 │
  │  A. Simple Averaging  — fast, no outlier reject  │
  │  B. RANSAC (default)  — robust, BDP ~50%         │
  │  C. K-Means (k=1)     — centroid = avg for k=1   │
  └──────────────┬───────────────────────────────────┘
                 │
                 ▼
  Lane Overlay on Original Image
```

---

## Repository Structure

```
cv_assignment/
├── CV_assignment1_group_problem2.ipynb   # Main notebook (33 cells, fully executed)
├── data/
│   └── udacity_lanes/
│       └── test_images/                  # 6 road images (960×540 px)
│           ├── solidWhiteCurve.jpg
│           ├── solidWhiteRight.jpg
│           ├── solidYellowCurve.jpg
│           ├── solidYellowCurve2.jpg
│           ├── solidYellowLeft.jpg
│           └── whiteCarLaneSwitch.jpg
├── test_images/                          # Place own road image here
│   └── own_road.jpg                      # Personal photo for validation
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements**: Python 3.8+

```bash
# Clone this repository
git clone <repo-url>
cd cv_assignment

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The Udacity CarND dataset must be cloned separately (public domain):

```bash
git clone --depth 1 https://github.com/udacity/CarND-LaneLines-P1.git data/udacity_lanes
```

This places the 6 test images at `data/udacity_lanes/test_images/`.

---

## Quick Start

**Run the notebook interactively:**

```bash
jupyter notebook CV_assignment1_group_problem2.ipynb
```

**Run all cells in sequence (headless):**

```bash
jupyter nbconvert --to notebook --execute CV_assignment1_group_problem2.ipynb \
    --output CV_assignment1_group_problem2_executed.ipynb
```

**Export to HTML for submission:**

```bash
jupyter nbconvert --to html CV_assignment1_group_problem2.ipynb
```

---

## Notebook Structure

The notebook is organized into 15 sections across 33 cells:

| # | Section | Description |
|---|---------|-------------|
| 1 | **Import Libraries** | OpenCV, NumPy, Matplotlib, scikit-learn, Pandas |
| 2 | **Data Acquisition** | Load 6 Udacity CarND dashcam images (960×540 px); display grid |
| 3 | **Preprocessing Pipeline** | BGR→HLS color masking (white + yellow) + Gaussian blur |
| 4 | **Canny Edge Detection** | Double-threshold hysteresis edge map (τ=50/150) |
| 5 | **ROI Masking** | Trapezoidal polygon mask isolating the road ahead |
| 6 | **Hough Transform** | Probabilistic Hough — raw line segment extraction |
| 7 | **Slope Classification** | Partition segments into left/right groups by slope sign |
| 8 | **ML Line Fitting** | Averaging, RANSAC, and K-Means — side-by-side comparison |
| 9 | **Full Pipeline Function** | Unified `lane_pipeline(img, method)` callable |
| 10 | **Inference on All Test Cases** | RANSAC pipeline run on all 6 images with overlay |
| 11 | **Own Image Validation** | All 3 methods applied to `test_images/own_road.jpg` |
| 12 | **Method Comparison Grid** | 3-image × 3-method visual comparison matrix |
| 13 | **Segment Count Bar Chart** | Left/right/total Hough segments per image |
| 14B | **Benchmarking** | Timing (5 runs × 6 images × 3 methods) + success rate analysis |
| 14/15 | **Analysis & Summary** | Threshold justification, limitations, improvement suggestions |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Udacity CarND-LaneLines-P1](https://github.com/udacity/CarND-LaneLines-P1) (public domain) |
| Images | 6 highway dashcam photographs |
| Resolution | 960 × 540 px |
| Lane types | Solid white, solid yellow, dashed white, dashed yellow |
| Conditions | Daytime, clear weather, straight road sections |

---

## ML Methods Comparison

| Method | Robustness | Speed | Breakdown Point | Notes |
|--------|-----------|-------|-----------------|-------|
| Simple Averaging | Low | Fastest | 0% | OLS — one outlier shifts estimate arbitrarily |
| **RANSAC** | **High** | Medium | **~50%** | **Recommended** — rejects shadow/dash outliers via inlier maximization |
| K-Means (k=1) | Low | Slowest | 0% | Centroid identical to arithmetic mean for k=1; EM overhead with no gain |

**Why RANSAC is preferred:**
Simple averaging minimises the OLS objective — optimal only under Gaussian noise. Lane detection produces non-Gaussian gross outliers (shadow edges, dashed-line fragments, guard-rail responses). RANSAC iteratively fits a candidate model on 2-point samples and counts inliers within residual threshold δ = 15 px, finding the maximum-inlier hypothesis. This is equivalent to an M-estimator with a hard-redescending weight function, yielding a breakdown point of ~50%.

---

## Key Parameters

### Preprocessing

| Parameter | Value | Justification |
|-----------|-------|---------------|
| White mask threshold | L > 200 | High lightness captures white markings regardless of hue |
| Yellow hue range | H ∈ [15°, 35°] | OpenCV HLS uses H ∈ [0°, 180°]; yellow sits at ~22° |
| Yellow saturation | S > 100 | Excludes pale/washed-out pixels while keeping vivid lane paint |
| Gaussian kernel | 5 × 5, σ = 0 | Smooths salt-and-pepper noise; σ = 0 lets OpenCV auto-derive sigma |

### Canny Edge Detection

| Parameter | Value | Justification |
|-----------|-------|---------------|
| τ_low | 50 | Low enough to maintain hysteresis continuity through faded dashes |
| τ_high | 150 | Suppresses asphalt texture and tar seams (typically < 120 gradient magnitude) |
| Ratio | 1:3 | Upper end of Canny's recommended 2:1–3:1 band — produces clean, thin edges |

### ROI Trapezoid

| Vertex | Coordinates | Role |
|--------|-------------|------|
| Bottom-left | (0.10W, H) | Excludes left road shoulder |
| Top-left | (0.45W, 0.60H) | Vanishing zone — left edge |
| Top-right | (0.55W, 0.60H) | Vanishing zone — right edge |
| Bottom-right | (0.95W, H) | Excludes right road shoulder |

### Hough Transform

| Parameter | Value | Justification |
|-----------|-------|---------------|
| ρ | 1 px | Pixel-level distance resolution |
| θ | 1° | Sufficient angular resolution for lane angles |
| threshold | 20 votes | Retains faint markings, rejects single-pixel noise |
| minLineLength | 20 px | Paired with maxLineGap — ignores noise fragments |
| **maxLineGap** | **300 px** | US highway dashes have 100–250 px gaps after ROI crop |

---

## Results

All **6/6** test images processed successfully with RANSAC — both left and right lane lines detected in every case.

| Image | Left Segments | Right Segments | Status |
|-------|--------------|----------------|--------|
| solidWhiteCurve.jpg | detected | detected | Pass |
| solidWhiteRight.jpg | detected | detected | Pass |
| solidYellowCurve.jpg | detected | detected | Pass |
| solidYellowCurve2.jpg | detected | detected | Pass |
| solidYellowLeft.jpg | detected | detected | Pass |
| whiteCarLaneSwitch.jpg | detected | detected | Pass |

---

## Rubric Coverage

| Rubric Criterion | Notebook Cell(s) | Details |
|-----------------|-----------------|---------|
| Data acquisition & display | Cell 2 | 6-image grid, shape printout |
| Preprocessing with justification | Cells 3, 14 | HLS masking + blur; parameter table |
| Canny edge detection | Cells 4, 14 | τ = 50/150, ratio justification |
| ROI masking | Cell 5 | Trapezoid vertices, before/after visualisation |
| Hough Transform | Cells 6, 14 | Raw segments + parameter justification |
| Slope classification | Cell 7 | ε = 0.3, coloured segment visualisation |
| ML-based line fitting (3 methods) | Cell 8 | Averaging, RANSAC, K-Means — math + visual |
| Full pipeline function | Cell 9 | `lane_pipeline(img, method)` |
| Inference on all test images | Cell 10 | 6/6 pass with overlay |
| Validation on own test image | Cell 11 | `test_images/own_road.jpg` |
| Method comparison & metrics | Cells 12, 13 | Grid comparison + bar chart |
| Benchmarking (timing + accuracy) | Cell 14B | 5-run timing × 3 methods × 6 images |
| Analysis & discussion | Cell 14 | Justifications, limitations, improvements |

---

## Validation — Adding Your Own Road Image

1. Place a road photograph at `test_images/own_road.jpg` (any resolution).
2. Cell 11 auto-detects the file and runs all three methods on it.
3. If the file is absent, Cell 11 falls back to a dataset image and prints a warning.

The cell prints per-image detection statistics (total Hough segments, left/right split, discarded noise count) alongside a 4-panel visual (original + 3 method outputs).

```bash
# Quick check — run only the validation cell
jupyter nbconvert --to notebook --execute CV_assignment1_group_problem2.ipynb \
    --ExecutePreprocessor.cell_timeout=60
```

---

## Limitations

| Limitation | Description |
|-----------|-------------|
| Curved roads | Straight-line model diverges from the true lane at bends (visible on `solidYellowCurve*` images) |
| Shadows | Strong gradient boundaries from shadows survive the HLS mask and appear as spurious lane candidates |
| Lane changes | Intermediate slopes between left/right gates cause misclassification |
| Night / rain | Low L-channel contrast drops edge strength below Canny thresholds |
| Fixed ROI | Assumes constant camera mount height and pitch — fails on hills or inclines |
| Dashed lanes at distance | maxLineGap = 300 px may under-bridge gaps at higher speeds or camera zoom |

**Suggested improvements:** bird's-eye perspective transform before Hough; degree-2 polynomial fitting for curves; Kalman / EMA temporal filter across video frames; CLAHE illumination normalisation; CNN semantic segmentation as a preprocessing mask.

---

## Assignment Information

| Field | Value |
|-------|-------|
| Course | Computer Vision — AIMLCZG525 |
| Assignment | Assignment 1 |
| Problem | Problem 2 — Straight Lane Line Detection |
| Dataset | Udacity CarND-LaneLines-P1 (public domain) |
| Language | Python 3.8+ |
| Key libraries | OpenCV 4.13, scikit-learn, NumPy, Matplotlib, Pandas |
