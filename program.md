# autoresearch — Lane Detection Hyperparameter Search

## Your role
You are an autonomous research agent optimizing a classical CV lane detection pipeline.
Your ONLY job: modify hyperparameters in `experiment.py`, run it, record the score, iterate.

## The task
Maximize the SCORE printed by `experiment.py`. Score range: 0–100. Higher is better.

Score components:
- **detection_rate** (60 pts): fraction of images where BOTH lanes detected
- **avg_coverage** (25 pts): how much of the ROI the detected lanes span
- **consistency** (15 pts): low slope variance across images (stable detection)

## The file you modify
`experiment.py` — ONLY edit the CONFIG section at the top (lines between the CONFIG markers).
Do NOT touch the pipeline functions or scoring code below.

## How to run an experiment
```bash
python3 experiment.py
```
Takes ~2-5 seconds per run.

## Workflow
1. Read `experiments_log.jsonl` to understand what has been tried
2. Reason about which parameters to change and why
3. Edit the CONFIG section in `experiment.py`
4. Run `python3 experiment.py`
5. If score improved: keep changes, move to next experiment
6. If score decreased: revert and try a different change
7. After each run, update `EXPERIMENT_NAME` and `EXPERIMENT_NOTES` with what you changed

## Parameter search space
| Parameter | Current | Search range | Notes |
|-----------|---------|--------------|-------|
| GAUSS_KERNEL | 5 | 3, 5, 7, 9 | Odd numbers only |
| HLS_WHITE_L | 200 | 170-220 | Too low = noise, too high = miss faint lanes |
| HLS_YELLOW_H_LOW | 15 | 10-20 | |
| HLS_YELLOW_H_HIGH | 35 | 30-45 | |
| HLS_YELLOW_S | 100 | 80-130 | |
| CANNY_LOW | 50 | 30-80 | Keep ratio to HIGH at 1:2 to 1:4 |
| CANNY_HIGH | 150 | 100-200 | |
| ROI_TOP_Y | 0.60 | 0.55-0.65 | Vanishing point height |
| ROI_TL_X | 0.45 | 0.40-0.50 | Top-left of trapezoid |
| ROI_TR_X | 0.55 | 0.50-0.60 | Top-right of trapezoid |
| HT_THRESHOLD | 20 | 10-40 | Lower = more lines detected |
| HT_MIN_LINE_LEN | 20 | 10-40 | |
| HT_MAX_LINE_GAP | 300 | 200-500 | Higher = better for dashed lanes |
| SLOPE_EPS | 0.3 | 0.2-0.5 | |
| RANSAC_THRESHOLD | 15 | 5-30 | |

## Strategy tips
- Change ONE parameter at a time (ablation study)
- If detection_rate is already 1.0, focus on coverage and consistency
- CANNY_LOW:CANNY_HIGH ratio should stay between 1:2 and 1:4
- GAUSS_KERNEL must be odd
- ROI_TL_X must be < ROI_TR_X
- Large HT_MAX_LINE_GAP helps with dashed lanes

## Baseline score
Run `python3 experiment.py` with default params to establish baseline.
Log baseline first, then start iterating.

## Stopping criteria
Run at least 15 experiments. Stop when score has not improved for 5 consecutive runs.

## Video Scoring (new)
Run `python3 video_pipeline.py --input all` to get temporal metrics.

| Metric | Description |
|--------|-------------|
| detection_rate | Fraction of frames with both lanes detected |
| temporal_consistency | 1 - (slope_std_left + slope_std_right) — higher = smoother |

For video optimization: focus on EMA_ALPHA (0.7–0.95) and RANSAC_THRESHOLD.
