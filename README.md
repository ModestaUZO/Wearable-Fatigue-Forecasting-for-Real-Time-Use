# Wearable Fatigue Forecasting for Construction Workers

The aim is to forecast short‑horizon changes in worker fatigue using time‑series signals from wearables (HR, HRV, EMG, skin/ambient temperature). Predictions are given as normalized fatigue (0–1) and mapped to a human‑readable ROF scale (1–10) for early warnings. Predict short‑term worker fatigue (next 5, 15, 30, 60 minutes) from 60s windows of wearable physiological data. Intended for prototyping real‑time supervisor dashboards or automated early‑warning systems on worksites.

# Quick highlights / results

- **Model**: LSTM multi‑horizon forecaster (outputs 4 values: +5, +15, +30, +60 minutes).

- **Sample performance (demo run)**: Train MSE converged quickly (example run: Epoch 1 train MSE=0.0155 | val MSE=0.0002; later epochs ~0.0001).

- **Early‑warning metrics (HIGH when ROF≥8)**: AUROC ~0.99–1.00 at short horizons; high precision/recall in the provided demo folds. See notebook evaluation cells for full per‑horizon numbers.

# Files in this repo

- `Wearable Fatigue Forecasting for Construction Workers.ipynb` — main Jupyter notebook used for exploration, preprocessing, training, evaluation, and demo inference.

- `fatigue_lstm_mh.pt` — saved PyTorch state_dict (optional backup).

- `fatigue_lstm_mh.ts` — TorchScript trace usable in a lightweight runtime (for dashboards/services).

- `fatigue_site_timeseries.csv` —  dataset path referenced in the notebook.


# Data format

Expected CSV columns:

- **worker_id** — `int`

- **time_s** — `seconds from session start`

- **Physiological features**: `hr_bpm, hrv_rmssd, emg_level, skin_temp_c, ambient_temp_c`

- **fatigue_now** — `normalized label in [0,1]`

ROF mapping used in code: rof = round(1 + 9*f) where f is 0–1.


# Notebook structure & annotated cell notes

Below is a short, numbered walkthrough matching the notebook cells so you (or a reviewer) can quickly navigate the code and results.

**Load data**

- Reads CSV into df_all and prints a head. Check column names and dtypes here.

**Sequence builder** `(make_sequences_multi)`

- Converts per‑worker time series into overlapping windows of CONTEXT_LEN (60s) and builds multi‑horizon targets for +5/+15/+30/+60 minutes.

**Train/val split + DataLoaders**

- Splits sample windows 80/20, wraps into FatigueDataset and DataLoader.


**Model definition (LSTMForecasterMH)**

- Single‑layer LSTM + linear head producing NUM_H outputs. Default hidden_size=64.

**Train loop**

- Standard PyTorch training loop using MSELoss and Adam. A short demo run uses 5 epochs.

**Evaluation / Early‑warning**

- Converts continuous predictions to ROF bands and computes classification metrics (AUROC, PR‑AUC, precision/recall/F1, Brier score) for the HIGH class (ROF≥8).

**Per‑worker inference & utilities**

- Provides helper functions: predict_multi_horizon, quick_worker_check, fatigue_to_rof, rof_band and a short supervisor message generator.

**Save artifacts (TorchScript)**

- Saves state_dict and TorchScript trace under artifacts/.


# Model(s) used — why LSTM?

LSTM (Long Short‑Term Memory) was chosen because it is a compact, well‑understood recurrent model that captures short‑range temporal dependencies and is easy to train for moderate dataset sizes.

**Why sensible for this task**: physiological signals have temporal patterns and short‑term dynamics (heart rate trends, EMG bursts) that LSTM can model using a fixed context window.

**Alternatives to try**: Temporal convolutional networks (TCN), 1D CNNs for feature extraction, GRU (lighter RNN), Transformer / attention models (for longer contexts), or simple models like Random Forests on hand‑crafted features for baseline comparisons.

# Evaluation summary

Example evaluation outputs included in the notebook (demo split):

- Per‑epoch losses: `Epoch 1: train MSE=0.0155 | val MSE=0.0002`, later epochs `~0.0001`.

- Early‑warning (HIGH when ROF≥8) per horizon:

   - +5m: positives=1108 negatives=1017 — AUROC=0.999, PR‑AUC=0.999, Best‑F1≈0.986

   - +15m: positives=1476 negatives=649 — AUROC=0.999, PR‑AUC=1.000

   - +30m: positives=2041 negatives=84 — AUROC=0.994, PR‑AUC=1.000

   - +60m: positives=2125 negatives=0 — only positives present; Brier=0.012

**Interpretation**: demo model performs strongly on the provided split. However, check class balance and leakage; near‑perfect metrics may indicate dataset artifacts, label smoothing, or low variability across splits — see Challenges & Limitations.

# Challenges & limitations

- **Class imbalance & label distribution**: some horizons had very skewed class balances (e.g., only positives at +60m in the run) which prevents reliable discrimination metrics.

- **Overfitting / data leakage risk**: extremely low validation MSE and almost perfect AUROC can be a sign of leakage (e.g., temporal leakage when overlapping windows are split incorrectly) or dataset ease. Verify that validation windows are strictly separated from training sessions per worker/time.

- **Small dataset / subject variability**: generalization to new workers or different tasks is untested — require cross‑subject validation.

- **Sensor noise & missing data**: real wearable deployments have dropouts and noise; the notebook currently assumes clean data.

- **Label reliability**: fatigue_now originates from ROF prompts — subjective labels may be noisy and biased.


# Recommendations & future work

- **Stronger validation**: use leave‑one‑subject‑out (LOSO) splits to estimate cross‑subject generalization.

- **Calibration & uncertainty**: add probabilistic outputs (e.g., MC‑dropout, ensembling) and compute calibration metrics / reliability diagrams.

- **Feature engineering**: add time‑domain features (rolling means, slopes, spectral HRV features) and test a shallow model baseline (RF / XGBoost) for comparison.

- **Robustness to missing data**: implement imputation, masking, or a model that accepts variable length / masked input.

- **Deployability**: wrap TorchScript model in a small service (FastAPI) and push predictions to a dashboard. Add unit tests for inference helpers.

- **User studies**: test whether the early warnings lead to measurable safety improvements or behavioral changes on site.

# How to run

- Place `fatigue_site_timeseries.csv` under `data/` (or update path in notebook).

- Launch the notebook: `jupyter lab` or `jupyter notebook` and run cells sequentially.

- Train by running the training cell (adjust `epochs`, `batch_size` as desired).

- Use `quick_worker_check(worker_id)` to produce a supervisor note for a worker.


# Conclusion 

This project demonstrates a compact pipeline for short‑horizon fatigue forecasting using wearable signals. A multi‑horizon LSTM forecaster successfully maps 60s windows of physiological data to fatigue forecasts at +5, +15, +30 and +60 minutes and produces interpretable ROF alerts suitable for supervisor dashboards. While demo results are promising, careful validation (LOSO), mitigation of label noise, and robustness checks against sensor drift are required before field deployment. Future work should prioritize cross‑subject validation, calibration of predicted probabilities, and field trials to measure real safety impact.



Author: [Modesta I. Uzo](https://www.linkedin.com/in/modesta-uzo-04281923a/)
