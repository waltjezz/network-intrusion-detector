# Robust Machine Learning for Network Intrusion Detection

An independent learning project comparing flow-based binary classifiers and exploring the tradeoff between attack recall and false alerts. This is a runnable experiment framework, not a live IDS or a claim of a novel robust model.

## Research question
How does choosing an alert threshold to limit validation false positives affect attack detection, and how sensitive are two baseline models to noisy features?

## Start here (Windows PowerShell)
Install Python 3.11 or 3.12, extract this folder, and open a terminal in it.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --demo --output demo-results
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py --demo --output demo-results
```

The demo is generated synthetic classification data with arbitrary features. It tests the software; it does not measure network intrusion detection performance. Included demo outputs were produced by an actual run. Version information is in `demo-results/run.json`; dependency ranges permit newer compatible versions, so results may vary.

## Run a network benchmark
1. Visit the official CIC-IDS2017 page: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the machine-learning CSV dataset following the site's current access instructions, and extract CSV files into `data/cicids2017/`. The dataset is not bundled.
3. Run:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir data/cicids2017 --max-rows 300000 --output results
```

Files are read in sorted order up to the row cap. Early files can contain only benign traffic: if the script reports missing classes, increase the cap (for example 3000000), or place an appropriate mixed-class CSV subset in a separate directory. More rows need more RAM and time. Inspect `run.json` for class prevalence, row count and source paths. List of source paths describes discovered CSVs; the cap may mean only some were read. Keep a record of the exact files and cap you used.

The label column must be `Label`; `BENIGN` means normal, all other nonempty labels mean attack. Numeric feature columns are retained; IP, timestamp, flow ID, label-like and port columns are excluded. Invalid numbers become missing values. Exact duplicate features are removed, and conflicting labels for identical features are excluded. Imputation and scaling fit only on training data, using scikit-learn pipelines: https://scikit-learn.org/stable/common_pitfalls.html

## What happens
- Split into 60% training, 20% validation and 20% test, stratified by binary label.
- Train a class-weighted logistic regression and random forest.
- Choose the threshold with highest validation recall subject to validation FPR <= 1%. Override with `--target-fpr 0.02`.
- Evaluate both default and tuned thresholds on the untouched test split.
- Repeat evaluation with Gaussian feature noise at 0.1, 0.25 and 0.5 times training feature standard deviation. The same noise is reused for fair model comparisons.

## Read the outputs
| File | Meaning |
| --- | --- |
| `metrics.csv` | Precision, recall, F1, FPR, ROC AUC, average precision and confusion counts for every model/policy/noise condition |
| `robustness.png` | F1 and false-positive rate as noise increases |
| `run.json` | Data type, preprocessing exclusions, split sizes, prevalence, seed, versions and limitations |
| `*.joblib` | Pipeline, feature names and tuned threshold; load only trusted model files |

Attack is the positive class. Recall = TP/(TP+FN); precision = TP/(TP+FP); FPR = FP/(FP+TN). F1 balances precision and recall. Average precision summarizes the precision-recall ranking; ROC AUC measures ranking across both classes. A higher threshold generally reduces alerts but may miss attacks. The validation FPR constraint is not a test or production guarantee.

## Predict on flow records
Input must contain the same numeric feature columns used in training. It may contain extra columns.

```powershell
.\.venv\Scripts\python.exe predict.py --model results/random_forest.joblib --csv new_flows.csv --output predictions.csv
```

This classifies precomputed flow features; it does not sniff packets. Scores are not calibrated probabilities or proof that an attack occurred.

## Limits and next experiments
Random splits can leave related sessions in both train and test; exact deduplication does not remove all correlation. Before claiming generalization, implement day/session-separated evaluation with a separately held-out validation day. A limited CSV subset may omit attack families. Binary classification does not evaluate unknown attack detection.

Gaussian noise can produce physically impossible network feature combinations and is only a numerical sensitivity test. It is not an adversarial attack, realistic distribution shift, or proof of robustness. Future experiments should constrain perturbations to valid flow features, evaluate unseen attack families, repeat splits with several seeds and confidence intervals, and compare performance at matching false-positive operating points.

For a PhD application, describe work you have run and understood. Keep synthetic and benchmark outcomes separate; do not use demo scores as network results. A defensible initial description after running the demo is: "Implemented a reproducible intrusion-detection experiment framework with validation-based threshold selection and feature-noise sensitivity analysis; benchmark evaluation in progress."

Dataset citation: Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018. Consult the official dataset page for citation and usage details.
