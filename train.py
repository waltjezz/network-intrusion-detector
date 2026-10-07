"""Reproducible flow-classification experiment; demo data are synthetic."""
import argparse, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42

def metrics(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)), false_positive_rate=float(fp / (fp + tn)), roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def choose_threshold(y, p, target):
    # Candidate thresholds use validation predictions only; >1 permits zero alerts.
    candidates = np.r_[0., np.unique(p), np.nextafter(1., 2.)]
    benign = np.sort(p[np.asarray(y) == 0]); attack = np.sort(p[np.asarray(y) == 1])
    fpr = (len(benign) - np.searchsorted(benign, candidates, side='left')) / len(benign)
    recall = (len(attack) - np.searchsorted(attack, candidates, side='left')) / len(attack)
    valid = np.flatnonzero(fpr <= target)
    best = valid[np.argmax(recall[valid])]
    return float(candidates[best])

def load_csv(directory, limit):
    files = sorted(Path(directory).rglob('*.csv'))
    if not files: raise ValueError('No CSV files found in data directory.')
    parts = []; remaining = limit
    for file in files:
        if remaining <= 0: break
        frame = pd.read_csv(file, nrows=remaining, low_memory=False)
        frame.columns = frame.columns.str.strip()
        parts.append(frame); remaining -= len(frame)
    df = pd.concat(parts, ignore_index=True)
    if 'Label' not in df: raise ValueError('CSV files must contain a Label column.')
    df = df.dropna(subset=['Label']).drop_duplicates()
    labels = df.pop('Label').astype(str).str.strip().str.upper()
    y = (labels != 'BENIGN').astype(int)
    # Remove identifiers that can leak capture/session identity or shortcut labels.
    drop = [c for c in df if any(s in c.lower() for s in ['ip', 'timestamp', 'flow id', 'label', 'port'])]
    X = df.drop(columns=drop).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    if X.shape[1] == 0 or y.nunique() != 2: raise ValueError('Need numeric features and both BENIGN and attack rows. Increase --max-rows or choose mixed CSVs.')
    # Conflicting labels on identical features are excluded, then identical features deduplicated.
    hashes = pd.util.hash_pandas_object(X, index=False)
    valid = ~hashes.isin(hashes.groupby(hashes).filter(lambda s: y.loc[s.index].nunique() > 1))
    keep = valid & ~hashes.duplicated()
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), [str(f) for f in files], drop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--data-dir')
    ap.add_argument('--max-rows', type=int, default=300000)
    ap.add_argument('--output', default='results')
    ap.add_argument('--target-fpr', type=float, default=.01)
    args = ap.parse_args()
    if args.demo == bool(args.data_dir): ap.error('Choose exactly one of --demo or --data-dir.')
    if not 0 <= args.target_fpr <= 1: ap.error('--target-fpr must be between 0 and 1.')
    if args.max_rows <= 0: ap.error('--max-rows must be positive.')
    if args.demo:
        X, y = make_classification(n_samples=5000, n_features=15, n_informative=8, weights=[.85,.15], class_sep=1.2, random_state=SEED)
        X = pd.DataFrame(X, columns=[f'synthetic_feature_{i}' for i in range(15)])
        y = pd.Series(y); sources = ['synthetic sklearn.make_classification']; dropped = []
    else: X, y, sources, dropped = load_csv(args.data_dir, args.max_rows)
    Xtrain, Xrest, ytrain, yrest = train_test_split(X,y,test_size=.4,stratify=y,random_state=SEED)
    Xval, Xtest, yval, ytest = train_test_split(Xrest,yrest,test_size=.5,stratify=yrest,random_state=SEED)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    # Scale perturbations using TRAINING variability, never test-derived statistics.
    scale = Xtrain.std().fillna(0).to_numpy()
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=Xtest.shape)
    models = {'logistic_regression':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED)), 'random_forest':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestClassifier(n_estimators=120,min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=-1))}
    rows=[]
    for name, model in models.items():
        model.fit(Xtrain,ytrain)
        threshold = choose_threshold(yval,model.predict_proba(Xval)[:,1],args.target_fpr)
        for severity in [0., .1, .25, .5]:
            shifted = Xtest.copy() if severity == 0 else Xtest + severity * noise * scale
            p = model.predict_proba(shifted)[:,1]
            for policy,t in [('default',.5),('validation_tuned',threshold)]:
                rows.append(dict(model=name,noise_severity=severity,policy=policy,threshold=t,**metrics(ytest,p,t)))
        joblib.dump({'pipeline':model,'threshold':threshold,'features':list(X.columns)},out / f'{name}.joblib')
    results=pd.DataFrame(rows); results.to_csv(out/'metrics.csv',index=False)
    metadata={'data_type':'SYNTHETIC DEMO' if args.demo else 'CSV BENCHMARK','seed':SEED,'sources':sources,'dropped_columns':dropped,'features':list(X.columns),'split':'stratified random 60/20/20; not time-separated','rows':len(X),'split_rows':{'train':len(Xtrain),'validation':len(Xval),'test':len(Xtest)},'attack_prevalence':float(y.mean()),'target_validation_fpr':args.target_fpr,'python':platform.python_version(),'sklearn':sklearn.__version__,'limitations':['Random splits may share correlated sessions across partitions.','Noise stress test is not a realistic adversarial attack or proof of robustness.','Validation FPR target is not guaranteed on test or production data.','CSV loading uses a sorted-file row cap; report coverage before interpreting results.']}
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for name in models:
        selected=results[(results.model==name)&(results.policy=='validation_tuned')]
        axes[0].plot(selected.noise_severity,selected.f1,'o-',label=name)
        axes[1].plot(selected.noise_severity,selected.false_positive_rate,'o-',label=name)
    for ax,title in zip(axes,['F1 under feature noise','False-positive rate under feature noise']):
        ax.set(xlabel='Noise / training feature standard deviation',ylabel=title,title=title); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(metadata['data_type']); fig.tight_layout(); fig.savefig(out/'robustness.png',dpi=160); plt.close(fig)
    print(metadata['data_type']); print(results[results.noise_severity==0].to_string(index=False)); print('Saved outputs to',out)

if __name__ == '__main__': main()
