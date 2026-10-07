"""Predict from trusted local model files and numeric flow-feature CSVs."""
import argparse
import joblib
import numpy as np
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--csv',required=True)
p.add_argument('--output',default='predictions.csv')
a=p.parse_args()
bundle=joblib.load(a.model) # Only load model files you trust: pickle can execute code.
x=pd.read_csv(a.csv); x.columns=x.columns.str.strip()
missing=set(bundle['features'])-set(x.columns)
if missing: raise ValueError(f'Missing features: {sorted(missing)}')
x=x[bundle['features']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
prob=bundle['pipeline'].predict_proba(x)[:,1]
pd.DataFrame({'attack_probability':prob,'predicted_attack':prob>=bundle['threshold']}).to_csv(a.output,index=False)
print('Saved',a.output)
