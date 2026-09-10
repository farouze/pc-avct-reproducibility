"""Frozen V26 analysis rerun from local feature banks; no outcome-based tuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json, hashlib, platform, re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedKFold
import legacy_helpers as legacy
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['PCAVCT_DATA_DIR']).resolve()
OUT=Path(os.environ['PCAVCT_OUTPUT_DIR']).resolve(); OUT.mkdir(parents=True,exist_ok=True)
SMOKE=os.environ.get('PCAVCT_MODE')=='smoke'
SEED=42; WINDOW_SECONDS=10; V26_TOP_K=24
V26_CAL_PATTERN=r'^\s*calibration\s+start'; V26_ANY_CAL_PATTERN=r'\bcalibration\b'
RAW=legacy.RAW; PC=legacy.PC; DELTA_RAW=legacy.DELTA_RAW; RELATIVE_RAW=legacy.RELATIVE_RAW
V12_PARAMS=legacy.V12_PARAMS; V10_ALPHA_GRID=legacy.V10_ALPHA_GRID; V10_THRESHOLD_GRID=legacy.V10_THRESHOLD_GRID
exec(compile((HERE/'original_v26_functions.py').read_text(),str(HERE/'original_v26_functions.py'),'exec'))
legacy.UCI_CACHE=ROOT/'uci_pc_features.csv'
aurora_path=ROOT/'aurora_raw_avct_features.csv'
aurora_features=pd.read_csv(aurora_path)
uci_prior_models,uci_prior_features=legacy.fit_uci_priors()
frames={k:v26_prepare_dose(k)[0] for k in [1,2,3]}
keys=set.intersection(*(set(v26_row_key(f)) for f in frames.values()))
for k,f in frames.items():
 f=f.loc[[v in keys for v in v26_row_key(f)]].copy()
 frames[k]=f.loc[~f.v26_is_calibration_labeled].reset_index(drop=True)
 assert frames[k].subject_id.nunique()==653 and len(frames[k])==7879
context=['context_phase_initial','context_phase_return','context_posture_seated','context_activity_post_exercise','context_duration_seconds','context_pressure_quality','context_optical_quality']
arms={'no_context':[], 'sensor_only':['context_optical_quality'],'wearable_state':['context_optical_quality','context_posture_seated','context_activity_post_exercise'],'full_historical_context':context}
records=[]; selections=[]
for dose in ([3] if SMOKE else [3,1,2]):
 for repeat in range(1 if SMOKE else 3):
  for fold,train,test in v26_participant_folds(frames[dose],5,repeat):
   if SMOKE and fold > 1: break
   for target in ['sbp','dbp']:
    core=v26_core_features(train,target)
    activearms=arms if dose==3 else {'full_historical_context':context}
    for arm,extra in activearms.items():
     features=[f for f in dict.fromkeys(core+extra) if f in train and train[f].notna().any() and train[f].nunique()>1]
     fit=train.dropna(subset=['delta_'+target]); model=make_model_v17(V12_PARAMS)
     model.fit(fit[features],fit['delta_'+target],histgradientboostingregressor__sample_weight=abl_weights(fit,'fst_band',False))
     gate=abl_fit_gate(fit,target,'baseline_'+target,model.predict(fit[features]),'fst_band',False)
     pred,active=abl_apply_gate(test,'baseline_'+target,model.predict(test[features]),gate,'fst_band')
     r=test[['subject_id','_source_order','fst_band',target,'baseline_'+target]].copy()
     r.columns=['subject_id','source_order','fst_band','observed','baseline'];r['prediction']=pred
     r['dose']=dose;r['repeat']=repeat+1;r['fold']=fold;r['target']=target;r['arm']=arm
     records.append(r)
     selections.append(dict(dose=dose,repeat=repeat+1,fold=fold,target=target,arm=arm,features=features,gate=gate))
   print(f'dose {dose} repeat {repeat+1} fold {fold} complete',flush=True)
   pd.concat(records).to_csv(OUT/'row_predictions.csv',index=False)
   (OUT/'selected_models.json').write_text(json.dumps(selections,indent=2))
pred=pd.concat(records)
pred['ae']=abs(pred.prediction-pred.observed);pred['retention_ae']=abs(pred.baseline-pred.observed)
per=pred.groupby(['dose','repeat','target','arm','subject_id','fst_band'],observed=True)[['ae','retention_ae']].mean().reset_index()
per.to_csv(OUT/'participant_repeat_errors.csv',index=False)
manifest=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,participants=653,rows=7879,repeats=1 if SMOKE else 3,folds_completed=1 if SMOKE else 5,seed=42,mode='smoke' if SMOKE else 'full',input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [aurora_path,legacy.UCI_CACHE]},scope='Full model-analysis rerun from cached feature banks; waveform extraction not rerun')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
