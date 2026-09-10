import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from inference import sign_flip_pvalue, holm_adjust
P=Path(os.environ['PCAVCT_OUTPUT_DIR']).resolve()
assert json.loads((P/'manifest.json').read_text())['mode']=='full', 'Smoke-test outputs are not inferential results'
r=pd.read_csv(P/'row_predictions.csv')
assert not r.duplicated(['dose','repeat','target','arm','subject_id','source_order']).any()
assert r.groupby(['dose','repeat','target','arm']).size().eq(7879).all()
assert r.groupby(['dose','repeat','subject_id']).fold.nunique().eq(1).all()
r['ae']=abs(r.prediction-r.observed);r['retention_ae']=abs(r.baseline-r.observed)
p=r.groupby(['dose','target','arm','subject_id','fst_band'])[['ae','retention_ae']].mean().reset_index()
p.to_csv(P/'participant_errors.csv',index=False)
def ci(v,level=.95):
 v=np.asarray(v);rng=np.random.default_rng(20260910)
 d=np.concatenate([v[rng.integers(len(v),size=(500,len(v)))].mean(axis=1) for _ in range(20)])
 return np.quantile(d,[(1-level)/2,1-(1-level)/2]).tolist()
summary=[]
for key,g in p.groupby(['dose','target','arm']):
 summary.append(dict(dose=int(key[0]),target=key[1],arm=key[2],mae=g.ae.mean(),ci=ci(g.ae),retention=g.retention_ae.mean(),gain_retention=(g.retention_ae-g.ae).mean(),gain_retention_ci=ci(g.retention_ae-g.ae),helped=float((g.retention_ae>g.ae).mean())))
contrasts=[];paired=[]
for target in ['sbp','dbp']:
 w=p[(p.dose==3)&(p.target==target)].pivot(index='subject_id',columns='arm',values='ae')
 for arm in ['wearable_state','full_historical_context','sensor_only']:
  d=w.no_context-w[arm]
  z=dict(target=target,arm=arm,gain=d.mean(),ci=ci(d),ci97_5=ci(d,.975))
  if arm=='wearable_state':
   z['paired_signflip_p']=sign_flip_pvalue(d);z['paired_t_p']=float(stats.ttest_1samp(d,0).pvalue)
   paired.append(pd.DataFrame({'subject_id':d.index,'target':target,'no_context_mae':w.no_context,'wearable_mae':w[arm],'contrast':d}))
  contrasts.append(z)
primary=[c for c in contrasts if c['arm']=='wearable_state']
for key in ['paired_signflip_p','paired_t_p']:
 for c,adjusted in zip(primary,holm_adjust([c[key] for c in primary])):c[key+'_holm']=float(adjusted)
pd.concat(paired).to_csv(P/'primary_paired_contrasts.csv',index=False)
subgroup=[]
for (target,band),g in p[(p.dose==3)&(p.arm=='wearable_state')].groupby(['target','fst_band']):
 subgroup.append(dict(target=target,band=band,n=len(g),mae=g.ae.mean(),ci=ci(g.ae)))
dosecontrast=[]
for target in ['sbp','dbp']:
 w=p[(p.target==target)&(p.arm=='full_historical_context')].pivot(index='subject_id',columns='dose',values='ae');d=w[1]-w[3]
 dosecontrast.append(dict(target=target,gain=d.mean(),ci=ci(d)))
out=dict(summary=summary,contrasts=contrasts,subgroup=subgroup,dosecontrast=dosecontrast)
(P/'statistics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(primary,indent=2))
