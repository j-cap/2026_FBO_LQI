"""Read-only consolidation of frozen v5/v6 results; no new plant simulation."""
import json,sys,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.actuator_bandwidth import augment_discrete,design
P=ROOT/'results';out=ROOT.parent/'docs/analysis';out.mkdir(exist_ok=True)
old=json.loads((P/'actuator_diagnostics_v2/protocol.json').read_text());cc=old['controller']
x=[-2.196941312402487,2.972435535863042,1.8815552853047848]
# Reuse stored identified models. No regeneration or new admission.
gains=[]
for seed in [355238,987946]:
 models=json.loads((P/f'actuator_bandwidth_v1_pilot/seeds/{seed}/sources.json').read_text())['models']
 for cid in ['nominal_00','payload_00','tire_degraded_00']:
  for tau in [.1,.15,.2,.3,.4]:
   a,b=augment_discrete(np.array(models[cid]['A']),np.array(models[cid]['B']),tau,.05)
   k,kr,rho=design(a,b,x,cc['state_scale'],.05)
   gains.append(dict(seed=seed,client=cid,tau=tau,K_beta=k[0],K_r=k[1],K_delta=k[2],K_eta=k[3],k_ref=kr,radius=rho))
pd.DataFrame(gains).to_csv(out/'shared_weights_gains.csv',index=False)
# Only aligned new Sobol reference-screening points; no mixing search trajectories.
land=[];near=[]
for seed in [355238,987946]:
 for tau in [.1,.15,.2,.3,.4]:
  maps={}
  for cid in ['nominal_00','payload_00','tire_degraded_00']:
   d=json.loads((P/f'actuator_corrected_v5/seeds/{seed}/{cid}/references/{tau:g}.json').read_text())
   maps[cid]={tuple(r['xi']):r['calibration'] for r in d['candidates'] if 'new_sobol' in r['origins']}
   good=[v['cost'] for v in maps[cid].values() if v['feasible'] and np.isfinite(v['cost'])]
   near.append(dict(seed=seed,client=cid,tau=tau,points=len(maps[cid]),feasible=len(good),
       within5=sum(v<=1.05*min(good) for v in good),within1=sum(v<=1.01*min(good) for v in good)))
  for c1,c2 in itertools.combinations(maps,2):
   keys=sorted(set(maps[c1])&set(maps[c2]));keys=[k for k in keys if all(maps[c][k]['feasible'] for c in [c1,c2])]
   u=np.array([maps[c1][k]['cost'] for k in keys]);v=np.array([maps[c2][k]['cost'] for k in keys])
   s=u<=1.05*min(u);t=v<=1.05*min(v)
   land.append(dict(seed=seed,tau=tau,client1=c1,client2=c2,n_joint=len(keys),spearman=spearmanr(u,v).statistic,
       near5_intersection=int(sum(s&t)),near5_union=int(sum(s|t)),near5_jaccard=float(sum(s&t)/sum(s|t))))
pd.DataFrame(land).to_csv(out/'aligned_landscapes.csv',index=False)
pd.DataFrame(near).to_csv(out/'grid_near_optimal_counts.csv',index=False)
# One record per unique simulated target/tau/weight episode, not per donor label.
metrics=[]
for f in (P/'actuator_transfer_v6').glob('seeds/*/*/evaluations/*/[0-9][0-9][0-9].json'):
 d=json.loads(f.read_text());metrics.append(d['coarse'])
e=pd.DataFrame(metrics)
d=pd.read_csv(P/'actuator_transfer_v6/processed/transfers.csv')
summary=d.groupby(['method','budget']).agg(n=('gap','size'),success5=('success_5','sum'),success1=('success_1','sum'),
 median_gap=('gap','median'),worst_gap=('gap','max')).reset_index()
summary.to_csv(out/'transfer_consolidated.csv',index=False)
s=dict(unique_episodes=len(e),peak_beta_deg=float(np.rad2deg(e.max_beta.max())),peak_actual_steering_deg=float(np.rad2deg(e.max_delta.max())),
 peak_actual_rate=float(e.max_rate.max()),max_episode_saturation_fraction=float(e.saturation_fraction.max()),
 median_episode_saturation_fraction=float(e.saturation_fraction.median()),episodes_with_saturation=int(sum(e.saturation_fraction>0)),
 spearman_min=float(min(r['spearman'] for r in land)),spearman_median=float(np.median([r['spearman'] for r in land])),
 jaccard_min=float(min(r['near5_jaccard'] for r in land)),jaccard_median=float(np.median([r['near5_jaccard'] for r in land])),
 grid_within5_range=[min(r['within5'] for r in near),max(r['within5'] for r in near)],
 grid_within1_range=[min(r['within1'] for r in near),max(r['within1'] for r in near)],
 grid_feasible_range=[min(r['feasible'] for r in near),max(r['feasible'] for r in near)])
(out/'transfer_mechanism_summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2));print(pd.DataFrame(gains).query("seed==355238 and client=='nominal_00'").to_string(index=False))
