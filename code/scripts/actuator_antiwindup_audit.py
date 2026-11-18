"""Paired replay of saved v3 episode reports; no optimization or reselection."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from actuator_bandwidth_experiment import context, write_json, valid
from actuator_diagnostics import make_banks, integration_check
from actuator_repair import digest, Progress
from src.actuator_antiwindup import evaluate
from src.episodes import build_multi_maneuver_episode_bank

PREVIOUS=ROOT/'results/actuator_repair_v3'
MODES=('legacy','sign_only','tracking')


def run_vehicle(seed,cid,files,old,cfg,out,smoke):
    progress=Progress(f'{seed}/{cid}');progress.say('reconstructing unchanged vehicle')
    fc,cc=old['fleet'],old['controller']
    fleet,models=context(seed,old['base'],fc,cc)
    client=next(c for c in fleet if c.client_id==cid);ac,bc=models[cid]
    banks=make_banks(client,old['config'],fc,cc)
    banks['test_new']=build_multi_maneuver_episode_bank(client,Ts=fc['Ts'],T_total=cc['T_total'],T0=cc['T0'],
        r_max=cc['r_max'],lane_change_time=fc['lane_change_time'],n_calibration_per_maneuver=cfg['test_per_maneuver'],
        n_test_per_maneuver=0,base_seed=cfg['test_seed'],tfilter=cc.get('tfilter'))
    for file in files:
        saved=json.loads(file.read_text());bank=banks[saved['role']]
        rows=saved['episodes']
        if smoke:
            wanted=187718234 if cid=='payload_00' else 1691183380
            rows=[r for r in rows if r['noise_seed']==wanted and r['maneuver']==1.]
            if len(rows)!=1:raise ValueError('Known regression episode missing')
        progress.say(f"{saved['label']}: {len(rows)} episodes, three update rules")
        for row in rows:
            target=out/'seeds'/str(seed)/cid/saved['label']/f"{row['index']:03d}.json"
            if target.exists():progress.hits+=1;continue
            matches=[e for e in bank.episodes if e.seed==row['noise_seed'] and e.T_lc==row['maneuver']]
            if len(matches)!=1:raise ValueError('Saved episode does not match reconstructed bank')
            one=replace(bank,episodes=matches);scores={}
            for mode in MODES:
                scores[mode]={}
                for steps,key in [(5,'coarse'),(10,'fine')]:
                    score=asdict(evaluate(client,ac,bc,saved['tau'],saved['xi'],one,'calibration',cc,fc['Ts'],steps,mode,.2))
                    scores[mode][key]=score
                    if mode=='legacy':
                        prior=row[key]
                        if score['feasible']!=prior['feasible'] or score['reason']!=prior['reason']:
                            raise ValueError(f'Legacy feasibility mismatch: {file} episode {row["index"]}')
                        for field in ['cost','max_beta','max_delta','max_rate','saturation_fraction','spectral_radius']:
                            if not np.isclose(score[field],prior[field],rtol=1e-7,atol=1e-10,equal_nan=True):
                                raise ValueError(f'Legacy {field} mismatch: {file} episode {row["index"]}')
                scores[mode]['integration']=integration_check(scores[mode]['coarse'],scores[mode]['fine'],.001)
            write_json(target,dict(seed=seed,client=cid,label=saved['label'],role=saved['role'],tau=saved['tau'],
                xi=saved['xi'],index=row['index'],noise_seed=row['noise_seed'],maneuver=row['maneuver'],scores=scores))
            progress.new+=1;progress.say(f"{saved['label']} episode {row['index']+1}",False)
        progress.say(f"completed {saved['label']}")


def analyze(out,expected):
    files=sorted(out.glob('seeds/*/*/*/*.json'))
    if len(files)!=expected:raise ValueError(f'Incomplete replay: {len(files)}/{expected} episodes; resume replay first')
    rows=[]
    for f in files:
        d=json.loads(f.read_text());identity={k:v for k,v in d.items() if k not in ['scores','xi']}
        for mode,scores in d['scores'].items():
            rows.append(dict(**identity,mode=mode,**scores['coarse'],integration_status=scores['integration']['status']))
    df=pd.DataFrame(rows);processed=out/'processed';processed.mkdir(exist_ok=True)
    df.to_csv(processed/'episodes.csv',index=False)
    summaries=[]
    for keys,g in df.groupby(['seed','client','label','role','tau','mode']):
        ok=g.feasible & np.isfinite(g.cost)
        summaries.append(dict(zip(['seed','client','label','role','tau','mode'],keys),episodes=len(g),
            feasible=bool(ok.all()),failed_episodes=int((~ok).sum()),
            cost=float(g.cost.mean()) if ok.all() else None,
            worst_feasible_cost=float(g.loc[ok,'cost'].max()) if ok.any() else None,
            max_beta=float(g.max_beta.max()),max_rate=float(g.max_rate.max()),
            integration_unresolved=int((g.integration_status!='pass').sum())))
    pd.DataFrame(summaries).to_csv(processed/'controllers.csv',index=False)
    print(f'Analysis complete: {len(files)} paired episodes. Infeasible banks have no aggregate cost.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','smoke','replay','analyze','collect'])
    p.add_argument('--n-jobs',type=int,default=2)
    a=p.parse_args();v3=json.loads((PREVIOUS/'protocol.json').read_text())
    oldpath=ROOT/v3['config']['previous_results']/'protocol.json';old=json.loads(oldpath.read_text())
    # Preserve fleet construction, synthesis and all previous simulation equations.
    for rel,h in {**old['code'],**v3['code'],**v3['prior_files']}.items():
        if digest(ROOT/rel)!=h:raise ValueError(f'Previous code/input changed: {rel}')
    files=sorted(PREVIOUS.glob('seeds/*/*/episodes/*.json'))
    if len(files)!=75:raise ValueError(f'Expected 75 saved controller reports, found {len(files)}')
    smoke=a.stage=='smoke'
    if smoke:
        files=[f for f in files if f.parts[-4]=='355238' and
               ((f.parts[-3]=='payload_00' and f.stem=='old_reference_0.2') or
                (f.parts[-3]=='tire_degraded_00' and f.stem=='new_reference_0.3'))]
        if len(files)!=2:raise ValueError('Known regression reports missing')
    expected=2 if smoke else sum(len(json.loads(f.read_text())['episodes']) for f in files)
    code=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/actuator_bandwidth_experiment.py',
        ROOT/'scripts/actuator_diagnostics.py',ROOT/'scripts/actuator_repair.py',ROOT/'scripts/phase2_fbo_comparison.py']
    protocol=dict(modes=list(MODES),tracking_time=.2,substeps=[5,10],smoke=smoke,expected_episodes=expected,
        code={f.relative_to(ROOT).as_posix():digest(f) for f in code},
        inputs={f.relative_to(ROOT).as_posix():digest(f) for f in files+[oldpath,PREVIOUS/'protocol.json']})
    if a.stage=='preflight':
        print(f'Preflight passed: {len(files)} saved controllers, {expected} episodes; previous hashes verified.');return
    out=ROOT/'results'/('actuator_antiwindup_v4'+('_smoke' if smoke else ''))
    manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Protocol mismatch; use a new namespace')
    write_json(manifest,protocol)
    if a.stage=='collect':
        analyze(out,expected)
        selected=[manifest]+sorted(out.glob('processed/*.csv'))+sorted(out.glob('seeds/*/*/*/*.json'))
        for i in range(0,len(selected),30):
            subprocess.run(['git','add','-f','--']+[str(f.relative_to(ROOT.parent)) for f in selected[i:i+30]],cwd=ROOT.parent,check=True)
        print(f'Staged {len(selected)} files; review, commit and push.');return
    if a.stage=='analyze':analyze(out,expected);return
    targets=sorted({(int(f.parts[-4]),f.parts[-3]) for f in files})
    with parallel_config(backend='loky',inner_max_num_threads=1):
        Parallel(n_jobs=a.n_jobs)(delayed(run_vehicle)(s,c,[f for f in files if f.parts[-4]==str(s) and f.parts[-3]==c],
            old,v3['config'],out,smoke) for s,c in targets)
    analyze(out,expected)


if __name__=='__main__':main()
