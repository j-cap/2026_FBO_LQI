"""Source-only leave-one-vehicle-out transfer pilot with corrected anti-windup."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from joblib import Parallel,delayed,parallel_config
from actuator_bandwidth_experiment import context,valid,write_json
from actuator_corrected import digest,Progress,incumbent
from actuator_diagnostics import integration_check,make_banks
from src.actuator_antiwindup import evaluate
from src.episodes import build_multi_maneuver_episode_bank
PREVIOUS=ROOT/'results/actuator_corrected_v5'
COMMON=[-2.196941312402487,2.972435535863042,1.8815552853047848]


def transfer_plan(target,targets,histories):
    """No target outcomes or references enter this function. Retain ALL donors."""
    rows=[]
    for donor in targets:
        if tuple(donor)==tuple(target):continue
        for method in ['sobol','bo_lower_penalty']:
            for seed in [101,202,303]:
                records=histories[(tuple(donor),method,seed)]
                for budget in [25,75]:
                    if len(records)<budget:raise ValueError('Incomplete source history')
                    x=incumbent(records[:budget])
                    if x is None:raise ValueError('No feasible source incumbent')
                    rows.append(dict(donor_seed=donor[0],donor_client=donor[1],method=method,
                        optimizer_seed=seed,budget=budget,xi=x))
    return rows


def summarize(episodes):
    # Mean episode RMSE, matching original evaluator. Do not average away failures.
    ok=all(valid(r['coarse']) and valid(r['fine']) for r in episodes)
    converged=all(r['integration']['status']=='pass' for r in episodes)
    return dict(feasible=ok,integration_pass=converged,
        cost=float(np.mean([r['coarse']['cost'] for r in episodes])) if ok else None,
        failed_episodes=sum(not valid(r['coarse']) or not valid(r['fine']) for r in episodes),
        unresolved_episodes=sum(r['integration']['status']!='pass' for r in episodes))


def classify(score,reference):
    if not score['feasible']:return dict(status='infeasible',gap=None,success_1=False,success_5=False)
    if not score['integration_pass']:return dict(status='unresolved_integration',gap=None,success_1=False,success_5=False)
    if not reference['feasible'] or not reference['integration_pass']:
        return dict(status='unresolved_reference',gap=None,success_1=False,success_5=False)
    gap=score['cost']/reference['cost']-1
    return dict(status='evaluated',gap=gap,success_1=gap<=.01,success_5=gap<=.05)


def run_target(target,plan,old,taus,out,smoke=False):
    seed,cid=target;fc,cc=old['fleet'],old['controller'];progress=Progress(f'{seed}/{cid}')
    progress.say('reconstructing fixed vehicle')
    fleet,models=context(seed,old['base'],fc,cc);client=next(c for c in fleet if c.client_id==cid);ac,bc=models[cid]
    bank=build_multi_maneuver_episode_bank(client,Ts=fc['Ts'],T_total=cc['T_total'],T0=cc['T0'],
        r_max=cc['r_max'],lane_change_time=fc['lane_change_time'],n_calibration_per_maneuver=40,
        n_test_per_maneuver=0,base_seed=1900000,tfilter=cc.get('tfilter'))
    previous_banks=list(make_banks(client,old['config'],fc,cc).values())
    for priorseed in [1500000,1600000,1700000,1800000]:
        previous_banks.append(build_multi_maneuver_episode_bank(client,Ts=fc['Ts'],T_total=cc['T_total'],T0=cc['T0'],
            r_max=cc['r_max'],lane_change_time=fc['lane_change_time'],n_calibration_per_maneuver=40,
            n_test_per_maneuver=0,base_seed=priorseed,tfilter=cc.get('tfilter')))
    current={e.seed for e in bank.episodes}
    if any(current & {e.seed for e in b.episodes} for b in previous_banks):raise ValueError('Episode overlap')
    if smoke:bank=replace(bank,episodes=[next(e for e in bank.episodes if e.T_lc==t) for t in fc['lane_change_time']])
    path=out/'seeds'/str(seed)/cid
    def score(x,tau):
        key=hashlib.sha256(json.dumps([list(map(float,x)),float(tau)]).encode()).hexdigest()
        directory=path/'evaluations'/key;rows=[]
        for i,ep in enumerate(bank.episodes):
            f=directory/f'{i:03d}.json'
            if f.exists():r=json.loads(f.read_text());progress.hits+=1
            else:
                one=replace(bank,episodes=[ep]);a=asdict(evaluate(client,ac,bc,tau,x,one,'calibration',cc,fc['Ts'],5))
                b=asdict(evaluate(client,ac,bc,tau,x,one,'calibration',cc,fc['Ts'],10))
                r=dict(index=i,noise_seed=ep.seed,maneuver=ep.T_lc,coarse=a,fine=b,integration=integration_check(a,b,.001))
                write_json(f,r);progress.new+=1
            rows.append(r);progress.say(f'tau={tau:g}, episode {i+1}/{len(bank.episodes)}',False)
        result=dict(xi=x,tau=tau,episodes=len(rows),evaluation_key=key,**summarize(rows))
        write_json(directory/'summary.json',result)
        return result
    for tau in taus:
        final=path/'assessment'/f'{tau:g}.json'
        if final.exists():progress.say(f'tau={tau:g}: complete, skipping');continue
        ref=json.loads((PREVIOUS/'seeds'/str(seed)/cid/'references'/f'{tau:g}.json').read_text())['selected']
        if ref is None:raise ValueError('Missing corrected reference')
        reference=score(ref['xi'],tau);rows=[]
        for i,r in enumerate(plan):
            ev=score(r['xi'],tau);rows.append(dict(**r,score=ev,**classify(ev,reference)))
            progress.say(f'tau={tau:g}: donor configuration {i+1}/{len(plan)}',False)
        shared=score(COMMON,tau)
        write_json(final,dict(target=target,tau=tau,reference=reference,transfers=rows,
            exploratory_common=dict(score=shared,**classify(shared,reference))))
        progress.say(f'tau={tau:g}: completed {len(rows)} donor configurations')


def analyze(out,targets,taus,plans,episode_count):
    rows=[];refs=[];common=[]
    for target in targets:
        for tau in taus:
            path=out/'seeds'/str(target[0])/target[1]
            d=json.loads((path/'assessment'/f'{tau:g}.json').read_text())
            if len(d['transfers'])!=len(plans[tuple(target)]):raise ValueError('Incomplete transfers')
            identity=dict(target_seed=target[0],target_client=target[1],tau=tau)
            allscores=[d['reference'],d['exploratory_common']['score']]+[r['score'] for r in d['transfers']]
            for score in allscores:
                directory=path/'evaluations'/score['evaluation_key']
                if score['episodes']!=episode_count or len(list(directory.glob('[0-9][0-9][0-9].json')))!=episode_count:
                    raise ValueError('Incomplete episode records')
            refs.append({**identity,**d['reference']})
            c=d['exploratory_common'];common.append({**identity,**{k:v for k,v in c.items() if k!='score'},**c['score']})
            for r in d['transfers']:
                rows.append({**identity,**{k:v for k,v in r.items() if k not in ['xi','score']},**r['score']})
    processed=out/'processed';processed.mkdir(exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(processed/'transfers.csv',index=False)
    pd.DataFrame(refs).to_csv(processed/'references.csv',index=False)
    pd.DataFrame(common).to_csv(processed/'exploratory_common.csv',index=False)
    summary=df.groupby(['method','budget','tau']).agg(n=('status','size'),success_5=('success_5','sum'),
        success_1=('success_1','sum'),median_gap=('gap','median'),worst_gap=('gap','max')).reset_index()
    summary.to_csv(processed/'summary.csv',index=False)
    print(summary.to_string(index=False));print('Paired developmental pilot: donor/optimizer repetitions are not independent vehicles.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','smoke','run','analyze','collect'])
    parser.add_argument('--n-jobs',type=int,default=2);a=parser.parse_args()
    prior=json.loads((PREVIOUS/'protocol.json').read_text())
    for rel,h in {**prior['code'],**prior['prior_files']}.items():
        if digest(ROOT/rel)!=h:raise ValueError(f'Previous input/code changed: {rel}')
    oldpath=ROOT/prior['config']['previous_results']/'protocol.json';old=json.loads(oldpath.read_text())
    targets=prior['targets'];histories={};files=[PREVIOUS/'protocol.json',oldpath]
    for target in targets:
        for method in ['sobol','bo_lower_penalty']:
            for seed in [101,202,303]:
                f=PREVIOUS/'seeds'/str(target[0])/target[1]/'audit'/f'{method}_{seed}.json'
                histories[(tuple(target),method,seed)]=json.loads(f.read_text())['records'];files.append(f)
    plans={tuple(t):transfer_plan(t,targets,histories) for t in targets}
    taus=old['actuator']['target_taus'];smoke=a.stage=='smoke'
    if smoke:
        targets=targets[:1];taus=[.2];plans={tuple(targets[0]):plans[tuple(targets[0])][:2]}
    files+=sorted(PREVIOUS.glob('seeds/*/*/references/*.json'))
    code=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)]+[ROOT/'scripts'/f for f in
        ['actuator_corrected.py','actuator_diagnostics.py','actuator_bandwidth_experiment.py','phase2_fbo_comparison.py']]
    protocol=dict(targets=targets,taus=taus,plans=[dict(target=list(t),transfers=plans[tuple(t)]) for t in targets],
        primary='sobol, budget 25; all five donors and three optimizer repeats retained',
        exploratory_common=COMMON,test_seed=1900000,episodes_per_maneuver=40,smoke=smoke,
        code={f.relative_to(ROOT).as_posix():digest(f) for f in code},inputs={f.relative_to(ROOT).as_posix():digest(f) for f in files})
    if a.stage=='preflight':
        print(f'Preflight passed: {len(targets)} targets, {len(taus)} taus, 1800 transfer rows plus 30 exploratory common rows; no target optimization.');return
    out=ROOT/'results'/('actuator_transfer_v6'+('_smoke' if smoke else ''));manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Protocol mismatch; use new namespace')
    write_json(manifest,protocol)
    if a.stage in ['run','smoke']:
        with parallel_config(backend='loky',inner_max_num_threads=1):
            Parallel(n_jobs=a.n_jobs)(delayed(run_target)(t,plans[tuple(t)],old,taus,out,smoke) for t in targets)
    analyze(out,targets,taus,plans,2 if smoke else 80)
    if a.stage=='collect':
        files=[manifest]+sorted(out.glob('processed/*.csv'))+sorted(out.glob('seeds/*/*/assessment/*.json'))+sorted(out.glob('seeds/*/*/evaluations/*/*.json'))
        for i in range(0,len(files),30):subprocess.run(['git','add','-f','--']+[f.relative_to(ROOT.parent).as_posix() for f in files[i:i+30]],cwd=ROOT.parent,check=True)
        print(f'Staged {len(files)} result files. Commit and push.')


if __name__=='__main__':main()
