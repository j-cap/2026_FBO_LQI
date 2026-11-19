"""Paired gain-adaptation and measurement-score diagnostics; no optimization."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from joblib import Parallel,delayed,parallel_config
from actuator_bandwidth_experiment import context,write_json,valid
from actuator_corrected import digest,Progress
from actuator_diagnostics import integration_check
from actuator_direct_transfer import COMMON
from src.actuator_mechanism import gains,episode
from src.episodes import build_multi_maneuver_episode_bank
P=ROOT/'results';PRIOR=P/'actuator_transfer_v6'
ARMS=['target_redesign','target_frozen_tau','donor_frozen','reference']


def controller_arms(target,models,tau,reference_xi,cc,ts):
    seed,cid=target;other=987946 if seed==355238 else 355238
    a,b=models[(seed,cid)];da,db=models[(other,cid)]
    return dict(target_redesign=gains(a,b,tau,COMMON,cc,ts),
        target_frozen_tau=gains(a,b,.2,COMMON,cc,ts),donor_frozen=gains(da,db,.2,COMMON,cc,ts),
        reference=gains(a,b,tau,reference_xi,cc,ts))


def run_target(target,old,models,taus,out,smoke):
    seed,cid=target;fc,cc=old['fleet'],old['controller'];progress=Progress(f'{seed}/{cid}');progress.say('reconstructing fixed target')
    fleet,rebuilt=context(seed,old['base'],fc,cc);client=next(c for c in fleet if c.client_id==cid)
    a,b=models[(seed,cid)]
    for saved,current in zip((a,b),rebuilt[cid]):np.testing.assert_allclose(saved,current,rtol=1e-10,atol=1e-10)
    bank=build_multi_maneuver_episode_bank(client,Ts=fc['Ts'],T_total=cc['T_total'],T0=cc['T0'],r_max=cc['r_max'],
        lane_change_time=fc['lane_change_time'],n_calibration_per_maneuver=40,n_test_per_maneuver=0,
        base_seed=1900000,tfilter=cc.get('tfilter'))
    if smoke:bank.episodes[:]=[next(e for e in bank.episodes if e.T_lc==t) for t in fc['lane_change_time']]
    trace_seeds={next(e.seed for e in bank.episodes if e.T_lc==t) for t in fc['lane_change_time']}
    for tau in taus:
        ref=json.loads((P/f'actuator_corrected_v5/seeds/{seed}/{cid}/references/{tau:g}.json').read_text())['selected']
        arms=controller_arms(target,models,tau,ref['xi'],cc,fc['Ts'])
        write_json(out/f'seeds/{seed}/{cid}/{tau:g}/gains.json',dict(arms=arms,common_xi=COMMON,reference_xi=ref['xi'],donor=[987946 if seed==355238 else 355238,cid]))
        key=hashlib.sha256(json.dumps([list(map(float,COMMON)),float(tau)]).encode()).hexdigest()
        for arm,g in arms.items():
            for i,ep in enumerate(bank.episodes):
                dest=out/f'seeds/{seed}/{cid}/{tau:g}/{arm}/{i:03d}.json'
                if dest.exists():progress.hits+=1;continue
                coarse=episode(client,a,b,tau,g,bank,ep,cc,fc['Ts'],5,ep.seed in trace_seeds)
                fine=episode(client,a,b,tau,g,bank,ep,cc,fc['Ts'],10)
                if arm=='target_redesign':
                    # Find the original v6 episode by seed, not smoke index.
                    prior_dir=PRIOR/f'seeds/{seed}/{cid}/evaluations/{key}'
                    original_index=next(j for j in range(80) if json.loads((prior_dir/f'{j:03d}.json').read_text())['noise_seed']==ep.seed)
                    original=json.loads((prior_dir/f'{original_index:03d}.json').read_text())
                    for resolution,new in [('coarse',coarse),('fine',fine)]:
                        for field in ['cost','max_beta','max_delta','max_rate','saturation_fraction','spectral_radius']:
                            np.testing.assert_allclose(new[field],original[resolution][field],rtol=1e-9,atol=1e-11)
                        assert new['feasible']==original[resolution]['feasible']
                true_a=dict(coarse,cost=coarse['true_cost']);true_b=dict(fine,cost=fine['true_cost'])
                if coarse['true_cost'] is None:true_a['cost']=float('nan')
                if fine['true_cost'] is None:true_b['cost']=float('nan')
                write_json(dest,dict(seed=seed,client=cid,tau=tau,arm=arm,noise_seed=ep.seed,maneuver=ep.T_lc,
                    coarse=coarse,fine=fine,integration=integration_check(coarse,fine,.001),
                    true_integration=integration_check(true_a,true_b,.001)))
                progress.new+=1;progress.say(f'{arm}, tau={tau:g}, episode {i+1}/{len(bank.episodes)}',False)
            progress.say(f'{arm}, tau={tau:g}: complete')


def analyze(out,targets,taus,n):
    rows=[];summaries=[]
    for seed,cid in targets:
        for tau in taus:
            for arm in ARMS:
                fs=sorted((out/f'seeds/{seed}/{cid}/{tau:g}/{arm}').glob('*.json'))
                if len(fs)!=n:raise ValueError(f'Incomplete {seed}/{cid}/{tau}/{arm}: {len(fs)}/{n}')
                group=[]
                for f in fs:
                    d=json.loads(f.read_text());c=d['coarse'];identity={k:d[k] for k in ['seed','client','tau','arm','noise_seed','maneuver']}
                    row=dict(**identity,**{k:v for k,v in c.items() if k!='trace'},integration=d['integration']['status'],true_integration=d['true_integration']['status'],fine_feasible=d['fine']['feasible'])
                    rows.append(row);group.append(row)
                df=pd.DataFrame(group);ok=bool((df.feasible & df.fine_feasible).all())
                summaries.append(dict(seed=seed,client=cid,tau=tau,arm=arm,feasible=ok,failed_episodes=int((~(df.feasible&df.fine_feasible)).sum()),
                    integration_pass=bool((df.integration=='pass').all()),true_integration_pass=bool((df.true_integration=='pass').all()),
                    **{k:float(df[k].mean()) if ok else None for k in ['cost','true_cost','measured_mse','true_mse','noise_mse','cross_mse']},
                    linear_radius=float(df.spectral_radius.iloc[0])))
    processed=out/'processed';processed.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(processed/'episodes.csv',index=False)
    df=pd.DataFrame(summaries);df.to_csv(processed/'controllers.csv',index=False)
    print(df.groupby('arm')[['feasible','integration_pass','true_integration_pass']].sum().to_string())
    print('True and measured errors come from identical noisy rollouts. Failed controllers remain explicit.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','smoke','run','analyze','collect']);p.add_argument('--n-jobs',type=int,default=2);args=p.parse_args()
    prev=json.loads((PRIOR/'protocol.json').read_text())
    for rel,h in {**prev['code'],**prev['inputs']}.items():
        if digest(ROOT/rel)!=h:raise ValueError(f'Previous code/input changed: {rel}')
    old=json.loads((P/'actuator_diagnostics_v2/protocol.json').read_text());models={};inputs=[PRIOR/'protocol.json',P/'actuator_diagnostics_v2/protocol.json']
    for seed in [355238,987946]:
        f=P/f'actuator_bandwidth_v1_pilot/seeds/{seed}/sources.json';inputs.append(f)
        for cid,m in json.loads(f.read_text())['models'].items():models[(seed,cid)]=(np.array(m['A']),np.array(m['B']))
    targets=prev['targets'];taus=prev['taus'];smoke=args.stage=='smoke'
    if smoke:targets=targets[:1];taus=[.2,.4]
    inputs+=sorted((P/'actuator_corrected_v5').glob('seeds/*/*/references/*.json'))
    # Hash only common-weight v6 records used for exact replay verification.
    for seed,cid in targets:
        for tau in taus:
            key=hashlib.sha256(json.dumps([list(map(float,COMMON)),float(tau)]).encode()).hexdigest()
            inputs+=sorted((PRIOR/f'seeds/{seed}/{cid}/evaluations/{key}').glob('*.json'))
    code=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)]+[ROOT/'scripts'/f for f in ['actuator_direct_transfer.py','actuator_corrected.py','actuator_diagnostics.py','actuator_bandwidth_experiment.py','phase2_fbo_comparison.py']]
    protocol=dict(targets=targets,taus=taus,arms=ARMS,common_xi=COMMON,test_seed=1900000,smoke=smoke,
        source_tau=.2,tracking_time=.2,donor_rule='same family in opposite fleet realization',
        code={f.relative_to(ROOT).as_posix():digest(f) for f in code},inputs={f.relative_to(ROOT).as_posix():digest(f) for f in inputs})
    if args.stage=='preflight':print('Preflight passed: 6 targets x 5 taus x 4 arms; fixed gains/weights, no optimization.');return
    out=P/('actuator_mechanism_v7'+('_smoke' if smoke else ''));manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Protocol mismatch')
    write_json(manifest,protocol)
    if args.stage in ['run','smoke']:
        with parallel_config(backend='loky',inner_max_num_threads=1):
            Parallel(n_jobs=args.n_jobs)(delayed(run_target)(t,old,models,taus,out,smoke) for t in targets)
    analyze(out,targets,taus,2 if smoke else 80)
    if args.stage=='collect':
        files=[manifest]+sorted(out.glob('processed/*.csv'))+sorted(out.glob('seeds/*/*/*/gains.json'))+sorted(out.glob('seeds/*/*/*/*/*.json'))
        for i in range(0,len(files),30):subprocess.run(['git','add','-f','--']+[f.relative_to(ROOT.parent).as_posix() for f in files[i:i+30]],cwd=ROOT.parent,check=True)
        print(f'Staged {len(files)} files. Commit and push.')


if __name__=='__main__':main()
