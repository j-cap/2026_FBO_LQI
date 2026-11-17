"""Step 2: fixed-vehicle source-optimizer and reference-validation diagnostics."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc
from joblib import Parallel, delayed, parallel_config
from actuator_bandwidth_experiment import context, bo, valid, write_json
from src.actuator_bandwidth import evaluate
from src.episodes import build_multi_maneuver_episode_bank
from src.io_utils import load_yaml, load_run_config, deep_update
from phase2_fbo_comparison import _resolve


def make_banks(client, cfg, fc, cc):
    """Use calibration role internally; keep external bank names and seeds distinct."""
    banks = {}
    used = set()
    for role, count in cfg['episodes_per_maneuver'].items():
        bank = build_multi_maneuver_episode_bank(
            client, Ts=fc['Ts'], T_total=cc['T_total'], T0=cc['T0'],
            r_max=cc['r_max'], lane_change_time=fc['lane_change_time'],
            n_calibration_per_maneuver=count, n_test_per_maneuver=0,
            base_seed=cfg['episode_seeds'][role], tfilter=cc.get('tfilter'))
        seeds = {ep.seed for ep in bank.episodes}
        if seeds & used:
            raise ValueError('Episode seed collision between banks')
        used |= seeds
        banks[role] = bank
    return banks


def search_reference(score, lower, upper, seed, cfg):
    """Calibrate first, validate a fixed shortlist; NEVER access final tests."""
    seen = {}
    def objective(x):
        key = tuple(float(v) for v in x)
        if key not in seen:
            seen[key] = dict(xi=list(key), calibration=score(x, 'calibration'))
        ev = seen[key]['calibration']
        return ev['cost'] if valid(ev) else 10.
    sizes = cfg['reference_grid_sizes']
    grid = lower + qmc.Sobol(3, scramble=True, seed=seed).random(sizes[-1]) * (upper-lower)
    convergence = []
    for n, x in enumerate(grid, 1):
        objective(x)
        if n in sizes:
            good = [r['calibration']['cost'] for r in seen.values() if valid(r['calibration'])]
            convergence.append(dict(n=n, best=min(good) if good else None))
    ranked = sorted((r for r in seen.values() if valid(r['calibration'])), key=lambda r:r['calibration']['cost'])
    fits = []
    for r in ranked[:cfg['reference_starts']]:
        fit = minimize(objective, r['xi'], method='Powell', bounds=list(zip(lower, upper)),
                       options=dict(maxfev=cfg['reference_maxfev'], xtol=1e-3, ftol=1e-4))
        # All intermediate evaluations are retained, including when maxfev is reached.
        fits.append(dict(success=bool(fit.success), nfev=int(fit.nfev), message=str(fit.message)))
    ranked = sorted((r for r in seen.values() if valid(r['calibration'])), key=lambda r:r['calibration']['cost'])
    # Fixed mix: best refined candidates AND best grid points. Avoid a shortlist
    # consisting entirely of virtually identical points from a single Powell run.
    grid_ranked = sorted((seen[tuple(float(v) for v in x)] for x in grid
                          if valid(seen[tuple(float(v) for v in x)]['calibration'])),
                         key=lambda r:r['calibration']['cost'])
    k = cfg['validation_shortlist']//2
    shortlist = {tuple(r['xi']):dict(r) for r in ranked[:k] + grid_ranked[:k]}
    for r in shortlist.values():
        r['validation'] = score(r['xi'], 'validation')
    eligible = [r for r in shortlist.values() if valid(r['validation'])]
    chosen = min(eligible, key=lambda r:r['validation']['cost']) if eligible else None
    first, last = convergence[0]['best'], convergence[-1]['best']
    change = (first-last)/last if first is not None and last is not None else None
    return dict(selected=chosen, shortlist=list(shortlist.values()), convergence=convergence,
                grid_relative_improvement=change, grid_check_pass=bool(change is not None and change<=cfg['reference_tolerance']),
                best_calibration=ranked[0] if ranked else None, optimizer_diagnostics=fits,
                n_calibration_evaluations=len(seen),
                status='validation_selected' if chosen else 'no_validated_reference')


def integration_check(coarse, fine, tolerance):
    same = coarse['feasible'] == fine['feasible'] and coarse['reason'] == fine['reason']
    error = abs(coarse['cost']-fine['cost'])/fine['cost'] if valid(coarse) and valid(fine) else None
    return dict(feasibility_agrees=bool(same), relative_cost_error=error,
                cost_check_pass=bool(error is not None and error<=tolerance),
                status=('pass' if same and error is not None and error<=tolerance else
                        'failure_reproduced' if same and not coarse['feasible'] else 'unresolved'))


def classify(score, reference, reference_ok, threshold=.05):
    if not valid(score):
        return 'infeasible', None
    if not reference_ok:
        return 'unresolved_reference', None
    regret = score['cost']/reference['cost']-1
    return ('success' if regret<=threshold else 'inaccurate'), regret


def run_seed(seed, stage, cfg, acfg, base, fc, cc, out):
    sd = out/'seeds'/str(seed)
    sd.mkdir(parents=True, exist_ok=True)
    fleet, models = context(seed, base, fc, cc)
    clients = [c for c in fleet if c.client_id.rsplit('_', 1)[1] in cfg['target_slots']]
    if stage=='preflight':
        for c in clients: make_banks(c,cfg,fc,cc)
        return dict(seed=seed, vehicles=len(clients))
    lower, upper = np.array(cc['xi_lower']), np.array(cc['xi_upper'])
    for client in clients:
        cid = client.client_id
        banks = make_banks(client,cfg,fc,cc)
        ad, bd = models[cid]
        def score(x, role, tau=acfg['source_tau'], steps=None):
            if role=='test' and stage!='test': raise RuntimeError('Final-test access outside test stage')
            steps = steps or acfg['substeps']
            key = json.dumps([cid,list(map(float,x)),role,float(tau),steps])
            path = sd/'cache'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
            if path.exists(): return json.loads(path.read_text())
            ev = asdict(evaluate(client,ad,bd,tau,x,banks[role],'calibration',cc,fc['Ts'],steps))
            write_json(path,ev)
            return ev
        if stage=='sources':
            for role in cfg['source_banks']:
                for optseed in cfg['optimizer_seeds']:
                    path = sd/'sources'/f'{cid}_{role}_{optseed}.json'
                    if path.exists(): continue
                    records, recs = bo(lambda x:score(x,role),max(cfg['budgets']),lower,upper,optseed,
                                      acfg['n_candidates'],cc['r_max'])
                    snapshots=[]
                    for b in cfg['budgets']:
                        x=recs[b-1]
                        snapshots.append(dict(budget=b,xi=x,
                            calibration=score(x,'calibration') if x is not None else None,
                            legacy=score(x,'legacy') if x is not None else None))
                    write_json(path,dict(client_id=cid,optimizer_seed=optseed,bank=role,records=records,snapshots=snapshots))
                    print(f'sources {seed}/{cid}/{role}/{optseed}',flush=True)
        elif stage=='references':
            for tau in acfg['target_taus']:
                path=sd/'references'/f'{cid}_{tau:g}.json'
                if path.exists(): continue
                ref=search_reference(lambda x,role:score(x,role,tau),lower,upper,seed+900,cfg)
                write_json(path,dict(client_id=cid,tau=tau,**ref))
                print(f'reference {seed}/{cid}/{tau}: {ref["status"]}',flush=True)
        elif stage=='test':
            path=sd/'tests'/f'{cid}.json'
            if path.exists(): continue
            # Require all calibration artifacts before allowing final assessment.
            refpaths=[sd/'references'/f'{cid}_{t:g}.json' for t in acfg['target_taus']]
            sourcepaths=[sd/'sources'/f'{cid}_{r}_{s}.json' for r in cfg['source_banks'] for s in cfg['optimizer_seeds']]
            if not all(p.exists() for p in refpaths+sourcepaths): raise FileNotFoundError('Run sources and references before test')
            refs=[]; sources=[]
            for p in refpaths:
                ref=json.loads(p.read_text()); selected=ref['selected']; ev=None; integration=None
                if selected:
                    ev=score(selected['xi'],'test',ref['tau'])
                    fine=score(selected['xi'],'test',ref['tau'],2*acfg['substeps'])
                    integration=integration_check(ev,fine,cfg['integration_tolerance'])
                refs.append(dict(client_id=cid,tau=ref['tau'],xi=selected['xi'] if selected else None,
                    test=ev,integration=integration,grid_check_pass=ref['grid_check_pass'],
                    reference_usable=bool(ev and valid(ev) and ref['grid_check_pass'] and integration['status']=='pass')))
            matched=next(r for r in refs if r['tau']==acfg['source_tau'])
            for p in sourcepaths:
                src=json.loads(p.read_text())
                for snapshot in src['snapshots']:
                    x=snapshot['xi']
                    ev=score(x,'test') if x is not None else dict(cost=None,feasible=False,reason='no_feasible_recommendation')
                    fine=score(x,'test',steps=2*acfg['substeps']) if x is not None else ev
                    ic=integration_check(ev,fine,cfg['integration_tolerance'])
                    status,regret=classify(ev,matched['test'],matched['reference_usable'])
                    if valid(ev) and ic['status']!='pass': status,regret='unresolved_integration',None
                    sources.append(dict(client_id=cid,optimizer_seed=src['optimizer_seed'],bank=src['bank'],
                        budget=snapshot['budget'],xi=x,test=ev,integration=ic,status=status,regret=regret))
            write_json(path,dict(references=refs,sources=sources))
            print(f'test {seed}/{cid}: complete',flush=True)


def analyze(out,cfg,acfg):
    refs=[];sources=[];search=[];history=[]
    for seed in cfg['fleet_seeds']:
        sd=out/'seeds'/str(seed)
        for family in ['nominal','payload','tire_degraded']:
            for slot in cfg['target_slots']:
                cid=f'{family}_{slot}'; p=sd/'tests'/f'{cid}.json'
                if not p.exists(): raise FileNotFoundError(f'Incomplete assessment: {p}')
                result=json.loads(p.read_text())
                for ref in result['references']:
                    ev=ref['test'] or {}; ic=ref['integration'] or {}
                    refs.append(dict(seed=seed,client_id=cid,tau=ref['tau'],reference_usable=ref['reference_usable'],
                        test_feasible=ev.get('feasible',False),test_reason=ev.get('reason','no_validated_reference'),
                        test_cost=ev.get('cost'),grid_check_pass=ref['grid_check_pass'],integration_status=ic.get('status','unresolved')))
                for row in result['sources']:
                    sources.append(dict(seed=seed,**{k:v for k,v in row.items() if k not in ['test','integration']},
                        cost=row['test']['cost'],feasible=row['test']['feasible'],reason=row['test']['reason'],
                        integration_status=row['integration']['status']))
                for tau in acfg['target_taus']:
                    ref=json.loads((sd/'references'/f'{cid}_{tau:g}.json').read_text())
                    search.append(dict(seed=seed,client_id=cid,tau=tau,status=ref['status'],
                        grid_relative_improvement=ref['grid_relative_improvement'],
                        validated_candidates=sum(valid(r['validation']) for r in ref['shortlist']),
                        refinement_converged=sum(r['success'] for r in ref['optimizer_diagnostics']),
                        refinement_starts=len(ref['optimizer_diagnostics']),n_calibration_evaluations=ref['n_calibration_evaluations']))
                ref=json.loads((sd/'references'/f'{cid}_{acfg["source_tau"]:g}.json').read_text())
                baseline=ref['best_calibration']['calibration']['cost'] if ref['best_calibration'] else None
                for role in cfg['source_banks']:
                    for optseed in cfg['optimizer_seeds']:
                        src=json.loads((sd/'sources'/f'{cid}_{role}_{optseed}.json').read_text())
                        for snap in src['snapshots']:
                            score=snap['calibration']; rec=src['records'][:snap['budget']]
                            history.append(dict(seed=seed,client_id=cid,bank=role,optimizer_seed=optseed,budget=snap['budget'],
                                expanded_calibration_cost=score['cost'] if score else None,
                                expanded_calibration_feasible=bool(score and valid(score)),
                                calibration_gap=(score['cost']/baseline-1) if score and valid(score) and baseline else None,
                                unique_evaluations=len({tuple(r['xi']) for r in rec}),
                                infeasible_evaluations=sum(not valid(r['score']) for r in rec)))
    processed=out/'processed';processed.mkdir(exist_ok=True)
    for name,rows in [('references',refs),('source_test',sources),('reference_search',search),('source_convergence',history)]:
        pd.DataFrame(rows).to_csv(processed/f'{name}.csv',index=False)
    df=pd.DataFrame(sources)
    counts=df.groupby(['bank','budget','status']).size().rename('count').reset_index()
    counts.to_csv(processed/'source_status_counts.csv',index=False)
    print(counts.to_string(index=False))
    print(pd.DataFrame(refs).groupby('tau').reference_usable.agg(['sum','count']).to_string())
    print('Descriptive pilot only. Inspect per-fleet results; no pooled-source independence assumption.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','sources','references','test','analyze','smoke'])
    p.add_argument('--config',default='config/runs/actuator_diagnostics.yaml')
    p.add_argument('--n-jobs',type=int,default=1)
    p.add_argument('--run-tag')
    a=p.parse_args();cfg=load_yaml(_resolve(a.config));acfg=load_yaml(_resolve(cfg['base_config']))
    base=load_run_config(_resolve(acfg['base_config']));base.pop('_config_path',None)
    fc=deep_update(load_yaml(_resolve(base['fleet_config'])),base.get('fleet_overrides',{}))
    cc=deep_update(load_yaml(_resolve(base['controller_config'])),base.get('controller_overrides',{}))
    if a.run_tag: cfg['run_tag']=a.run_tag
    if a.stage=='smoke':
        cfg.update(fleet_seeds=cfg['fleet_seeds'][:1],optimizer_seeds=[101],budgets=[3],
            reference_grid_sizes=[4,8],reference_starts=1,reference_maxfev=8,validation_shortlist=4,
            episodes_per_maneuver={r:1 for r in cfg['episodes_per_maneuver']})
        acfg.update(target_taus=[acfg['source_tau']],n_candidates=8)
    if len(set(cfg['episode_seeds'].values()))!=4: raise ValueError('Use four distinct episode-bank seeds')
    if cfg['budgets']!=sorted(set(cfg['budgets'])) or min(cfg['budgets'])<2: raise ValueError('Invalid budgets')
    out=ROOT/'results'/(cfg['run_tag']+('_smoke' if a.stage=='smoke' else ''))
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/actuator_bandwidth_experiment.py',ROOT/'scripts/phase2_fbo_comparison.py']
    protocol=dict(config=cfg,actuator=acfg,base=base,fleet=fc,controller=cc,
        code={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in files})
    if a.stage=='preflight':
        print([run_seed(s,'preflight',cfg,acfg,base,fc,cc,out) for s in cfg['fleet_seeds']])
        print(f'Output: {out}; no optimization or test assessment performed');return
    out.mkdir(parents=True,exist_ok=True);manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol: raise ValueError('Protocol changed: choose a new --run-tag')
    write_json(manifest,protocol)
    if a.stage=='analyze':analyze(out,cfg,acfg);return
    stages=['sources','references','test'] if a.stage=='smoke' else [a.stage]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for stage in stages:
            Parallel(n_jobs=a.n_jobs)(delayed(run_seed)(s,stage,cfg,acfg,base,fc,cc,out) for s in cfg['fleet_seeds'])
    if a.stage in ['test','smoke']:analyze(out,cfg,acfg)


if __name__=='__main__':main()
