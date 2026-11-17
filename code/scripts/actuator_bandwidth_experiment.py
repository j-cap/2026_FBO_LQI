"""Staged, resumable actuator-bandwidth experiment; see docs/actuator_bandwidth.md."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import minimize
from scipy.stats import qmc, spearmanr
from src.actuator_bandwidth import continuous_vehicle, evaluate
from src.episodes import build_multi_maneuver_episode_bank
from src.io_utils import load_run_config,load_yaml,deep_update,set_global_seed
from src.optimization.gp_surrogate import propose_next
from src.dynamics_distance import pairwise_uncertainty_aware_distance
from src.ifac_bridge import pack_theta_AB
from phase2_fbo_comparison import build_fleet_and_identify,_resolve


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,default=lambda v: v.item() if isinstance(v,np.generic) else v.tolist()));temp.replace(path)


def valid(score):
    return score['feasible'] and np.isfinite(score['cost']) and score['cost']>0


def bo(score, budget, lower, upper, seed, candidates, cost_scale, peers=(), initial=None):
    """Calibration-only EI; optional transferred first query, counted in budget."""
    starts=qmc.Sobol(3,scramble=True,seed=seed).random(2)
    grid=qmc.Sobol(3,scramble=True,seed=seed+1).random(candidates)
    records=[]; recommendations=[]
    for n in range(budget):
        if n<2:
            xi=np.asarray(initial) if n==0 and initial is not None else lower+starts[n]*(upper-lower)
        else:
            pool=list(peers)+records
            x=np.array([(np.array(r['xi'])-lower)/(upper-lower) for r in pool])
            y=np.array([r['training_y'] for r in pool])
            feasible=[r['training_y'] for r in records if valid(r['score'])]
            index,_=propose_next(x,y,np.ones(len(pool)),grid,min(feasible) if feasible else min(y),seed=seed+n)
            xi=lower+grid[index]*(upper-lower)
        ev=score(xi)
        records.append(dict(xi=xi.tolist(),score=ev,training_y=float(ev['cost']/cost_scale) if valid(ev) else 10.))
        good=[r for r in records if valid(r['score'])]
        recommendations.append(min(good,key=lambda r:r['score']['cost'])['xi'] if good else None)
    return records,recommendations


def context(seed,base,fc,cc):
    set_global_seed(seed)
    kw=dict(u_min=cc['u_min'],u_max=cc['u_max'],u_rate_max=cc['u_rate_max'],
            beta_max=np.deg2rad(cc['beta_max_deg']),noise_std=cc['noise_std'],process_noise=cc['process_noise'],
            plant_mode=cc['plant_mode'],mu=cc['tire_mu'])
    fleet,_=build_fleet_and_identify(deep_update(fc,{'seed':fc['seed']+seed}),cc,base,
            eval_kwargs=kw,state_scale=cc['state_scale'],fixed_R=cc['fixed_R'])
    models={c.client_id:continuous_vehicle(c.Ad_hat,c.Bd_hat,fc['Ts']) for c in fleet}
    return fleet,models


def one_seed(seed,stage,cfg,base,fc,cc,out,slots,selected_taus):
    sd=out/'seeds'/str(seed);sd.mkdir(parents=True,exist_ok=True)
    path=sd/(stage+'.json' if stage!='bo' else 'bo_'+hashlib.sha256(json.dumps(selected_taus).encode()).hexdigest()[:8]+'.json')
    if path.exists():
        print(f'{stage}: seed={seed} checkpoint complete',flush=True);return
    fleet,models=context(seed,base,fc,cc);clients={c.client_id:c for c in fleet}
    lower=np.asarray(cc['xi_lower']);upper=np.asarray(cc['xi_upper']);ts=fc['Ts']
    # Fixed task-scale normalization needs no extra target measurement or feasible baseline.
    scale=float(cc['r_max']);cache={};banks={}
    def get_score(cid,tau,xi,role='calibration',steps=None):
        steps=cfg['substeps'] if steps is None else steps
        key=json.dumps([cid,float(tau),[float(v) for v in xi],role,steps])
        name=hashlib.sha256(key.encode()).hexdigest();file=sd/'cache'/(name+'.json')
        if name in cache:return cache[name]
        if file.exists():
            result=json.loads(file.read_text());cache[name]=result;return result
        if (cid,role) not in banks:
            banks[cid,role]=build_multi_maneuver_episode_bank(clients[cid],Ts=ts,T_total=cc['T_total'],T0=cc['T0'],
                r_max=cc['r_max'],lane_change_time=fc['lane_change_time'],
                n_calibration_per_maneuver=base['n_calibration_episodes'] if role=='calibration' else 0,
                n_test_per_maneuver=20 if role=='test' else 0,
                base_seed=base['episode_base_seed'] if role=='calibration' else 900000,tfilter=cc.get('tfilter'))
        ac,bc=models[cid]
        result=asdict(evaluate(clients[cid],ac,bc,tau,xi,banks[cid,role],role,cc,ts,steps))
        write_json(file,result);cache[name]=result;return result
    source_file=sd/'sources.json'
    if stage=='sources':
        histories={}
        for cid in clients:
            partial=sd/'source_clients'/(cid+'.json')
            if partial.exists():histories[cid]=json.loads(partial.read_text());continue
            records,_=bo(lambda xi:get_score(cid,cfg['source_tau'],xi),cfg['source_budget'],lower,upper,
                         seed,cfg['n_candidates'],scale)
            histories[cid]=records;write_json(partial,records)
            print(f'sources seed={seed}: {cid} complete',flush=True)
        write_json(source_file,dict(histories=histories,models={cid:dict(A=ac.tolist(),B=bc.tolist()) for cid,(ac,bc) in models.items()}))
        return
    if not source_file.exists():raise FileNotFoundError(f'Run sources first: {source_file}')
    histories=json.loads(source_file.read_text())['histories']
    ids=list(clients);D=pairwise_uncertainty_aware_distance([pack_theta_AB(c.Ad_hat,c.Bd_hat) for c in fleet],[c.W_raw for c in fleet])
    targets=[cid for cid in ids if cid.rsplit('_',1)[1] in slots]
    common_grid=lower+qmc.Sobol(3,scramble=True,seed=seed+900).random(max(cfg['landscape_sizes']))*(upper-lower)
    rows=[];landscapes=[];references=[];convergence=[]
    taus=cfg['target_taus'] if stage!='bo' else selected_taus
    if stage=='bo':
        if not (sd/'diagnostic.json').exists():raise FileNotFoundError('Run diagnostic before BO')
        prior=json.loads((sd/'diagnostic.json').read_text())
    for cid in targets:
        source_ids=[s for s in ids if s!=cid]
        choices={s:min([r for r in histories[s] if valid(r['score'])],key=lambda r:r['score']['cost'])['xi']
                 if any(valid(r['score']) for r in histories[s]) else None for s in source_ids}
        nearest=min(source_ids,key=lambda s:(D[ids.index(cid),ids.index(s)],s))
        for tau in taus:
            part=sd/'conditions'/f'{stage}_{cid}_{tau:g}.json'
            if part.exists():
                data=json.loads(part.read_text());rows+=data['rows'];landscapes+=data.get('landscapes',[])
                references+=data.get('references',[]);convergence+=data.get('convergence',[]);continue
            local=[];land=[];refs=[];conv=[]
            header=dict(seed=seed,client_id=cid,family=cid.rsplit('_',1)[0],tau=tau)
            def test_row(method,xi,reference,**extra):
                ev=get_score(cid,tau,xi,'test') if xi is not None else dict(cost=float('nan'),feasible=False,reason='no_feasible_source_or_recommendation')
                ref_ok=valid(reference)
                regret=ev['cost']/reference['cost']-1 if valid(ev) and ref_ok else float('inf')
                return dict(**header,method=method,xi=xi,**ev,reference_valid=ref_ok,
                            reference_cost=reference['cost'],regret=regret,
                            success=bool(ref_ok and valid(ev) and regret<=.05),**extra)
            if stage=='diagnostic':
                scores=[get_score(cid,tau,xi) for xi in common_grid]
                good=[k for k,s in enumerate(scores) if valid(s)]
                for n in cfg['landscape_sizes']:
                    g=[s['cost'] for s in scores[:n] if valid(s)]
                    conv.append(dict(**header,grid_size=n,best_cost=min(g) if g else None,feasible_fraction=len(g)/n))
                if good:
                    ranked=sorted(good,key=lambda k:scores[k]['cost']);best_x=common_grid[ranked[0]].tolist();best=scores[ranked[0]]
                    for k in ranked[:cfg['reference_starts']]:
                        def objective(x):
                            ev=get_score(cid,tau,x)
                            return ev['cost']/scale if valid(ev) else 10.
                        fit=minimize(objective,common_grid[k],method='Powell',bounds=list(zip(lower,upper)),
                                     options={'maxfev':cfg['reference_maxfev'],'xtol':1e-3,'ftol':1e-4})
                        ev=get_score(cid,tau,fit.x)
                        if valid(ev) and ev['cost']<best['cost']:best=ev;best_x=fit.x.tolist()
                    reference=get_score(cid,tau,best_x,'test')
                    # Numerical check against the next-largest nested grid, not a global-optimality certificate.
                    earlier=conv[-2]['best_cost'] if len(conv)>1 else None
                    change=(earlier-best['cost'])/best['cost'] if earlier is not None else float('inf')
                    refs.append(dict(**header,xi=best_x,calibration_cost=best['cost'],test=reference,
                        refinement_change=change,reference_check_pass=bool(valid(reference) and change<=cfg['reference_tolerance'])))
                else:
                    reference=dict(cost=float('nan'),feasible=False,reason='no_feasible_reference')
                    refs.append(dict(**header,xi=None,test=reference,reference_check_pass=False))
                for s,xi in choices.items():local.append(test_row('direct_uniform',xi,reference,source_id=s,n_local_evals=0))
                local.append(test_row('direct_dynamics',choices[nearest],reference,source_id=nearest,n_local_evals=0))
                nominal=[get_score(cid,cfg['source_tau'],xi) for xi in common_grid]
                joint=[k for k in range(len(scores)) if valid(scores[k]) and valid(nominal[k])]
                rho=float(spearmanr([nominal[k]['cost'] for k in joint],[scores[k]['cost'] for k in joint]).statistic) if len(joint)>2 else float('nan')
                def near(v):
                    g=[s['cost'] for s in v if valid(s)]
                    return {k for k,s in enumerate(v) if valid(s) and s['cost']<=1.05*min(g)} if g else set()
                n0,n1=near(nominal),near(scores)
                land.append(dict(**header,spearman_joint=rho,joint_feasible_fraction=len(joint)/len(scores),
                                 near_optimal_jaccard=len(n0&n1)/len(n0|n1) if n0|n1 else 0.,
                                 grid=[dict(xi=x.tolist(),score=s) for x,s in zip(common_grid,scores)]))
                # Integration convergence at the selected reference; never used to choose weights.
                if refs[-1]['xi'] is not None:
                    fine=get_score(cid,tau,refs[-1]['xi'],'test',2*cfg['substeps'])
                    coarse=reference
                    error=abs(fine['cost']-coarse['cost'])/fine['cost'] if valid(fine) and valid(coarse) else None
                    refs[-1]['integration_relative_error']=error
                    refs[-1]['integration_check_pass']=bool(error is not None and error<=1e-3 and fine['feasible']==coarse['feasible'])
            else:
                ref_item=next(r for r in prior['references'] if r['client_id']==cid and r['tau']==tau)
                reference=ref_item['test']
                if not ref_item['reference_check_pass'] or not ref_item.get('integration_check_pass',False):
                    raise ValueError(f'Reference/integration check failed at {seed}/{cid}/{tau}; inspect diagnostic before BO')
                peers=[r for s in source_ids for r in histories[s]]
                for method in ['local','transfer_initialized','fleet_informed']:
                    initial=choices[nearest] if method in ['transfer_initialized','fleet_informed'] else None
                    if method in ['transfer_initialized','fleet_informed'] and initial is None:raise ValueError('Selected source has no feasible weights')
                    records,recs=bo(lambda xi:get_score(cid,tau,xi),cfg['target_budget'],lower,upper,seed,
                        cfg['n_candidates'],scale,peers if method=='fleet_informed' else (),initial)
                    for n,xi in enumerate(recs,1):local.append(test_row(method,xi,reference,source_id='',n_local_evals=n))
                    write_json(sd/'bo_histories'/f'{cid}_{tau:g}_{method}.json',records)
            data=dict(rows=local,landscapes=land,references=refs,convergence=conv);write_json(part,data)
            rows+=local;landscapes+=land;references+=refs;convergence+=conv
            print(f'{stage} seed={seed} target={cid} tau={tau:g}: complete',flush=True)
    write_json(path,dict(rows=rows,landscapes=landscapes,references=references,convergence=convergence))


def analyze(out,seeds,stage,taus=None):
    filename=stage+'.json' if stage!='bo' else 'bo_'+hashlib.sha256(json.dumps(taus).encode()).hexdigest()[:8]+'.json'
    data=[json.loads((out/'seeds'/str(seed)/filename).read_text()) for seed in seeds]
    processed=out/'processed';processed.mkdir(exist_ok=True)
    rows=pd.DataFrame([r for d in data for r in d['rows']]);rows.to_csv(processed/(stage+'_scores.csv'),index=False)
    target=rows.groupby(['seed','client_id','tau','method','n_local_evals']).agg(success=('success','mean'),feasible=('feasible','mean'),reference_valid=('reference_valid','all')).reset_index()
    summary=target.groupby(['tau','method','n_local_evals']).agg(success=('success','mean'),feasible=('feasible','mean'),reference_valid=('reference_valid','all'),n_targets=('client_id','size')).reset_index()
    summary.to_csv(processed/(stage+'_summary.csv'),index=False)
    if stage=='diagnostic':
        pd.DataFrame([{k:v for k,v in r.items() if k!='grid'} for d in data for r in d['landscapes']]).to_csv(processed/'landscape_metrics.csv',index=False)
        pd.DataFrame([r for d in data for r in d['references']]).to_csv(processed/'reference_checks.csv',index=False)
        pd.DataFrame([r for d in data for r in d['convergence']]).to_csv(processed/'grid_convergence.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
    plotted=summary if stage=='diagnostic' else summary[summary.n_local_evals==summary.n_local_evals.max()]
    for method,g in plotted.groupby('method'):
        ax.plot(g.tau,g.success,'o-',label=method)
    ax.set(xlabel='Target actuator time constant [s]',ylabel='Fraction within 5% of numerical reference',ylim=(-.02,1.02))
    ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),fontsize=8)
    fig.savefig(figures/(stage+'_success.png'),dpi=160);fig.savefig(figures/(stage+'_success.pdf'));plt.close(fig)
    print(summary.to_string(index=False))
    if stage=='diagnostic':
        refs=[r for d in data for r in d['references']]
        print('Reference checks passed:',sum(r['reference_check_pass'] for r in refs),'/',len(refs))
        print('Integration checks passed:',sum(r.get('integration_check_pass',False) for r in refs),'/',len(refs))
        print('Inspect reference_checks.csv before interpreting the 5% threshold.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','smoke','sources','diagnostic','bo','analyze'])
    p.add_argument('--config',default='config/runs/actuator_bandwidth.yaml')
    p.add_argument('--run-tag',help='New output namespace for an intentional protocol revision')
    p.add_argument('--scope',choices=['pilot','full'],default='pilot')
    p.add_argument('--n-jobs',type=int,default=1)
    p.add_argument('--taus',nargs='+',type=float)
    p.add_argument('--analyze-stage',choices=['diagnostic','bo'],default='diagnostic')
    a=p.parse_args();cfg=load_yaml(_resolve(a.config));base=load_run_config(_resolve(cfg['base_config']));base.pop('_config_path',None)
    if a.run_tag:cfg['run_tag']=a.run_tag
    if a.taus:a.taus=sorted(set(a.taus))
    fc=deep_update(load_yaml(_resolve(base['fleet_config'])),base.get('fleet_overrides',{}))
    cc=deep_update(load_yaml(_resolve(base['controller_config'])),base.get('controller_overrides',{}))
    if cfg['source_tau'] not in cfg['target_taus'] or min(cfg['target_taus'])<=0:raise ValueError('Need positive taus including matched control')
    if a.taus and cfg['source_tau'] not in a.taus:raise ValueError('Include matched source_tau in BO comparison')
    if a.taus and not set(a.taus)<=set(cfg['target_taus']):raise ValueError('BO taus must be existing diagnostic conditions')
    if (a.stage=='bo' or a.stage=='analyze' and a.analyze_stage=='bo') and not a.taus:raise ValueError('Specify --taus after reviewing diagnostics')
    seeds=base['seeds'] if a.scope=='full' else base['seeds'][:cfg['pilot_seed_count']]
    slots=base['target_slots'] if a.scope=='full' else cfg['pilot_target_slots']
    if a.stage=='smoke':
        seeds=seeds[:1];fc['clients_per_regime']=2;slots=['00'];cfg=dict(cfg,source_budget=3,n_candidates=8,
            landscape_sizes=[4,8],reference_starts=1,reference_maxfev=8,target_taus=[cfg['source_tau']],substeps=3)
        # Smoke checks plumbing, not scientific adequacy; reduced fleet is separately named.
    tag=cfg['run_tag']+'_'+('smoke' if a.stage=='smoke' else a.scope)
    out=ROOT/'results'/tag
    files=list((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/phase2_fbo_comparison.py']
    protocol=dict(config=cfg,base=base,fleet=fc,controller=cc,seeds=seeds,slots=slots,
                  code={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in files})
    if a.stage=='preflight':
        for seed in seeds[:1]:
            fleet,models=context(seed,base,fc,cc)
            print(f'Preflight: {len(fleet)} identified vehicle embeddings valid')
        print(f'{len(seeds)} fleets; {len(slots)*3} targets/fleet; taus={cfg["target_taus"]}; output={out}')
        return
    out.mkdir(parents=True,exist_ok=True);mp=out/'protocol.json'
    if mp.exists() and json.loads(mp.read_text())!=protocol:raise ValueError('Protocol/code changed; choose new run_tag in config')
    write_json(mp,protocol)
    if a.stage=='analyze':analyze(out,seeds,a.analyze_stage,a.taus);return
    stages=['sources','diagnostic'] if a.stage=='smoke' else [a.stage]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for stage in stages:
            Parallel(n_jobs=a.n_jobs)(delayed(one_seed)(s,stage,cfg,base,fc,cc,out,slots,a.taus) for s in seeds)
            if stage!='sources':analyze(out,seeds,stage,a.taus)


if __name__=='__main__':main()
