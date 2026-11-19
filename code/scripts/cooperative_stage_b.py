"""From-scratch cooperative discovery pilot with a sealed, one-candidate audit."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from scipy.stats import qmc, norm, t as student_t
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from joblib import Parallel, delayed, parallel_config
from actuator_bandwidth_experiment import context, write_json
from actuator_corrected import digest, Progress
from actuator_diagnostics import integration_check
from src.actuator_mechanism import gains, episode
from src.episodes import build_multi_maneuver_episode_bank

P=ROOT/'results'
METHODS=['transfer','repair','coordinated','selective']


def trial_plans(cfg,scope):
    seeds=cfg['optimizer_seeds'][:1] if scope=='pilot' else cfg['optimizer_seeds']
    donors=len(cfg['fleet_seeds'])*len(cfg['clients'])
    return [(s,m,d) for s in seeds for m in METHODS for d in (range(donors) if m in ['transfer','repair'] else [-1])]


def bank_seeds(optseed,condition,role,n,blocked):
    key=f'stage_b_v1/{optseed}/{condition}/{role}'
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
    # Search and audit use distinct disjoint uint32 quarters; remove old audit seeds.
    lo,hi=(2**31,3*2**30) if role=='search' else (3*2**30,2**32)
    values=[];seen=set(blocked)
    while len(values)<n:
        z=int(rng.integers(lo,hi))
        if z not in seen:values.append(z);seen.add(z)
    return values


def posterior(X,observations,cfg):
    if not observations:return np.ones(len(X)),np.full(len(X),cfg['gp_prior_std'])
    indices=[r['candidate'] for r in observations]
    y=np.array([r['score'] for r in observations])-1.
    alpha=np.array([max(r['variance'],1e-5) for r in observations])
    kernel=ConstantKernel(cfg['gp_prior_std']**2,constant_value_bounds='fixed')*Matern(
        length_scale=cfg['gp_length_scale'],length_scale_bounds='fixed',nu=2.5)
    gp=GaussianProcessRegressor(kernel=kernel,alpha=alpha,optimizer=None,normalize_y=False)
    gp.fit(X[indices],y)
    mu,sd=gp.predict(X,return_std=True)
    return mu+1.,np.maximum(sd,1e-10)


def local_proposal(X,obs,cfg):
    seen={r['candidate'] for r in obs}
    choices=[i for i in range(len(X)) if i not in seen]
    if not choices:return None
    if len(obs)<2:return choices[0]
    mu,sd=posterior(X,obs,cfg)
    best=min(r['score'] for r in obs)
    imp=best-mu;z=imp/sd;ei=imp*norm.cdf(z)+sd*norm.pdf(z)
    return max(choices,key=lambda k:(ei[k],-k))


class Policy:
    """Only scalar observations and candidate coordinates enter this coordinator."""
    def __init__(self,X,nconditions,cfg,query):
        self.X=X;self.nc=nconditions;self.cfg=cfg;self.query=query
        self.obs=[];self.cache={}
    def get(self,c,k,phase):
        if (c,k) in self.cache:return self.cache[(c,k)]
        if len(self.obs)>=self.cfg['search_query_cap']:return None
        r=self.query(c,k,phase,len(self.obs));self.obs.append(r);self.cache[c,k]=r
        return r
    def local(self,c,n):
        for _ in range(n):
            obs=[r for r in self.obs if r['condition']==c]
            k=local_proposal(self.X,obs,self.cfg)
            if k is None or self.get(c,k,'local_calibration') is None:break
        obs=[r for r in self.obs if r['condition']==c]
        return min(obs,key=lambda r:(r['score'],r['candidate']))['candidate'] if obs else None
    def screen(self,k):
        # Same fixed rotating order for every method/candidate; cached observations charged once.
        order=list(np.roll(np.arange(self.nc),-(k%self.nc)))
        for c in order:
            r=self.get(int(c),k,'fleet_screen')
            if r is None:return 'budget',None
            if not r['valid'] or r['score']>1:return 'reject',int(c)
        return 'pass',None
    def fallback(self):
        # Prefer the candidate with most passing observed conditions; break ties by worst observed score.
        ks={r['candidate'] for r in self.obs}
        if not ks:return None
        def rank(k):
            rs=[r for r in self.obs if r['candidate']==k]
            return (sum(r['valid'] and r['score']<=1 for r in rs),-max(r['score'] for r in rs),-k)
        return max(ks,key=rank)
    def run(self,method,source):
        if method in ['transfer','repair']:
            k=self.local(source,self.cfg['source_query_budget'])
            while k is not None:
                state,c=self.screen(k)
                if state=='pass' or method=='transfer' or state=='budget':return k,state
                old=len(self.obs);k=self.local(c,self.cfg['repair_query_budget'])
                if len(self.obs)==old:return k,'budget'
            return self.fallback(),'budget'
        if method=='coordinated':
            for k in range(len(self.X)):
                state,_=self.screen(k)
                if state=='pass':return k,state
                if state=='budget':break
            return self.fallback(),'budget'
        while len(self.obs)<self.cfg['search_query_cap']:
            mus=[];sds=[]
            for c in range(self.nc):
                mu,sd=posterior(self.X,[r for r in self.obs if r['condition']==c],self.cfg)
                mus.append(mu);sds.append(sd)
            mu=np.array(mus);sd=np.array(sds)
            # Optimistic minimax LCB selects weights. Worst UCB selects which condition to query.
            optimistic=np.max(mu-self.cfg['lcb_multiplier']*sd,axis=0)
            rejected={r['candidate'] for r in self.obs if not r['valid'] or r['score']>1}
            available=[k for k in range(len(self.X)) if k not in rejected and any((c,k) not in self.cache for c in range(self.nc))]
            if not available:break
            k=min(available,key=lambda k:(optimistic[k],k))
            todo=[c for c in range(self.nc) if (c,k) not in self.cache]
            c=max(todo,key=lambda c:(mu[c,k]+self.cfg['lcb_multiplier']*sd[c,k],-c))
            self.get(c,k,'selective_query')
            if all((c,k) in self.cache and self.cache[c,k]['valid'] and self.cache[c,k]['score']<=1 for c in range(self.nc)):
                return k,'pass'
        return self.fallback(),'budget'


def setup(smoke):
    cfgpath=ROOT/'config/runs/cooperative_stage_b.json';cfg=json.loads(cfgpath.read_text())
    if smoke:
        cfg.update(fleet_seeds=cfg['fleet_seeds'][:1],clients=cfg['clients'][:1],taus=[.2,.4],
            candidate_count=8,search_query_cap=6,source_query_budget=2,repair_query_budget=1,
            search_episodes_per_maneuver=2,audit_episodes_per_maneuver=3,optimizer_seeds=[1701])
    oldpath=P/'actuator_diagnostics_v2/protocol.json';old=json.loads(oldpath.read_text())
    # Read only the previous noise-seed allocation, never its scores or frozen candidates.
    previous=P/'cooperative_validation_v1/protocol.json';prior=json.loads(previous.read_text())
    blocked={x for bank in prior['noise_seeds'].values() for x in bank}
    conditions=[(s,c,tau) for s in cfg['fleet_seeds'] for c in cfg['clients'] for tau in cfg['taus']]
    inputs=[cfgpath,oldpath,previous,ROOT/'config/runs/cooperative_acceptance.json']
    models={}
    for s in cfg['fleet_seeds']:
        f=P/f'actuator_bandwidth_v1_pilot/seeds/{s}/sources.json';inputs.append(f)
        source=json.loads(f.read_text())['models']
        for c in cfg['clients']:models[s,c]=(np.array(source[c]['A']),np.array(source[c]['B']))
    # Freeze all simulator code/config dependencies; no historical calibration outputs enter a policy.
    code=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'config').rglob('*.yaml'))
    code += [Path(__file__)]+[ROOT/'scripts'/f for f in ['actuator_bandwidth_experiment.py','actuator_corrected.py','actuator_diagnostics.py','phase2_fbo_comparison.py']]
    manifest=dict(cfg=cfg,smoke=smoke,conditions=conditions,files={f.relative_to(ROOT).as_posix():digest(f) for f in sorted(set(inputs+code))})
    return cfg,old,conditions,models,blocked,manifest


class Evaluator:
    def __init__(self,cfg,old,conditions,models,blocked,optseed,X,out,progress):
        self.cfg,self.old,self.conditions,self.models,self.blocked,self.optseed,self.X,self.out,self.progress=cfg,old,conditions,models,blocked,optseed,X,out,progress
        self.clients={};self.banks={}
        self.cc=old['controller'];self.fc=old['fleet'];self.threshold=np.deg2rad(cfg['threshold_deg_per_s'])
    def prepare(self,c,role):
        seed,cid,tau=self.conditions[c]
        if (seed,cid) not in self.clients:
            fleet,rebuilt=context(seed,self.old['base'],self.fc,self.cc)
            for v in fleet:
                if (seed,v.client_id) in self.models:
                    a,b=self.models[seed,v.client_id]
                    for saved,current in zip((a,b),rebuilt[v.client_id]):np.testing.assert_allclose(saved,current,rtol=1e-10,atol=1e-10)
                    self.clients[seed,v.client_id]=v
        client=self.clients[seed,cid]
        if (c,role) not in self.banks:
            n=self.cfg[f'{role}_episodes_per_maneuver']
            bank=build_multi_maneuver_episode_bank(client,Ts=self.fc['Ts'],T_total=self.cc['T_total'],T0=self.cc['T0'],r_max=self.cc['r_max'],lane_change_time=[1.,2.],n_calibration_per_maneuver=n,n_test_per_maneuver=0,base_seed=3900000,tfilter=self.cc.get('tfilter'))
            seeds=bank_seeds(self.optseed,c,role,2*n,self.blocked)
            for ep,s in zip(bank.episodes,seeds):ep.seed=s;ep.role=role
            self.banks[c,role]=bank
        a,b=self.models[seed,cid]
        return client,a,b,tau,self.banks[c,role]
    def evaluate(self,c,k,role):
        client,a,b,tau,bank=self.prepare(c,role)
        lo=np.array(self.cc['xi_lower']);hi=np.array(self.cc['xi_upper']);xi=lo+self.X[k]*(hi-lo)
        try:g=gains(a,b,tau,xi,self.cc,self.fc['Ts'])
        except (ValueError,np.linalg.LinAlgError) as error:
            return dict(condition=c,candidate=k,valid=False,score=10.,variance=.01,rollouts=0,numerical_rollouts=0,reason=type(error).__name__,cells=[])
        cells=[];rollouts=0;allvalid=True
        for maneuver in [1.,2.]:
            eps=[ep for ep in bank.episodes if ep.T_lc==maneuver]
            rows=[]
            for start in range(0,len(eps),25):
                f=self.out/f'evaluations/{role}/{c:02d}/{k:03d}/{maneuver:g}/{start:04d}.json'
                if f.exists():
                    block=json.loads(f.read_text())
                    if [r['noise_seed'] for r in block]!=[ep.seed for ep in eps[start:start+25]]:raise ValueError('Checkpoint seed mismatch')
                    self.progress.hits+=len(block)
                else:
                    block=[]
                    for ep in eps[start:start+25]:
                        coarse=episode(client,a,b,tau,g,bank,ep,self.cc,self.fc['Ts'],5)
                        fine=episode(client,a,b,tau,g,bank,ep,self.cc,self.fc['Ts'],10)
                        block.append(dict(noise_seed=ep.seed,coarse=coarse,fine=fine,integration=integration_check(coarse,fine,.001)))
                        self.progress.new+=1;self.progress.say(f'{role} condition={c}, candidate={k}, maneuver={maneuver:g}',False)
                    write_json(f,block)
                rows.extend(block)
            rollouts+=len(rows)
            valid=all(r['coarse']['feasible'] and r['fine']['feasible'] and r['integration']['status']=='pass' and np.isfinite(r['coarse']['cost']) and np.isfinite(r['fine']['cost']) for r in rows)
            allvalid &= valid
            x=np.array([r['coarse']['cost'] for r in rows]);finite=np.isfinite(x).all()
            mean=float(x.mean()) if finite else None;se=float(x.std(ddof=1)/np.sqrt(len(x))) if finite else None
            margin=max(abs(r['coarse']['cost']-r['fine']['cost']) for r in rows) if finite else None
            cell=dict(maneuver=maneuver,n=len(rows),valid=valid,mean=mean,se=se,numerical_margin=margin)
            if role=='audit':
                half=student_t.ppf(1-self.cfg['alpha_per_trial']/(2*len(self.conditions)*2),len(x)-1)*se if finite else np.inf
                lower=mean-half-margin if finite else None;upper=mean+half+margin if finite else None
                state='invalid' if not valid else 'supported' if upper<=self.threshold else 'exceeds' if lower>self.threshold else 'unresolved'
                cell.update(lower=lower,upper=upper,status=state)
            cells.append(cell)
        score=max(z['mean'] for z in cells)/self.threshold if allvalid else 10.
        variance=max(z['se']**2 for z in cells)/self.threshold**2 if allvalid else .01
        return dict(condition=c,candidate=k,valid=bool(allvalid),score=float(score),variance=float(variance),rollouts=rollouts,numerical_rollouts=2*rollouts,cells=cells)


def run_trial(plan,cfg,old,conditions,models,blocked,root,smoke):
    seed,method,donor=plan;out=root/f'trials/{seed}/{method}/donor_{donor}'
    progress=Progress(f'{seed}/{method}/{donor}');progress.say('starting/resuming')
    X=qmc.Sobol(3,scramble=True,seed=seed+92000).random_base2(math.ceil(math.log2(cfg['candidate_count'])))[:cfg['candidate_count']]
    evaluator=Evaluator(cfg,old,conditions,models,blocked,seed,X,out,progress)
    def query(c,k,phase,index):
        f=out/f'queries/{index:03d}.json'
        if f.exists():
            r=json.loads(f.read_text())
            if (r['condition'],r['candidate'],r['phase'])!=(c,k,phase):raise ValueError('Policy replay mismatch')
            return r
        r=evaluator.evaluate(c,k,'search');r.update(phase=phase,index=index)
        write_json(f,r);progress.say(f'query {index+1}/{cfg["search_query_cap"]}: {phase}, score={r["score"]:.3f}')
        return r
    policy=Policy(X,len(conditions),cfg,query)
    source=donor*len(cfg['taus'])+cfg['taus'].index(cfg['source_tau']) if donor>=0 else None
    candidate,state=policy.run(method,source)
    if candidate is None:raise ValueError('No candidate proposed')
    selected=dict(candidate=candidate,xi=(np.array(old['controller']['xi_lower'])+X[candidate]*(np.array(old['controller']['xi_upper'])-np.array(old['controller']['xi_lower']))).tolist(),screen_status=state,queries=len(policy.obs))
    choice=out/'selected.json'
    if choice.exists() and json.loads(choice.read_text())!=selected:raise ValueError('Frozen final candidate changed')
    write_json(choice,selected)
    # Policy is finished. Audit observations never enter its state or select a replacement.
    audit=[]
    for c in range(len(conditions)):
        f=out/f'audit/{c:02d}.json'
        if f.exists():r=json.loads(f.read_text())
        else:r=evaluator.evaluate(c,candidate,'audit');write_json(f,r)
        if (r['condition'],r['candidate'])!=(c,candidate):raise ValueError('Audit identity mismatch')
        audit.append(r);progress.say(f'sealed audit condition {c+1}/{len(conditions)}')
        # A fixed-size completed cell may rule out this frozen output. No further optimization.
        if any(z['status'] in ['invalid','exceeds'] for z in r['cells']):break
    statuses=[z['status'] for r in audit for z in r['cells']]
    status='invalid' if 'invalid' in statuses else 'exceeds' if 'exceeds' in statuses else 'supported' if len(audit)==len(conditions) and all(z=='supported' for z in statuses) else 'unresolved'
    result=dict(seed=seed,method=method,donor=donor,**selected,status='smoke_only' if smoke else status,
        search_queries=len(policy.obs),search_rollouts=sum(r['rollouts'] for r in policy.obs),audit_rollouts=sum(r['rollouts'] for r in audit),
        total_rollouts=sum(r['rollouts'] for r in policy.obs+audit),numerical_rollouts=sum(r['numerical_rollouts'] for r in policy.obs+audit),audit_conditions=len(audit))
    write_json(out/'result.json',result)
    return result


def verify_trial(path,cfg,conditions,blocked,seed):
    """Reject incomplete results and verify the charged episode ledger before collection."""
    result=json.loads((path/'result.json').read_text())
    queries=sorted((path/'queries').glob('*.json'))
    if len(queries)!=result['search_queries']:raise ValueError('Query ledger incomplete')
    observations=[json.loads(f.read_text()) for f in queries]
    if [r['index'] for r in observations]!=list(range(len(observations))):raise ValueError('Query indices invalid')
    if len({(r['condition'],r['candidate']) for r in observations})!=len(observations):raise ValueError('Duplicate charged query')
    audit=[json.loads((path/f'audit/{c:02d}.json').read_text()) for c in range(result['audit_conditions'])]
    selected=json.loads((path/'selected.json').read_text())
    if result['candidate']!=selected['candidate']:raise ValueError('Final candidate changed')
    for role,records in [('search',observations),('audit',audit)]:
        for r in records:
            if role=='audit' and r['candidate']!=selected['candidate']:raise ValueError('Audit candidate changed')
            if r['rollouts']==0:
                if r['valid']:raise ValueError('Empty valid evaluation')
                continue
            n=cfg[f'{role}_episodes_per_maneuver']
            expected=bank_seeds(seed,r['condition'],role,2*n,blocked)
            total=0
            for mi,maneuver in enumerate([1.,2.]):
                rows=[]
                for start in range(0,n,25):
                    f=path/f"evaluations/{role}/{r['condition']:02d}/{r['candidate']:03d}/{maneuver:g}/{start:04d}.json"
                    rows.extend(json.loads(f.read_text()))
                if [z['noise_seed'] for z in rows]!=expected[mi*n:(mi+1)*n]:raise ValueError('Episode ledger mismatch')
                total+=len(rows)
            if total!=r['rollouts'] or 2*total!=r['numerical_rollouts']:raise ValueError('Rollout accounting mismatch')
    for role,records in [('search',observations),('audit',audit)]:
        if sum(r['rollouts'] for r in records)!=result[f'{role}_rollouts']:raise ValueError('Trial accounting mismatch')
    if result['total_rollouts']!=result['search_rollouts']+result['audit_rollouts']:raise ValueError('Total accounting mismatch')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','smoke','run','analyze','collect']);p.add_argument('--scope',choices=['pilot','full'],default='pilot');p.add_argument('--n-jobs',type=int,default=2);args=p.parse_args()
    if args.n_jobs<1:p.error('Positive worker count required')
    smoke=args.stage=='smoke';cfg,old,conditions,models,blocked,manifest=setup(smoke)
    plans=trial_plans(cfg,args.scope)
    root=P/(cfg['tag']+('_smoke' if smoke else ''));protocol=root/'protocol.json'
    print(f'{len(plans)} trials, {len(conditions)} vehicle/bandwidth conditions; cap {cfg["search_query_cap"]} queries/trial.',flush=True)
    if args.stage=='preflight':print('Preflight passed; no historical weights/scores loaded into policies.',flush=True);return
    if protocol.exists() and json.loads(protocol.read_text())!=json.loads(json.dumps(manifest)):raise ValueError('Protocol changed; refusing mixed resume')
    if args.stage in ['analyze','collect'] and not protocol.exists():raise ValueError('Run missing')
    write_json(protocol,manifest)
    if args.stage in ['run','smoke']:
        with parallel_config(backend='loky',inner_max_num_threads=1):
            results=Parallel(n_jobs=args.n_jobs)(delayed(run_trial)(plan,cfg,old,conditions,models,blocked,root,smoke) for plan in plans)
    else:
        results=[]
        for seed,method,donor in plans:
            f=root/f'trials/{seed}/{method}/donor_{donor}/result.json'
            if not f.exists():raise ValueError(f'Incomplete: {f}')
            results.append(verify_trial(f.parent,cfg,conditions,blocked,seed))
    processed=root/'processed';processed.mkdir(exist_ok=True)
    df=pd.DataFrame(results);df.to_csv(processed/f'{args.scope}_trials.csv',index=False)
    print(df[['seed','method','donor','status','search_queries','total_rollouts']].to_string(index=False),flush=True)
    print('No inference from successful runs alone; retain failed/unresolved trials. Audit cost included.',flush=True)
    if args.stage=='collect':
        files=[protocol,processed/f'{args.scope}_trials.csv']
        for seed,method,donor in plans:
            trial=root/f'trials/{seed}/{method}/donor_{donor}'
            files+=sorted(trial.rglob('*.json'))
        for i in range(0,len(files),40):subprocess.run(['git','add','-f','--']+[f.relative_to(ROOT.parent).as_posix() for f in files[i:i+40]],cwd=ROOT.parent,check=True)
        print(f'Staged {len(files)} files for {args.scope}. Commit and push.',flush=True)

if __name__=='__main__':main()
