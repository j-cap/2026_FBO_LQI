"""Matched spectral D/E families, reusing the frozen Gaussian/peer machinery.

Family names are external labels only. Each family delegates planning and
updates to the original D/E implementation with its own public feature map.
The four methods never exchange histories or catalogues with another arm.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from . import tracking_hierarchical as joint
from . import tracking_peer_cost as peer
from .tracking_online import features


LEGACY=[peer.METHODS[0],peer.METHODS[1],joint.SHARED,joint.LOCAL]


def shared(rank):return f'peer_spectral_shared_{rank}_bo'
def local(rank):return f'peer_spectral_local_{rank}_bo'
def spectral_methods(ranks):return [m for rank in ranks for m in (shared(rank),local(rank))]


def basis_descriptor(phi):
    a=np.ascontiguousarray(phi,dtype='<f8')
    return dict(rows=a.shape[0],coefficients=a.shape[1],sha256=hashlib.sha256(a.tobytes()).hexdigest())


def save_bases(path,phis,domain):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,domain_X=domain.X,actions=np.asarray(domain.actions,dtype=str),
            **{f'phi_{rank}':phi for rank,phi in phis.items()})
    temporary.replace(path)


def load_bases(path,ranks,domain,expected):
    with np.load(path,allow_pickle=False) as data:
        if set(data.files) != {'domain_X','actions',*[f'phi_{rank}' for rank in ranks]}:
            raise RuntimeError('Pinned public basis keys differ.')
        if (not np.array_equal(data['actions'],np.asarray(domain.actions,dtype=str)) or
            not np.allclose(data['domain_X'],domain.X,rtol=0,atol=1e-14)):
            raise RuntimeError('Pinned public input grid differs.')
        result={}
        for rank in ranks:
            phi=np.asarray(data[f'phi_{rank}'],dtype=float)
            if (phi.shape != (len(domain.actions),rank+1) or not np.isfinite(phi).all() or
                not np.array_equal(phi[:,0],np.ones(len(phi)))):
                raise RuntimeError('Invalid pinned spectral basis.')
            if basis_descriptor(phi) != expected[str(rank)]:
                raise RuntimeError('Pinned spectral basis hash differs.')
            result[rank]=phi
    return result


def family_cfg(cfg,rank):
    return dict(cfg,methods=[joint.SHARED,joint.LOCAL],model=dict(cfg['model'],features=rank))


def families(state,cfg,phis):
    if joint.SHARED in cfg['methods']:
        yield dict(rank=cfg['model']['features'],names={joint.SHARED:joint.SHARED,joint.LOCAL:joint.LOCAL},
            phi=features(state['domain'],cfg['model']))
    for rank in cfg['spectral_ranks']:
        if shared(rank) in cfg['methods']:
            yield dict(rank=rank,names={joint.SHARED:shared(rank),joint.LOCAL:local(rank)},phi=phis[rank])


def view(data,names):
    return dict(round=data['round'],oracle_cache=data['oracle_cache'],
        methods={old:data['methods'][new] for old,new in names.items()})


def fresh_state(domain,cfg,clients):
    known=LEGACY+spectral_methods(cfg['spectral_ranks'])
    if not cfg['methods'] or len(set(cfg['methods'])) != len(cfg['methods']) or not set(cfg['methods']).issubset(known):
        raise ValueError('Unknown or duplicated spectral method.')
    if (joint.SHARED in cfg['methods']) != (joint.LOCAL in cfg['methods']):
        raise ValueError('The original Bayesian control needs both arms.')
    state=dict(round=0,oracle_cache={},methods={})
    originals=[m for m in cfg['methods'] if m in LEGACY]
    if originals:
        old=joint.fresh_state(domain,dict(cfg,methods=originals),clients)
        state['methods'].update(old['methods'])
    for rank in cfg['spectral_ranks']:
        requested=[m in cfg['methods'] for m in (shared(rank),local(rank))]
        if any(requested) and not all(requested):
            raise ValueError('Each spectral size needs shared and local arms.')
        if all(requested):
            old=joint.fresh_state(domain,family_cfg(cfg,rank),clients)
            state['methods'][shared(rank)]=old['methods'][joint.SHARED]
            state['methods'][local(rank)]=old['methods'][joint.LOCAL]
    return state


def plan_round(data,domain,cfg,clients,seed,phis):
    plans=[]
    baselines=[m for m in cfg['methods'] if m in peer.METHODS[:2]]
    if baselines:
        plans.extend(peer.plan_round(data,domain,dict(cfg,methods=baselines),clients,seed,features(domain,cfg['model'])))
    for family in families(dict(domain=domain),cfg,phis):
        names=family['names']
        proposed=joint.plan_round(view(data,names),domain,family_cfg(cfg,family['rank']),clients,seed,family['phi'])
        plans.extend(dict(p,method=names[p['method']]) for p in proposed)
    return plans


def commit_round(data,plans,observations,domain,cfg,clients,phis):
    next_round=data['round']+1
    baselines=[m for m in cfg['methods'] if m in peer.METHODS[:2]]
    if baselines:
        selected=[(p,o) for p,o in zip(plans,observations) if p['method'] in baselines]
        original=view(data,{m:m for m in baselines})
        peer.commit_round(original,[p for p,_ in selected],[o for _,o in selected],domain,
            dict(cfg,methods=baselines),clients,features(domain,cfg['model']))
    for family in families(dict(domain=domain),cfg,phis):
        names=family['names'];reverse={v:k for k,v in names.items()}
        selected=[(dict(p,method=reverse[p['method']]),o) for p,o in zip(plans,observations) if p['method'] in reverse]
        original=view(data,names)
        joint.commit_round(original,[p for p,_ in selected],[o for _,o in selected],domain,
            family_cfg(cfg,family['rank']),clients,family['phi'])
        for old,new in names.items():
            for event in data['methods'][new]['events'][-len(clients):]:event['method']=new
    data['round']=next_round


def validate_state(data,domain,cfg,clients,seed,phis,complete=False):
    if set(data['methods']) != set(cfg['methods']):
        raise RuntimeError('Spectral method set differs.')
    baselines=[m for m in cfg['methods'] if m in peer.METHODS[:2]]
    if baselines:peer.validate_state(view(data,{m:m for m in baselines}),domain,dict(cfg,methods=baselines),clients,seed,complete)
    initial=None
    for method,record in data['methods'].items():
        first=[{k:e[k] for k in peer.OBSERVATION_FIELDS} for e in record['events'][:len(clients)]]
        if initial is None:initial=first
        elif first != initial:raise RuntimeError('Spectral arms have different factory observations.')
    for family in families(dict(domain=domain),cfg,phis):
        names=family['names'];original=deepcopy(view(data,names))
        width=family['phi'].shape[1]
        for old,record in original['methods'].items():
            for event in record['events']:event['method']=old
            for snapshot in record['snapshots'].values():
                snapshot['model_payload_bytes']=0;snapshot['payload_bytes']=snapshot['peer_payload_bytes']
            for log in record['learning']:
                log.update(model_payload_bytes=0,payload_bytes=log['peer_payload_bytes'],
                    cumulative_model_payload_bytes=0,cumulative_payload_bytes=log['cumulative_peer_payload_bytes'])
            record['model_payload_bytes']=0;record['payload_bytes']=record['peer_payload_bytes']
        peer.validate_state(original,domain,family_cfg(cfg,family['rank']),clients,seed,complete)
        for old,new in names.items():
            record=data['methods'][new];total=0
            for log in record['learning']:
                expected=joint.model_payload_bytes(log['eligible_training_counts'],width,
                    len(cfg['hierarchical']['sharing_fractions'])) if old==joint.SHARED else 0
                total+=expected
                if (log['model_payload_bytes']!=expected or log['payload_bytes']!=expected+log['peer_payload_bytes'] or
                    log['cumulative_model_payload_bytes']!=total or
                    log['cumulative_payload_bytes']!=total+log['cumulative_peer_payload_bytes']):
                    raise RuntimeError('Spectral model communication differs.')
            if record['model_payload_bytes']!=total or record['payload_bytes']!=total+record['peer_payload_bytes']:
                raise RuntimeError('Spectral total payload differs.')
            for snapshot in record['snapshots'].values():
                prior=record['learning'][snapshot['round']-2] if snapshot['round']>1 else {}
                expected=prior.get('cumulative_model_payload_bytes',0)
                if snapshot['model_payload_bytes']!=expected or snapshot['payload_bytes']!=expected+snapshot['peer_payload_bytes']:
                    raise RuntimeError('Spectral snapshot contains unused/later training.')
            if data['round']:
                posterior=joint.cost_posterior(record['histories'],family['phi'],cfg['gp'],cfg['hierarchical'],old==joint.SHARED)
                actual=record['cost_posterior']
                if actual['information_round']!=data['round']:
                    raise RuntimeError('Stale or future spectral posterior.')
                for key,value in posterior.items():
                    if not np.allclose(actual[key],value,rtol=1e-10,atol=1e-10):
                        raise RuntimeError('Spectral posterior differs: '+key)


def run_fleet(domain,cfg,clients,seed,observe_batch,phis,checkpoint=None,state=None,stop_round=None):
    data=deepcopy(state) if state is not None else fresh_state(domain,cfg,clients)
    if data['round']:validate_state(data,domain,cfg,clients,seed,phis)
    limit=cfg['queries_per_client'] if stop_round is None else stop_round
    if not data['round']<=limit<=cfg['queries_per_client']:raise ValueError('Invalid stopping round.')
    while data['round']<limit:
        plans=plan_round(data,domain,cfg,clients,seed,phis)
        observations=observe_batch(plans,data['oracle_cache'])
        if len(observations)!=len(plans):raise RuntimeError('Oracle observation count differs.')
        commit_round(data,plans,observations,domain,cfg,clients,phis)
        if checkpoint is not None:checkpoint(data)
    return data


def replay_methods(data,domain,cfg,clients,seed,phis):
    methods=[m for m in cfg['methods'] if m not in peer.METHODS[:2]]
    def recorded(plans,cache):
        observations=[]
        for plan in plans:
            event=data['methods'][plan['method']]['events'][(plan['round']-1)*len(clients)+plan['client']]
            if any(plan[k]!=event[k] for k in plan):raise RuntimeError('Spectral decision replay differs.')
            observations.append({k:event[k] for k in peer.OBSERVATION_FIELDS if k!='action'})
        return observations
    replay=run_fleet(domain,dict(cfg,methods=methods),clients,seed,recorded,phis,stop_round=data['round'])
    for method in methods:
        if replay['methods'][method]['events']!=data['methods'][method]['events']:
            raise RuntimeError('Spectral event trace replay differs.')
        for a,b in zip(replay['methods'][method]['learning'],data['methods'][method]['learning']):
            for key in ('expected_sharing_fraction','sharing_weights','component_log_evidence'):
                if not np.allclose(a[key],b[key],rtol=1e-10,atol=1e-10):
                    raise RuntimeError('Spectral learning replay differs.')
