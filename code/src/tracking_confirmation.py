"""Fresh physical fleet design and fixed-denominator confirmation analysis."""
from copy import deepcopy

import numpy as np
import pandas as pd

from .tracking_context import stable_minimum
from .tracking_online import seed_for
from .tracking_spectral import LEGACY, shared, local

METHODS = LEGACY[:2]+[shared(256),local(256)]
PAIRS = [(shared(256),LEGACY[0],'primary'),(shared(256),LEGACY[1],'secondary'),
         (shared(256),local(256),'sharing_control')]


def pinned_design(generated, saved):
    """Reload the exact saved physical values, tolerating platform roundoff only."""
    def check(a,b):
        if isinstance(a,dict):
            if not isinstance(b,dict) or set(a)!=set(b):
                raise RuntimeError('Pinned physical design keys differ.')
            for key in a:check(a[key],b[key])
        elif isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b):
                raise RuntimeError('Pinned physical design length differs.')
            for x,y in zip(a,b):check(x,y)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            if not isinstance(b,(int,float)) or not np.isclose(a,b,rtol=1e-13,atol=1e-14):
                raise RuntimeError('Pinned physical design values differ.')
        elif a!=b:
            raise RuntimeError('Pinned physical design identity differs.')
    check(generated,saved)
    return deepcopy(saved)


def design(base_vehicles, requested, smoke=False):
    """Independent bounded manufacturing scatter around the frozen nine slots.

    Fixed family/intervention composition is retained. No rollout or score is
    accepted by this function; all draws are kept. Client IDs reveal no family.
    """
    common = next(v for v in base_vehicles if v['id']=='compact_nominal')
    selected = base_vehicles if not smoke else [v for v in base_vehicles if v['variant']=='nominal']
    n = len(selected)
    groups = {'homogeneous':[dict(id=f'client_{i:02d}',plant_id='hom_compact',
        family=common['family'],params=deepcopy(common['params'])) for i in range(n)]}
    jobs, draws = [], []
    homogeneous_seeds = requested['homogeneous_seeds'][:1] if smoke else requested['homogeneous_seeds']
    for i, seed in enumerate(homogeneous_seeds):
        jobs.append(dict(condition='homogeneous',fleet_id='homogeneous',replicate=i,seed=seed))
    fleet_seeds = requested['fleet_seeds'][:1] if smoke else requested['fleet_seeds']
    repeats = 1 if smoke else requested['runs_per_fleet']
    bounds = requested['scatter']
    for number, seed in enumerate(fleet_seeds):
        fleet_id = f'fleet_{number:02d}'
        rng = np.random.default_rng(seed)
        clients = []
        for i, vehicle in enumerate(selected):
            p = deepcopy(vehicle['params'])
            z = {key:float(rng.uniform(-bound,bound)) for key,bound in bounds.items()}
            mf, lf = np.exp(z['log_mass_bound']),np.exp(z['log_wheelbase_bound'])
            length = p['l_f']+p['l_r']
            front_fraction = p['l_f']/length+z['cg_fraction_bound']
            if not 0 < front_fraction < 1:
                raise ValueError('Predeclared CG scatter violates positive axle distances.')
            p.update(m=p['m']*mf,I_z=p['I_z']*mf*lf**2*np.exp(z['log_inertia_shape_bound']),
                l_f=length*lf*front_fraction,l_r=length*lf*(1-front_fraction))
            p['pacejka_f_lat']['B'] *= np.exp(z['log_front_B_bound'])
            p['pacejka_r_lat']['B'] *= np.exp(z['log_rear_B_bound'])
            plant_id = fleet_id+'_'+vehicle['id']
            clients.append(dict(id=f'client_{i:02d}',plant_id=plant_id,family=vehicle['family'],params=p))
            draws.append(dict(fleet_id=fleet_id,fleet_seed=seed,client=f'client_{i:02d}',
                plant_id=plant_id,base_vehicle=vehicle['id'],family=vehicle['family'],draw=z,params=p))
        groups[fleet_id] = clients
        for replicate in range(repeats):
            jobs.append(dict(condition='heterogeneous',fleet_id=fleet_id,replicate=replicate,
                seed=requested['heterogeneous_seed_start']+number*requested['runs_per_fleet']+replicate))
    if len({j['seed'] for j in jobs})!=len(jobs):
        raise ValueError('Confirmation programme seeds must be distinct.')
    return groups,jobs,draws


def aggregate_candidates(batches):
    """Every declared holdout/calibration repeat must be present before calling."""
    grouped = {}
    for batch in batches:
        for row in batch['rows']:
            grouped.setdefault(row['candidate'],[]).append(row)
    return {candidate:dict(eligible=all(r['complete'] and r['feasible'] and r['integration_resolved'] for r in rows),
        J=float(np.mean([r['J'] for r in rows])) if all(r['complete'] and r['J'] is not None for r in rows) else None)
        for candidate,rows in grouped.items()}


def calibration_reference(aggregated):
    labels = sorted(c for c,row in aggregated.items() if row['eligible'])
    if not labels:
        return None
    return labels[stable_minimum([aggregated[c]['J'] for c in labels],labels)]


def evaluate_job(job,result,domain,cfg,references,validation):
    """Same metrics as development; unavailable references keep the denominator."""
    rows,cells,queries,learning = [],[],[],[]
    base = {k:job[k] for k in ('condition','fleet_id','replicate','seed')}
    for method,data in result['state']['methods'].items():
        queries.extend(dict(**base,**{k:v for k,v in e.items() if k!='recommendations'}) for e in data['events'])
        learning.extend(dict(**base,method=method,**r) for r in data['learning'])
        for budget in cfg['fleet_budgets']:
            snapshot = data['snapshots'][str(budget)]
            prefix = data['events'][:budget]
            for i,client in enumerate(result['clients']):
                common = dict(**base,method=method,fleet_budget=budget,client=client['id'],
                    plant_id=client['plant_id'],family=client['family'],calibration_rounds=snapshot['round'],
                    **{k:snapshot[k] for k in ('payload_bytes','peer_payload_bytes','model_payload_bytes')})
                outcomes = []
                for context,candidate in zip(domain.contexts,snapshot['recommendations'][i]):
                    ref = references[client['plant_id'],context]
                    val = validation[client['plant_id'],context,candidate] if candidate is not None else None
                    eligible = val is not None and bool(val['eligible'])
                    available = bool(ref['eligible'] and ref['J'] is not None and ref['J']>0)
                    gap = float(val['J']/ref['J']-1) if eligible and available else None
                    cell = dict(**common,context=context,candidate=candidate,eligible=eligible,
                        has_local_eligible_gain=candidate is not None,reference_available=available,
                        reference_candidate=ref['candidate'],reference_J=ref['J'] if available else None,
                        J=float(val['J']) if eligible else None,relative_gap=gap,
                        success=gap is not None and gap<=cfg['relative_tolerance'])
                    cells.append(cell);outcomes.append(cell)
                fully = all(c['eligible'] for c in outcomes)
                local_events = [e for e in prefix if e['client']==i]
                rows.append(dict(**common,actual_client_queries=len(local_events),fully_feasible=fully,
                    context_success=float(np.mean([c['success'] for c in outcomes])),
                    eligible_context_coverage=float(np.mean([c['has_local_eligible_gain'] for c in outcomes])),
                    validation_eligible_context_coverage=float(np.mean([c['eligible'] for c in outcomes])),
                    reference_coverage=float(np.mean([c['reference_available'] for c in outcomes])),
                    all_contexts_success=all(c['success'] for c in outcomes),
                    J=float(np.mean([c['J'] for c in outcomes])) if fully else None,
                    failed_queries=sum(not e['complete'] or not e['feasible'] for e in local_events),
                    unresolved_queries=sum(not e['integration_resolved'] for e in local_events),
                    ineligible_queries=sum(not e['eligible'] for e in local_events)))
    return tuple(pd.DataFrame(x) for x in (rows,cells,queries,learning))


def paired_tables(trials,cfg,requested,mode):
    aggregate,units = [],[]
    for (condition,budget),group in trials.groupby(['condition','fleet_budget'],sort=False):
        for method,baseline,role in PAIRS:
            keys = ['fleet_id','seed','client']
            a = group[group.method==method].set_index(keys).sort_index()
            b = group[group.method==baseline].set_index(keys).sort_index()
            if a.empty or not a.index.equals(b.index):
                raise RuntimeError('Unpaired confirmation comparison.')
            delta = a.context_success-b.context_success
            unit_ids = (a.index.get_level_values('fleet_id').to_numpy() if condition=='heterogeneous' else
                np.asarray(['seed_'+str(s) for s in a.index.get_level_values('seed')]))
            unit_effects = []
            for unit in sorted(set(unit_ids)):
                mask = unit_ids==unit
                aa,bb = a.iloc[np.flatnonzero(mask)],b.iloc[np.flatnonzero(mask)]
                good = aa.fully_feasible & bb.fully_feasible
                effect = float((aa.context_success-bb.context_success).mean())
                unit_effects.append(effect)
                units.append(dict(condition=condition,fleet_budget=budget,method=method,baseline=baseline,role=role,
                    replication_unit=unit,programme_runs=aa.index.get_level_values('seed').nunique(),
                    client_runs=len(aa),context_success_delta=effect,
                    eligible_method=int(aa.fully_feasible.sum()),eligible_baseline=int(bb.fully_feasible.sum()),
                    jointly_eligible_pairs=int(good.sum()),
                    mean_J_delta_joint=float((aa.loc[good,'J']-bb.loc[good,'J']).mean()) if good.any() else None))
            effects = np.asarray(unit_effects)
            lower,upper = None,None
            if len(effects)>=2:
                rng = np.random.default_rng(seed_for('confirmation_interval',requested['bootstrap_seed'],condition,budget,baseline))
                resampled = effects[rng.integers(0,len(effects),size=(requested['bootstrap_repeats'],len(effects)))].mean(axis=1)
                lower,upper = (float(x) for x in np.quantile(resampled,[.025,.975]))
            good = a.fully_feasible & b.fully_feasible
            ma,mb = (float(x.loc[good,'J'].mean()) for x in (a,b)) if good.any() else (None,None)
            complete_refs = bool((a.reference_coverage==1).all() and (b.reference_coverage==1).all())
            support = bool(mode=='full' and budget==cfg['fleet_budgets'][-1] and len(effects)==10 and
                complete_refs and lower is not None and lower>1e-12 and float(delta.mean())>1e-12 and
                a.fully_feasible.sum()>=b.fully_feasible.sum() and ma is not None and ma<mb)
            aggregate.append(dict(condition=condition,fleet_budget=budget,method=method,baseline=baseline,role=role,
                replication_units=len(effects),matched_client_runs=len(a),context_success_delta=float(delta.mean()),
                unit_bootstrap_lower=lower,unit_bootstrap_upper=upper,positive_units=int((effects>1e-12).sum()),
                negative_units=int((effects< -1e-12).sum()),tied_units=int((abs(effects)<=1e-12).sum()),
                worst_unit_context_delta=float(effects.min()),eligible_method=int(a.fully_feasible.sum()),
                eligible_baseline=int(b.fully_feasible.sum()),jointly_eligible_pairs=int(good.sum()),
                mean_J_method_joint=ma,mean_J_baseline_joint=mb,
                mean_J_delta_joint=ma-mb if ma is not None else None,
                relative_J_reduction=1-ma/mb if ma is not None else None,
                complete_reference_coverage=complete_refs,confirmation_supported=support))
    return pd.DataFrame(aggregate),pd.DataFrame(units)
