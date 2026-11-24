"""Trace paid seed-1102 discoveries and peer transfers; propose no new gains."""
from copy import deepcopy
from pathlib import Path

import tracking_spectral as runner
import numpy as np
import pandas as pd

from src.tracking_budget import dense_references, incumbents
from src.tracking_peer_cost import OBSERVATION_FIELDS, peer_proposal
from src.tracking_spectral import LEGACY, shared, local, run_fleet

OUT = runner.REPO/'docs/analysis/spectral_discovery_chains'
METHODS = LEGACY[:2]+[shared(256), local(256)]


def replay(result, state):
    clients = result['clients']
    cfg = dict(state['cfg'], methods=METHODS, spectral_ranks=[256])
    def recorded(plans, cache):
        rows = []
        for plan in plans:
            event = result['state']['methods'][plan['method']]['events'][
                (plan['round']-1)*len(clients)+plan['client']]
            if any(plan[k] != event[k] for k in plan):
                raise RuntimeError('Discovery replay differs from the paid decision.')
            rows.append({k:event[k] for k in OBSERVATION_FIELDS if k != 'action'})
        return rows
    actual = run_fleet(state['domain'], cfg, clients, result['seed'], recorded, state['phis'])
    for method in METHODS:
        if actual['methods'][method]['events'] != result['state']['methods'][method]['events']:
            raise RuntimeError('Discovery event trace differs.')


def root_event(events, event, nclients):
    seen = set()
    while event['mode'] == 'peer_gain_verification':
        key = (event['client'], event['local_evaluation'])
        if key in seen:
            raise RuntimeError('Cyclic peer provenance.')
        seen.add(key)
        source = events[(event['peer_donor_query']-1)*nclients+event['peer_donor']]
        if source['round'] >= event['round'] or source['candidate'] != event['candidate'] or not source['eligible']:
            raise RuntimeError('Invalid paid peer provenance.')
        event = source
    return event


def inspect(result, state, tables):
    refs, validation = tables
    clients, domain = result['clients'], state['domain']
    n = len(clients)
    rows, final_rows = [], []
    def near(i, context, candidate):
        if candidate is None:
            return False
        plant = clients[i]['plant_id']
        val, ref = validation[plant, context, candidate], refs[plant, context]
        return bool(val['eligible'] and ref['eligible'] and val['J'] <= 1.05*ref['J'])
    for method in METHODS:
        data = result['state']['methods'][method]
        events = data['events']
        histories = [[] for _ in clients]
        previous = None
        for number, event in enumerate(events):
            i = event['client']
            if number % n == 0:
                previous = deepcopy(histories)
            context = event['context']
            available = []
            tested = {o['action'] for o in previous[i]}
            for donor, history in enumerate(previous):
                if donor == i:
                    continue
                good = [o for o in history if o['eligible'] and domain.actions[o['action']][0] == context]
                if good:
                    best = min(good, key=lambda o:(o['J'],domain.actions[o['action']][1]))
                    candidate = domain.actions[best['action']][1]
                    if best['action'] not in tested and near(i,context,candidate):
                        available.append((donor,candidate))
            if event['round'] > 1:
                proposal, _ = peer_proposal(previous,events,domain,i,context)
                if (proposal is None) != (event['mode'] == 'feasibility_penalized_LCB'):
                    raise RuntimeError('Peer-priority barrier differs.')
            root = root_event(events,event,n)
            rows.append(dict(condition=result['condition'], seed=result['seed'], method=method,
                client=clients[i]['id'], plant_id=clients[i]['plant_id'], family=clients[i]['family'],
                round=event['round'], context=context, candidate=event['candidate'], mode=event['mode'],
                eligible=event['eligible'], observed_J=event['J'],
                validation_near_reference=near(i,context,event['candidate']),
                untried_near_reference_peer_incumbents=len(set(c for _,c in available)),
                near_reference_peer_available=bool(available),
                near_reference_peer_available_but_other_gain_selected=bool(available) and not near(i,context,event['candidate']),
                root_client=clients[root['client']]['id'], root_round=root['round'],
                root_mode=root['mode'], root_candidate=root['candidate'],
                peer_donor=event['peer_donor'], peer_donor_query=event['peer_donor_query']))
            histories[i].append({k:event[k] for k in OBSERVATION_FIELDS})
        for i, history in enumerate(histories):
            for context,candidate in zip(domain.contexts,incumbents(history,domain)):
                selected = [e for e in events if e['client'] == i and e['context'] == context and
                    e['candidate'] == candidate and e['eligible']]
                event = selected[0] if selected else None
                root = root_event(events,event,n) if event else None
                final_rows.append(dict(condition=result['condition'], method=method, client=clients[i]['id'],
                    context=context, candidate=candidate, validation_near_reference=near(i,context,candidate),
                    origin_client=clients[root['client']]['id'] if root else None,
                    origin_round=root['round'] if root else None, origin_mode=root['mode'] if root else None,
                    recipient_verification_round=event['round'] if event else None))
    return rows, final_rows


def main():
    state = runner.setup()
    results = []
    paths = []
    for condition in state['cfg']['conditions']:
        job = dict(condition=condition,seed=1102)
        path = runner.path_for(state,job)
        result = runner.read(path)
        runner.validate_job(state,job,result,complete=True)
        combined = runner.comparison_result(state,job,result)
        replay(combined,state)
        results.append(combined)
        paths.append(path)
        print('Replayed paid discovery decisions: '+condition,flush=True)
    # Validation is used only to describe the already-fixed paid query traces.
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv',float_precision='round_trip')
    tables = dense_references(raw[raw.role == 'calibration'],raw[raw.role == 'validation'],state['domain'])
    events, finals = [], []
    for result in results:
        rows, final_rows = inspect(result,state,tables)
        events.extend(rows);finals.extend(final_rows)
    events, finals = pd.DataFrame(events), pd.DataFrame(finals)
    summaries = []
    for key,g in events.groupby(['condition','method','context'],sort=False):
        good = g[g.eligible & g.validation_near_reference]
        discoveries = good[good['mode'] != 'peer_gain_verification']
        f = finals[(finals.condition == key[0]) & (finals.method == key[1]) & (finals.context == key[2])]
        summaries.append(dict(zip(['condition','method','context'],key),requests=len(g),
            bo_requests=int((g['mode'] == 'feasibility_penalized_LCB').sum()),
            peer_verifications=int((g['mode'] == 'peer_gain_verification').sum()),
            near_reference_gains_requested=int(g.validation_near_reference.sum()),
            eligible_near_reference_tests=len(good), paid_near_reference_discoveries=len(discoveries),
            first_near_reference_discovery_round=int(discoveries['round'].min()) if len(discoveries) else None,
            discovered_near_reference_candidates=','.join(sorted(discoveries.candidate.unique())),
            decisions_with_near_reference_peer_available=int(g.near_reference_peer_available.sum()),
            decisions_with_near_reference_peer_bypassed=int(g.near_reference_peer_available_but_other_gain_selected.sum()),
            final_near_reference_clients=int(f.validation_near_reference.sum())))
    summary = pd.DataFrame(summaries)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in [('paid_events',events),('final_origins',finals),('context_summary',summary)]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    runner.save(OUT/'verification.json',dict(campaign_signature=state['signature'],seed=1102,
        conditions=state['cfg']['conditions'],methods=METHODS,paid_events=len(events),
        all_four_method_decisions_replayed=True,strict_prior_round_peer_chains=True,
        validation_used_for_posthoc_annotation_only=True,new_rollouts=0,new_proposals=0,
        analysis_source_sha256=runner.digest(Path(__file__)),
        input_hashes={p.relative_to(runner.REPO).as_posix():runner.digest(p) for p in paths}))
    print(summary[summary.context.isin(['s20_f0.1','s30_f0.1'])].to_string(index=False))


if __name__ == '__main__':
    main()
