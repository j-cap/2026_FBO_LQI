"""Full-controller replay of archived programmes under lossless deltas."""
from copy import deepcopy

from . import tracking_communication as previous
from . import tracking_peer_cost as peer
from .tracking_delta_catalogue import run_fleet, validate_state


def replay(source, domain, cfg, clients):
    recorded, interval, seed = source['state'], source['interval'], source['seed']
    n, contacts = len(clients), []
    def observe(plans, cache):
        observations = []
        for plan in plans:
            expected = recorded['methods'][plan['method']]['events'][(plan['round'] - 1) * n + plan['client']]
            previous.old.assert_equivalent(plan, {k: expected[k] for k in plan})
            observations.append({k: expected[k] for k in peer.OBSERVATION_FIELDS if k != 'action'})
        return observations
    def checkpoint(state):
        r = state['round']
        if r % interval:
            return
        for method, data in state['methods'].items():
            # Reconstruct the original full publication and compare its exact
            # packet hashes with the frozen run. This shadow state is never
            # passed back to a learner or an observation callback.
            shadow = deepcopy(data)
            shadow['network']['last_contact_round'] = r - interval
            full = previous.publish(shadow, method, domain, n, r)
            old_contact = recorded['methods'][method]['network']['contacts'][r // interval - 1]
            actual = data['network']['contacts'][-1]
            if full['peer'] != old_contact['peer'] or actual['cost'] != old_contact['cost']:
                raise RuntimeError('Source full catalogue or cost packets differ.')
            if data['network']['catalogue'] != shadow['network']['catalogue'] or data['network']['summaries'] != shadow['network']['summaries']:
                raise RuntimeError('Delta catalogue or eligible-cost cache differs.')
            expected_cache = {f'{record[0]}:{domain.actions[record[2]][0]}': record for record in shadow['network']['catalogue']}
            if any(cache != expected_cache for cache in data['network']['peer_receiver_caches']):
                raise RuntimeError('A recipient lost catalogue information.')
            contacts.append(dict(method=method, round=r, full_peer_bytes=full['peer']['application_bytes'],
                delta_peer_bytes=actual['peer']['application_bytes'], changed_records=actual['peer']['records'],
                catalogue_records=full['peer']['records'], all_recipient_caches_identical=True))
    state = run_fleet(domain, cfg, clients, seed, interval, observe, checkpoint=checkpoint, recorded=recorded)
    validate_state(state, domain, cfg, clients, seed, interval, complete=True)
    traffic = []
    for method, data in state['methods'].items():
        original = recorded['methods'][method]
        previous.old.assert_equivalent(data['events'], original['events'])
        if data['histories'] != original['histories']:
            raise RuntimeError('Paid local histories differ.')
        for budget, snapshot in data['snapshots'].items():
            if any(snapshot[k] != original['snapshots'][budget][k] for k in ('round', 'recommendations', 'queries_per_client')):
                raise RuntimeError('A local recommendation differs.')
        final = str(cfg['fleet_budgets'][-1])
        traffic.append(dict(method=method, original_used_bytes=original['snapshots'][final]['payload_bytes'],
            delta_used_bytes=data['snapshots'][final]['payload_bytes'], original_all_bytes=original['payload_bytes'],
            delta_all_bytes=data['payload_bytes'], packet_deliveries=data['snapshots'][final]['packet_deliveries']))
        if data['snapshots'][final]['packet_deliveries'] != original['snapshots'][final]['packet_deliveries']:
            raise RuntimeError('Lossless deltas changed packet/contact counts.')
    return dict(seed=seed, interval=interval, paid_decisions=n * cfg['queries_per_client'] * 4,
        all_decisions_identical=True, all_recipient_caches_identical=True, contacts=contacts, traffic=traffic,
        new_simulations=0, new_optimization=0)
