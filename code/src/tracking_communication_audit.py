"""No-rollout equivalence audit of the compact every-round protocol."""
import time

import numpy as np

from . import tracking_peer_cost as peer
from . import tracking_six_speed as old
from .tracking_dense_gain import A, B, D, E, ExactCostSnapshot
from .tracking_communication import run_fleet, validate_state


def audit_replay(source, domain, cfg, clients):
    seed, recorded = source['seed'], source['state']
    n = len(clients)
    references, checks = {}, []
    started = time.perf_counter()
    def compare_predictions(method, client, previous_round, actions, mean, variance, model):
        key = method, previous_round
        if key not in references:
            histories = [h[:previous_round] for h in recorded['methods'][method]['histories']]
            references[key] = ExactCostSnapshot(histories, domain, cfg['gp'], cfg['hierarchical'], method == D)
        reference = references[key]
        em, ev = reference.predict(client, actions)
        np.testing.assert_allclose(mean, em, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(variance, ev, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(model.weights, reference.weights, rtol=2e-8, atol=2e-8)
        covariance_error = 0.
        if method == D:
            if model.support != reference.support:
                raise RuntimeError('Decoded active input dictionary differs.')
            for actual, expected in zip(model.components, reference.components):
                np.testing.assert_allclose(actual[1], expected[1], rtol=2e-8, atol=2e-8)
                np.testing.assert_allclose(actual[2], expected[2], rtol=2e-8, atol=2e-8)
                covariance_error = max(covariance_error, float(np.max(abs(actual[2] - expected[2]), initial=0.)))
        checks.append(dict(method=method, client=client, round=previous_round + 1, inputs=len(actions),
            maximum_mean_error=float(np.max(abs(mean - em), initial=0.)),
            maximum_variance_error=float(np.max(abs(variance - ev), initial=0.)),
            maximum_weight_error=float(np.max(abs(model.weights - reference.weights), initial=0.)),
            maximum_coefficient_covariance_error=covariance_error))
    def observe(plans, cache):
        outcomes = []
        for plan in plans:
            event = recorded['methods'][plan['method']]['events'][(plan['round'] - 1) * n + plan['client']]
            # New cache-age fields have no legacy counterpart. Every original
            # decision, GP diagnostic and peer-provenance field is compared.
            keys = [k for k in plan if k in event]
            old.assert_equivalent({k: plan[k] for k in keys}, {k: event[k] for k in keys})
            outcomes.append({k: event[k] for k in peer.OBSERVATION_FIELDS if k != 'action'})
        return outcomes
    compact = run_fleet(domain, cfg, clients, seed, 1, observe, recorded=recorded,
        prediction_hook=compare_predictions)
    validate_state(compact, domain, cfg, clients, seed, 1, complete=True)
    traffic = []
    for method, data in compact['methods'].items():
        original = recorded['methods'][method]
        if data['histories'] != original['histories']:
            raise RuntimeError('Decoded histories differ.')
        for actual, expected in zip(data['events'], original['events']):
            old.assert_equivalent({k: actual[k] for k in expected}, expected)
        for budget, snapshot in data['snapshots'].items():
            if any(snapshot[k] != original['snapshots'][budget][k] for k in ('recommendations', 'queries_per_client', 'round')):
                raise RuntimeError('Local controller recommendations differ.')
        final = str(cfg['fleet_budgets'][-1])
        before = original['snapshots'][final]
        after = data['snapshots'][final]
        body_model = sum(log['cost']['body_bytes'] for log in data['network']['contacts']
            if log['round'] < cfg['queries_per_client'] and log['cost'])
        body_peer = sum(log['peer']['body_bytes'] for log in data['network']['contacts']
            if log['round'] < cfg['queries_per_client'])
        if method == D:
            samples = sum(o['eligible'] for h in original['histories'] for o in h[:-1])
            if body_model != samples * 20 * (n + 1):
                raise RuntimeError('Incremental cost records were repeated or omitted.')
        if body_peer != 2 * before['peer_payload_bytes']:
            raise RuntimeError('Catalogue metadata accounting differs.')
        traffic.append(dict(method=method, legacy_dense_model_payload_bytes=before['model_payload_bytes'],
            legacy_peer_reference_bytes=before['peer_payload_bytes'], legacy_total_reference_bytes=before['payload_bytes'],
            compact_cost_record_body_bytes=body_model, peer_record_body_bytes=body_peer,
            application_cost_bytes=after['model_payload_bytes'], application_peer_bytes=after['peer_payload_bytes'],
            application_total_bytes=after['payload_bytes'], packet_deliveries=after['packet_deliveries'],
            all_24_round_updates_application_bytes=data['payload_bytes'],
            legacy_model_plus_same_peer_application_bytes=before['model_payload_bytes'] + after['peer_payload_bytes']))
    return dict(seed=seed, all_decisions_and_recommendations_identical=True,
        posterior_checks=checks, traffic=traffic, paid_decisions=n * cfg['queries_per_client'] * 4,
        new_simulations=0, new_optimization=0, elapsed_seconds=time.perf_counter() - started)
