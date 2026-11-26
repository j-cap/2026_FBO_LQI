"""Persistent sender/coordinator/receiver caches for lossless peer updates."""
from copy import deepcopy

from .tracking_communication_wire import PEER, MODES
from .tracking_delta_wire import exchange


def empty_cache(clients):
    return dict(peer_protocol='delta_catalogue_v1', peer_sender_caches=[{} for _ in range(clients)],
        peer_server_cache={}, peer_receiver_caches=[{} for _ in range(clients)])


def local_catalogues(data, domain, clients):
    result = []
    for client, history in enumerate(data['histories']):
        current = {}
        for context in domain.contexts:
            good = [(trial + 1, o) for trial, o in enumerate(history) if o['eligible'] and domain.actions[o['action']][0] == context]
            if good:
                trial, observation = min(good, key=lambda x: (x[1]['J'], domain.actions[x[1]['action']][1]))
                event = data['events'][(trial - 1) * clients + client]
                current[context] = [client, trial, observation['action'], MODES.index(event['mode']), observation['J']]
        result.append(current)
    return result


def deliver(data, domain, clients, round_id):
    network = data['network']
    if network['peer_protocol'] != 'delta_catalogue_v1' or round_id <= network['last_contact_round']:
        raise RuntimeError('Invalid delta protocol or repeated contact.')
    current = local_catalogues(data, domain, clients)
    previous = network['peer_sender_caches']
    baseline = {f'{i}:{context}': record for i, rows in enumerate(previous) for context, record in rows.items()}
    if baseline != network['peer_server_cache'] or any(cache != baseline for cache in network['peer_receiver_caches']):
        raise RuntimeError('Sender, coordinator or recipient cache is out of sync.')
    outgoing = []
    for client, rows in enumerate(current):
        if not set(previous[client]) <= set(rows):
            raise RuntimeError('Catalogue removal requires an explicit reset protocol.')
        changes = []
        for context, record in rows.items():
            old = previous[client].get(context)
            if old != record:
                if old is not None and (record[1] <= old[1] or record[-1] > old[-1]):
                    raise RuntimeError('Stale or worsening catalogue update.')
                changes.append(record)
        outgoing.append(changes)
    received, statistics = exchange(PEER, round_id, outgoing, clients, len(domain.actions))
    target = deepcopy(baseline)
    for record in received:
        key = f'{record[0]}:{domain.actions[record[2]][0]}'
        target[key] = list(record)
    expected = {f'{i}:{context}': record for i, rows in enumerate(current) for context, record in rows.items()}
    if target != expected:
        raise RuntimeError('Delta delivery does not reconstruct the full catalogue.')
    receivers = []
    for old_cache in network['peer_receiver_caches']:
        cache = deepcopy(old_cache)
        for record in received:
            cache[f'{record[0]}:{domain.actions[record[2]][0]}'] = list(record)
        if cache != expected:
            raise RuntimeError('Recipient catalogue differs after delta delivery.')
        receivers.append(cache)
    network.update(peer_sender_caches=deepcopy(current), peer_server_cache=target, peer_receiver_caches=receivers)
    network['catalogue'] = [deepcopy(current[i][context]) for i in range(clients) for context in domain.contexts if context in current[i]]
    return statistics
