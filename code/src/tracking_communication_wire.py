"""Versioned application packets; byte counts come from actual serialization."""
import hashlib
import math
import struct

HEADER = struct.Struct('<IIII')  # version, channel, publication round, record count
COST_RECORD = struct.Struct('<IIId')  # client, local trial, public input, J
PEER_RECORD = struct.Struct('<IIIId')  # same IDs, source-mode code, J
VERSION, COST, PEER = 1, 1, 2
MODES = ['factory_initialization', 'peer_gain_verification', 'feasibility_penalized_LCB']


def encode(channel, publication_round, records):
    if channel not in (COST, PEER) or not 1 <= publication_round < 2**32:
        raise ValueError('Invalid packet channel or round.')
    layout = COST_RECORD if channel == COST else PEER_RECORD
    body = b''.join(layout.pack(*record) for record in records)
    return HEADER.pack(VERSION, channel, publication_round, len(records)) + body


def decode(packet, channel, publication_round, clients, actions, expected_client=None):
    if channel not in (COST, PEER):
        raise ValueError('Unknown channel.')
    if len(packet) < HEADER.size:
        raise ValueError('Truncated packet header.')
    version, kind, round_id, count = HEADER.unpack_from(packet)
    layout = COST_RECORD if channel == COST else PEER_RECORD
    if (version, kind, round_id) != (VERSION, channel, publication_round) or len(packet) != HEADER.size + count * layout.size:
        raise ValueError('Packet identity or length differs.')
    records = [layout.unpack_from(packet, HEADER.size + j * layout.size) for j in range(count)]
    for record in records:
        client, trial, action = record[:3]
        if (client >= clients or not 1 <= trial <= round_id or action >= actions or
                (expected_client is not None and client != expected_client) or
                not math.isfinite(record[-1]) or record[-1] <= 0 or
                (channel == PEER and record[3] >= len(MODES))):
            raise ValueError('Invalid record identity, provenance or cost.')
    if len({r[:2] for r in records}) != len(records):
        raise ValueError('Repeated observation identity within packet.')
    return records


def exchange(channel, round_id, outgoing, clients, actions):
    """One packet per sender and one coordinator batch delivered to every client.

    Empty packets retain their application header. Packet routing, IP/TLS,
    acknowledgements and physical network latency are outside this simulation.
    """
    if clients < 1 or len(outgoing) != clients:
        raise ValueError('Every client must supply one packet.')
    received, packets, uplink = [], [], 0
    for client, records in enumerate(outgoing):
        packet = encode(channel, round_id, records)
        received.extend(decode(packet, channel, round_id, clients, actions, client))
        uplink += len(packet)
        packets.append(dict(direction='uplink', sender=client, bytes=len(packet),
            sha256=hashlib.sha256(packet).hexdigest(), records=len(records)))
    broadcast = encode(channel, round_id, received)
    replicated = [decode(broadcast, channel, round_id, clients, actions) for _ in range(clients)]
    if any(rows != received for rows in replicated):
        raise RuntimeError('Replicated receiver caches differ.')
    packets.append(dict(direction='broadcast', recipients=clients, bytes_per_recipient=len(broadcast),
        sha256=hashlib.sha256(broadcast).hexdigest(), records=len(received)))
    record_size = COST_RECORD.size if channel == COST else PEER_RECORD.size
    return replicated[0], dict(uplink_bytes=uplink, downlink_bytes=clients * len(broadcast),
        body_bytes=len(received) * record_size * (clients + 1),
        application_bytes=uplink + clients * len(broadcast), packet_deliveries=2 * clients,
        coordinator_broadcasts=1, records=len(received), packets=packets)
