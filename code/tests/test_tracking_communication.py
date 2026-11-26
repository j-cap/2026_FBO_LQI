"""Codec, remote-data isolation, periodic contacts, checkpoint resume and audit gates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import tracking_communication as runner
from src.tracking_communication import fresh_state, visible_histories, publish, run_fleet, validate_state
from src.tracking_communication_wire import COST, PEER, COST_RECORD, PEER_RECORD, encode, decode, exchange
from src.tracking_dense_gain import A, D, E, ExactCostSnapshot, METHODS


class CommunicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = runner.setup(smoke=True)

    def test_packet_sizes_roundtrip_and_recipient_accounting(self):
        self.assertEqual(COST_RECORD.size, 20)
        self.assertEqual(PEER_RECORD.size, 24)
        record = (0, 1, 12, float(np.nextafter(1., 2.)))
        packet = encode(COST, 4, [record])
        self.assertEqual(len(packet), 36)
        self.assertEqual(decode(packet, COST, 4, 3, 99), [record])
        received, stats = exchange(COST, 4, [[record], [], []], 3, 99)
        self.assertEqual(received, [record])
        self.assertEqual(stats['body_bytes'], 20 * 4)
        self.assertEqual(stats['application_bytes'], (36 + 16 + 16) + 3 * 36)
        self.assertEqual(stats['packet_deliveries'], 6)
        self.assertEqual(stats['coordinator_broadcasts'], 1)

    def test_invalid_records_packets_and_foreign_sender_rejected(self):
        good = encode(COST, 2, [(0, 1, 4, 1.)])
        for packet, channel, round_id, expected in [
                (good[:-1], COST, 2, None), (good, PEER, 2, None),
                (good, COST, 1, None), (good, COST, 2, 1),
                (encode(COST, 2, [(0, 3, 4, 1.)]), COST, 2, None),
                (encode(COST, 2, [(0, 1, 99, 1.)]), COST, 2, None),
                (encode(COST, 2, [(0, 1, 4, float('nan'))]), COST, 2, None),
                (encode(COST, 2, [(0, 1, 4, 0.)]), COST, 2, None),
                (encode(COST, 2, [(0, 1, 4, 1.)] * 2), COST, 2, None),
                (encode(PEER, 2, [(0, 1, 4, 3, 1.)]), PEER, 2, None)]:
            with self.subTest(channel=channel, round_id=round_id, expected=expected), self.assertRaises(ValueError):
                decode(packet, channel, round_id, 3, 99, expected)
        with self.assertRaises(ValueError):
            exchange(COST, 1, [[], []], 3, 99)

    def test_only_delivered_peer_costs_are_visible_and_own_data_stays_fresh(self):
        s = self.state
        data = fresh_state(s['cfg'], s['clients'])['methods'][D]
        data['network']['summaries'] = {'1:1': [1, 1, 1, 2.], '2:1': [2, 1, 1, 3.]}
        data['histories'] = [[dict(action=0, eligible=True, J=1.)],
            [dict(action=1, eligible=True, J=2.), dict(action=2, eligible=True, J=7.)],
            [dict(action=1, eligible=True, J=3.)]]
        visible = visible_histories(data, 0, 3)
        self.assertEqual([len(h) for h in visible], [1, 1, 1])
        self.assertEqual(visible[1][0]['action'], visible[2][0]['action'])
        before = ExactCostSnapshot(visible, s['domain'], s['cfg']['gp'], s['cfg']['hierarchical'], True).predict(0, [10, 50])
        data['histories'][1][-1]['J'] = 10000.
        after = ExactCostSnapshot(visible_histories(data, 0, 3), s['domain'], s['cfg']['gp'], s['cfg']['hierarchical'], True).predict(0, [10, 50])
        np.testing.assert_array_equal(before, after)
        data['histories'][0].append(dict(action=3, eligible=True, J=4.))
        fresh = visible_histories(data, 0, 3)
        self.assertEqual(len(fresh[0]), 2)
        own_updated = ExactCostSnapshot(fresh, s['domain'], s['cfg']['gp'], s['cfg']['hierarchical'], True).predict(0, [10, 50])
        self.assertFalse(np.array_equal(before, own_updated))
        data['histories'][0].append(dict(action=5, eligible=False, J=None))
        self.assertEqual(len(visible_histories(data, 0, 3)[0]), 2)

    @staticmethod
    def observe(plans, cache):
        # A deterministic test oracle with occasional failed controllers.
        return [dict(eligible=p['action'] % 19 != 18,
            J=float(1 + (p['action'] % 17) / 20) if p['action'] % 19 != 18 else None,
            complete=p['action'] % 19 != 18, feasible=True, integration_resolved=True, substeps=4) for p in plans]

    def test_contacts_round_resume_json_replay_and_tamper_detection(self):
        s = self.state
        cfg, clients, domain, seed = s['cfg'], s['clients'], s['domain'], s['jobs'][0]['seed']
        first = run_fleet(domain, cfg, clients, seed, 4, self.observe, stop_round=3)
        for method in METHODS:
            data = first['methods'][method]
            self.assertEqual(data['payload_bytes'], 0)
            self.assertEqual(data['network']['summaries'], {})
        fourth = run_fleet(domain, cfg, clients, seed, 4, self.observe, state=first, stop_round=4)
        for data in fourth['methods'].values():
            self.assertEqual(data['network']['last_contact_round'], 4)
            self.assertEqual(data['snapshots']['12']['payload_bytes'], 0)
        original = deepcopy(fourth)
        resumed = run_fleet(domain, cfg, clients, seed, 4, self.observe, state=json.loads(json.dumps(fourth)))
        self.assertEqual(fourth, original)
        for method, data in resumed['methods'].items():
            self.assertEqual([r['round'] for r in data['network']['contacts']], [4, 8])
            self.assertEqual(data['snapshots']['24']['packet_deliveries'], (12 if method == D else 6))
            for event in data['events'][12:15]:
                self.assertEqual(event['remote_information_round'], 4)
                self.assertEqual(event['remote_age_rounds'], 0)
            for event in data['events'][15:18]:
                self.assertEqual(event['remote_information_round'], 4)
                self.assertEqual(event['remote_age_rounds'], 1)
            if method != D:
                self.assertEqual(data['network']['summaries'], {})
        with patch('src.tracking_budget.minimize', runner.prohibited), patch('src.tracking_six_speed.minimize', runner.prohibited):
            validate_state(resumed, domain, cfg, clients, seed, 4, complete=True, replay=True)
        for field in ('bytes', 'sha256'):
            altered = deepcopy(resumed)
            packet = altered['methods'][D]['network']['contacts'][0]['cost']['packets'][0]
            packet[field] = 999 if field == 'bytes' else 'corrupt'
            with self.assertRaises(RuntimeError):
                validate_state(altered, domain, cfg, clients, seed, 4, complete=True)
        altered = deepcopy(resumed)
        altered['methods'][D]['events'][12]['remote_information_round'] = 5
        with self.assertRaises(RuntimeError):
            validate_state(altered, domain, cfg, clients, seed, 4, complete=True)

    def test_new_cost_records_only_failures_stay_private_and_duplicate_ids_conflict(self):
        s = self.state
        state = run_fleet(s['domain'], s['cfg'], s['clients'], s['jobs'][0]['seed'], 2, self.observe, stop_round=4)
        data = state['methods'][D]
        expected = sum(o['eligible'] for h in data['histories'] for o in h)
        self.assertEqual(sum(c['cost']['records'] for c in data['network']['contacts']), expected)
        self.assertEqual(len(data['network']['summaries']), expected)
        # A second packet containing an existing UID with changed value cannot silently overwrite it.
        altered = deepcopy(data)
        altered['network']['last_contact_round'] = 0
        key = next(iter(altered['network']['summaries']))
        altered['network']['summaries'][key][-1] += 1.
        with self.assertRaisesRegex(RuntimeError, 'Conflicting'):
            publish(altered, D, s['domain'], 3, 4)

    def test_design_payload_and_shared_validation_union(self):
        full = runner.setup()
        self.assertEqual(len(full['pool']), 1292)
        self.assertEqual(len(full['jobs']), 9)
        self.assertEqual(sum(len(full['clients']) * full['cfg']['queries_per_client'] * 4 for _ in full['jobs']), 7776)
        self.assertEqual(set(runner.payload(full)), {'cfg', 'physics', 'clients', 'pool', 'refs', 'domain', 'out', 'signature'})
        self.assertFalse(set(runner.payload(full)) & {'source_out', 'calibration', 'validation', 'audit_specification', 'reference_input_hashes'})
        selection = {c: set(full['domain'].candidates[:3]) for c in full['domain'].contexts}
        tasks = runner.evaluation_tasks(full, selection)
        self.assertEqual(sum(len(t['candidates']) for t in tasks), 18 * 3 * 4)
        self.assertEqual(len({t['noise_seed'] for t in tasks}), 4)
        self.assertEqual(len({(t['context'], t['repeat'], t['chunk']) for t in tasks}), len(tasks))
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(self.state, 1)


if __name__ == '__main__':
    unittest.main()
