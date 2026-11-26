"""Lossless cache semantics, causal replay, fresh design and programme uncertainty."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import tracking_delta_catalogue as runner
from src import tracking_delta_catalogue as delta
from src.tracking_communication_wire import PEER, decode as old_decode
from src.tracking_delta_wire import encode, decode
from src.tracking_dense_gain import A, D, METHODS
from src.tracking_delta_report import bootstrap_effects


class DeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = runner.setup(smoke=True)

    def test_versioned_packet_keeps_exact_size_and_rejects_old_decoder(self):
        records = [(0, 1, 0, 0, float(np.nextafter(1., 2.)))]
        packet = encode(PEER, 1, records)
        self.assertEqual(len(packet), 16 + 24)
        self.assertEqual(decode(packet, PEER, 1, 3, 100), records)
        with self.assertRaises(ValueError):
            old_decode(packet, PEER, 1, 3, 100)
        with self.assertRaises(ValueError):
            decode(packet[:-1], PEER, 1, 3, 100)

    def catalogue_data(self):
        s = self.state
        data = delta.fresh_state(s['cfg'], s['clients'])['methods'][A]
        for i in range(3):
            data['histories'][i].append(dict(action=0, eligible=True, J=2.))
            data['events'].append(dict(mode='factory_initialization'))
        delta.publish(data, A, s['domain'], 3, 1)
        return data

    def test_changed_entries_only_all_recipient_caches_and_empty_headers(self):
        s, data = self.state, self.catalogue_data()
        self.assertEqual(data['network']['contacts'][0]['peer']['records'], 3)
        for i in range(3):
            data['histories'][i].append(dict(action=1, eligible=i != 2, J=1. if i == 0 else 3. if i == 1 else None))
            data['events'].append(dict(mode='feasibility_penalized_LCB'))
        delta.publish(data, A, s['domain'], 3, 2)
        self.assertEqual(data['network']['contacts'][1]['peer']['records'], 1)
        for cache in data['network']['peer_receiver_caches']:
            self.assertEqual(cache, data['network']['peer_server_cache'])
            self.assertEqual(len(cache), 3)
        delta.publish(data, A, s['domain'], 3, 3)
        last = data['network']['contacts'][-1]['peer']
        self.assertEqual(last['records'], 0)
        self.assertEqual(last['application_bytes'], 16 * 3 * 2)
        self.assertEqual(last['packet_deliveries'], 6)

    def test_desynchronized_cache_repeated_contact_and_removal_rejected(self):
        s, original = self.state, self.catalogue_data()
        data = deepcopy(original)
        data['network']['peer_receiver_caches'][1].clear()
        with self.assertRaisesRegex(RuntimeError, 'out of sync'):
            delta.publish(data, A, s['domain'], 3, 2)
        with self.assertRaisesRegex(RuntimeError, 'repeated contact'):
            delta.publish(deepcopy(original), A, s['domain'], 3, 1)
        data = deepcopy(original)
        data['histories'][1].clear()
        with self.assertRaisesRegex(RuntimeError, 'removal'):
            delta.publish(data, A, s['domain'], 3, 2)

    @staticmethod
    def observe(plans, cache):
        return [dict(eligible=True, J=float(1 + (p['action'] % 17) / 20), complete=True,
            feasible=True, integration_resolved=True, substeps=4) for p in plans]

    def test_json_resume_replay_and_cache_packet_timestamp_corruption(self):
        s = self.state
        args = s['domain'], s['cfg'], s['clients'], s['jobs'][0]['seed'], 2
        first = delta.run_fleet(*args, self.observe, stop_round=3)
        original = deepcopy(first)
        complete = delta.run_fleet(*args, self.observe, state=json.loads(json.dumps(first)))
        self.assertEqual(first, original)
        with patch('src.tracking_budget.minimize', runner.prohibited), patch('src.tracking_six_speed.minimize', runner.prohibited):
            delta.validate_state(complete, *args, complete=True, replay=True)
        for kind in ('cache', 'packet', 'timestamp'):
            altered = deepcopy(complete)
            if kind == 'cache':
                altered['methods'][D]['network']['peer_receiver_caches'][1].clear()
            elif kind == 'packet':
                altered['methods'][D]['network']['contacts'][0]['peer']['packets'][0]['sha256'] = 'changed'
            else:
                altered['methods'][D]['events'][3]['remote_information_round'] = 1
            with self.assertRaises(RuntimeError):
                delta.validate_state(altered, *args, complete=True)

    def test_full_confirmation_design_fresh_validation_and_evaluator_isolation(self):
        s = runner.setup()
        self.assertEqual(len(s['jobs']), 20)
        self.assertEqual(len(s['pool']), 1292)
        self.assertEqual(len(s['clients']) * len(s['jobs']) * s['cfg']['queries_per_client'] * len(METHODS), 17280)
        self.assertEqual(set(j['interval'] for j in s['jobs']), {1, 2})
        self.assertTrue(all(j['seed'] >= 10000 for j in s['jobs']))
        self.assertEqual(set(runner.payload(s)), {'cfg', 'physics', 'clients', 'pool', 'refs', 'domain', 'out', 'signature'})
        tasks = runner.evaluation_tasks(s, {c: set(s['domain'].candidates[:3]) for c in s['domain'].contexts})
        self.assertEqual(sum(len(t['candidates']) for t in tasks), 18 * 3 * 4)
        self.assertEqual(len({t['noise_seed'] for t in tasks}), 4)
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(self.state, 1)

    def test_uncertainty_resamples_programmes_and_rejects_client_pseudoreplication(self):
        effects = pd.DataFrame([dict(interval=2, method=D, baseline=A, fleet_budget=216,
            seed=i, success_delta_pp=value) for i, value in enumerate([-8., 1., 9.])])
        keys = ['interval', 'method', 'baseline', 'fleet_budget']
        result = bootstrap_effects(effects, self.state['requested'], keys)
        self.assertEqual(result.programme_units.iloc[0], 3)
        self.assertLess(result.lower_percentile_pp.iloc[0], 0)
        self.assertGreater(result.upper_percentile_pp.iloc[0], 0)
        with self.assertRaises(RuntimeError):
            bootstrap_effects(pd.concat([effects, effects]), self.state['requested'], keys)
        single = bootstrap_effects(effects.iloc[:1], self.state['requested'], keys)
        self.assertFalse(single.uncertainty_available.iloc[0])
        self.assertTrue(pd.isna(single.lower_percentile_pp.iloc[0]))


if __name__ == '__main__':
    unittest.main()
