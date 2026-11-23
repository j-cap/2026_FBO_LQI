"""Scientific invariants and leakage/checkpoint checks for the new experiment.

Uses unittest so it also runs without installing a separate test dependency.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.signal import cont2discrete

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from src.tracking_task import (baseline_tangent_model, candidates, control, fleet, integrate_reference,
                               lateral_linearization, measurement_noise, oracle_features,
                               references, resolved_rollout, rhs, rollout)
from src.tracking_reference.cfl_dynamics import dbm_ct
from src.tracking_reference.cfl_controller import althoff_fb
from src.tracking_analysis import (aggregate, nearest_context, near_optimal_sets,
                                   q2, select_policies)
import tracking_diagnostic as runner


def config():
    return json.loads((ROOT/'config/runs/tracking_diagnostic.json').read_text())


class TrackingPhysicsTests(unittest.TestCase):
    def test_vectorized_rhs_matches_original_cfl_across_nonlinear_regimes(self):
        rng = np.random.default_rng(127)
        x = rng.normal(size=(35, 6))*[10, 10, .3, 8, 2, .4]
        x[:, 3] += 18
        x[:3, 3] = [0., -.1, -5.]
        u = rng.normal(size=(35, 2))*[.3, 10.]
        for vehicle in fleet(config()):
            p = vehicle['params']
            derivative, aux = rhs(x, u, p, auxiliary=True)
            for i in range(len(x)):
                expected, scalar = dbm_ct(p, x[i], u[i])
                np.testing.assert_allclose(derivative[i], expected, atol=2e-12, rtol=2e-12)
                self.assertAlmostEqual(aux['ay'][i], scalar['ay'], places=11)
                self.assertAlmostEqual(aux['util'][i], scalar['util'], places=12)
        self.assertTrue((aux['util'] <= 1+1e-12).all())

    def test_control_matches_original_including_inner_clip_and_limit_order(self):
        cfg = config(); p = fleet(cfg)[0]['params']; ref = references(cfg)[0]
        x = np.array([[0., -100., -.5, 21., .2, -.1], [2., 2., .2, 18., -.1, .2]])
        gain = np.array([cfg['gains']['nominal']]*2)*4
        previous = np.array([.19, -.18])
        k = 180
        actual, flags = control(x, ref, k, gain, previous, p, cfg)
        lim = dict(max_steer_angle_rad=np.deg2rad(12), max_lon_accel_mps2=3., max_lon_decel_mps2=6.)
        for i in range(2):
            current = dict(zip(('X', 'Y', 'psi', 'vx', 'vy', 'r'), x[i]))
            desired = dict(zip(('X', 'Y', 'psi'), ref['pose'][k]))
            desired.update(vx=ref['v'][k], r=ref['r'][k])
            g = dict(zip(('K1', 'K2', 'K3'), gain[i])); g.update(K5=1., K6=.1)
            fb, ax, _ = althoff_fb(current, desired, g, lim)
            angle = np.clip(fb+np.arctan((p['l_f']+p['l_r'])*ref['kappa'][k]),
                            -lim['max_steer_angle_rad'], lim['max_steer_angle_rad'])
            rate_step = np.deg2rad(cfg['limits']['steer_rate_deg_s'])*cfg['dt']
            expected = previous[i]+np.clip(angle-previous[i], -rate_step, rate_step)
            np.testing.assert_allclose(actual[i], [expected, ax], atol=1e-14)
        self.assertTrue(flags[:, 0].any())

    def test_payload_preserves_geometry_and_uses_new_cg_inertia(self):
        cfg = config(); vehicles = fleet(cfg)
        for i in range(0, 9, 3):
            base, loaded, tire = [v['params'] for v in vehicles[i:i+3]]
            m, mp = base['m'], .2*base['m']
            wheelbase = base['l_f']+base['l_r']; d = .15*wheelbase
            shift = mp*d/(m+mp)
            self.assertAlmostEqual(loaded['l_f']+loaded['l_r'], wheelbase)
            self.assertAlmostEqual(loaded['l_f']-base['l_f'], shift)
            self.assertAlmostEqual(loaded['I_z'], base['I_z']+m*shift**2+mp*(d-shift)**2)
            self.assertEqual(tire['m'], base['m'])
            self.assertEqual(tire['pacejka_r_lat'], base['pacejka_r_lat'])
            self.assertAlmostEqual(tire['pacejka_f_lat']['B'], .85*base['pacejka_f_lat']['B'])

    def test_oracle_linearization_matches_small_signal_lateral_dynamics(self):
        cfg = config(); ref = references(cfg)[0]
        p = fleet(cfg)[0]['params']
        a, b = lateral_linearization(p, 20.)
        dx = np.array([1e-5, -2e-5]); du = 1e-6
        x = np.array([[0., 0., 0., 20., *dx]])
        np.testing.assert_allclose(rhs(x, [[du, 0]], p)[0, 4:], a@dx+b*du, atol=1e-10)
        f = oracle_features(fleet(cfg)[0], ref, cfg)
        self.assertTrue(f['open_loop_stable'])
        self.assertTrue(f['baseline_closed_loop_stable'])
        self.assertGreater(f['tau'], 0.)
        self.assertTrue(np.isfinite(f['hy']).all())

    def test_sampled_descriptor_matches_independent_global_coordinate_model(self):
        cfg = config(); p = fleet(cfg)[0]['params']; v = 30.
        a, b = lateral_linearization(p, v)
        # Independently evolve physical [Y,psi,vy,r] and desired Y/psi under
        # held curvature, then subtract to recover the desired-frame tangent.
        ap = np.zeros((4, 4)); ap[0, 1] = v; ap[0, 2] = 1.; ap[1, 3] = 1.
        ap[2:, 2:] = a
        bp = np.array([[0.], [0.], [b[0]], [b[1]]])
        ad, bd, _, _, _ = cont2discrete((ap, bp, np.eye(4), np.zeros((4, 1))), cfg['dt'])
        acl, bc, feedback, direct = baseline_tangent_model(p, v, cfg)
        xp = np.array([.1, -.01, .02, -.03]); desired = np.array([.2, .015])
        z = np.array([desired[0]-xp[0], desired[1]-xp[1], xp[2], xp[3]])
        curvature = .001
        delta = feedback@z+direct*curvature
        xp_next = ad@xp+bd[:, 0]*delta
        dt = cfg['dt']
        yd = desired[0]+v*desired[1]*dt+.5*v*v*curvature*dt*dt
        psid = desired[1]+v*curvature*dt
        independent = np.array([yd-xp_next[0], psid-xp_next[1], *xp_next[2:]])
        np.testing.assert_allclose(acl@z+bc*curvature, independent, atol=1e-14)


class TrackingReferenceAndRolloutTests(unittest.TestCase):
    def test_pose_quadrature_matches_analytic_circle(self):
        t = np.arange(0, 5.001, .05); v = np.full_like(t, 20.); k = np.full_like(t, .002)
        pose = integrate_reference(t, v, k)
        angle = .04*t
        exact = np.column_stack((np.sin(angle)/.002, (1-np.cos(angle))/.002, angle))
        np.testing.assert_allclose(pose, exact, atol=1e-9, rtol=1e-10)

    def test_matched_demand_and_held_out_shape_speed_separation(self):
        cfg = config(); controlled = references(cfg); held = references(cfg, realism=True)
        self.assertEqual(len(controlled), 6)
        self.assertEqual(len(held), 15)
        for r in controlled:
            self.assertAlmostEqual(max(abs(r['ay_ref'])), 2.5)
            np.testing.assert_allclose(r['r'], r['v']*r['kappa'])
            np.testing.assert_array_equal(r['kappa'][r['t'] < 2.], 0.)
            np.testing.assert_array_equal(r['kappa'][r['t'] > 22.], 0.)
        for v in (20, 25, 30):
            for kind in ('single', 'double', 'curve'):
                self.assertIn(f'cfl_{kind}_s{v}', [r['id'] for r in held])
        np.testing.assert_allclose(controlled[0]['ay_ref'], controlled[3]['ay_ref'], atol=1e-14)
        self.assertNotEqual(held[0]['id'], controlled[0]['id'])
        self.assertIn(nearest_context(held[0], controlled, cfg['dt']), [r['id'] for r in controlled])

    def short_case(self):
        cfg = config(); cfg['duration_s'] = 4.
        cfg['references'].update(start_s=.5, end_s=3.5, ramp_s=.4)
        return cfg, fleet(cfg)[0]['params'], references(cfg)[0], candidates(cfg, smoke=True)[:2]

    def test_common_noise_and_batch_independence(self):
        cfg, p, ref, pool = self.short_case()
        noise = measurement_noise(cfg, ref['id'], 1, 80)
        np.testing.assert_array_equal(noise, measurement_noise(cfg, ref['id'], 1, 80))
        self.assertFalse(np.array_equal(noise, measurement_noise(cfg, ref['id'], 2, 80)))
        batch, _ = rollout(p, ref, pool, cfg, 1, 2)
        for c, result in zip(pool, batch):
            single, _ = rollout(p, ref, [c], cfg, 1, 2)
            self.assertEqual(result, single[0])

    def test_failures_are_retained_and_never_shortened_into_good_scores(self):
        cfg, p, ref, pool = self.short_case()
        cfg['feasibility']['peak_beta_deg'] = 1e-9
        rows, _ = resolved_rollout(p, ref, pool, cfg, 1)
        self.assertTrue(all(r['complete'] and not r['feasible'] and r['J'] is not None for r in rows))
        ref['r'][0] = 8.
        rows, _ = rollout(p, ref, pool, cfg, 1, 2)
        self.assertTrue(all(not r['complete'] and not r['feasible'] and r['J'] is None for r in rows))


class TrackingSelectionAndCheckpointTests(unittest.TestCase):
    def synthetic(self):
        rows = []
        for vehicle in ('v0', 'v1'):
            for context in ('c0', 'c1'):
                for role in ('calibration', 'validation'):
                    for candidate, cost in [('g000', 3.), ('g001', 1.)]:
                        rows.append(dict(vehicle=vehicle, context=context, candidate=candidate,
                                         role=role, usable=True, J=cost, rmse_y=.1, rms_rate_deg_s=2.))
        return pd.DataFrame(rows), [{'id': 'v0'}, {'id': 'v1'}], [{'id': 'c0'}, {'id': 'c1'}], [
            {'id': 'g000', 'kind': 'scaling'}, {'id': 'g001', 'kind': 'sobol'}]

    def test_validation_cannot_change_choices_and_failure_is_not_dropped(self):
        frame, vehicles, refs, pool = self.synthetic()
        chosen = select_policies(frame, vehicles, refs, pool)
        self.assertEqual(chosen['common'], 'g001')
        frame.loc[frame.role == 'validation', 'J'] = -10000
        self.assertEqual(chosen, select_policies(frame, vehicles, refs, pool))
        mask = ((frame.vehicle == 'v0') & (frame.context == 'c0') &
                (frame.role == 'calibration') & (frame.candidate == 'g001'))
        frame.loc[mask, 'usable'] = False
        chosen = select_policies(frame, vehicles, refs, pool)
        self.assertEqual(chosen['common'], 'g000')
        self.assertEqual(chosen['vehicle_fixed']['v1'], 'g001')

    def test_empty_near_optimal_sets_are_unavailable(self):
        frame, vehicles, refs, _ = self.synthetic()
        frame.loc[(frame.vehicle == 'v0') & (frame.context == 'c0'), 'usable'] = False
        cells, overlaps = q2(frame, config(), vehicles, refs)
        empty = cells[(cells.vehicle == 'v0') & (cells.context == 'c0')].iloc[0]
        self.assertEqual(empty.near_optimal_candidates, 0)
        self.assertFalse(overlaps.iloc[0].available)
        self.assertTrue(pd.isna(overlaps.iloc[0].jaccard))

    def test_checkpoint_identity_and_missing_collect_guard(self):
        cfg, vehicles, refs, realism, pool, _, data, sig = runner.setup()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            runner.save(out/'run_manifest.json', {'signature': sig})
            state = (cfg, vehicles, refs, realism, pool, out, data, sig)
            with patch.object(runner.subprocess, 'run') as proc:
                with self.assertRaisesRegex(RuntimeError, 'Missing batch'):
                    runner.collect(state)
                proc.assert_not_called()
        batch = dict(vehicle='v', reference='r', role='calibration', seed=1,
                     signature='wrong', rows=[])
        with self.assertRaisesRegex(RuntimeError, 'signature'):
            runner.validate_batch(batch, 'v', 'r', 'calibration', 1, pool, 'expected')
        batch['signature'] = 'expected'
        with self.assertRaisesRegex(RuntimeError, 'candidate pool'):
            runner.validate_batch(batch, 'v', 'r', 'calibration', 1, pool, 'expected')

    def test_immutable_signature_rejects_changed_inputs(self):
        state = list(runner.setup())
        with tempfile.TemporaryDirectory() as tmp:
            state[5] = Path(tmp)
            runner.save(Path(tmp)/'run_manifest.json', {'signature': 'different'})
            with self.assertRaisesRegex(RuntimeError, 'different code/configuration'):
                runner.initialize(tuple(state))


if __name__ == '__main__':
    unittest.main()
