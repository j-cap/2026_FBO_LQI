"""Public feature construction and unchanged D inference for arbitrary kernels."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from test_tracking_hierarchical import fixture
from src.tracking_kernel_features import basis_bank,pivoted_factor,query_posterior
from src.tracking_hierarchical import cost_posterior
from src.tracking_online import features
from assess_tracking_kernels import accuracy_metrics,selection_table


class PublicBasisTests(unittest.TestCase):
    def test_full_deterministic_bases_recover_public_kernel_and_low_rank_error_is_PSD(self):
        cfg,domain,_,_=fixture();n=len(domain.X)
        specs=[dict(family=f,rank=m) for f in ['spectral','pivoted_cholesky'] for m in [3,n]]
        exact,bank=basis_bank(domain,cfg['gp'],cfg['model'],specs)
        for family in ['spectral','pivoted_cholesky']:
            np.testing.assert_allclose(bank[f'{family}_{n}']['kernel'],exact,atol=1e-10)
            low=bank[f'{family}_3']
            self.assertEqual(low['feature_width'],4)
            self.assertGreaterEqual(np.linalg.eigvalsh(exact-low['kernel']).min(),-1e-10)
        _,again=basis_bank(domain,cfg['gp'],cfg['model'],specs)
        np.testing.assert_array_equal(bank['spectral_3']['F'],again['spectral_3']['F'])
        self.assertEqual(bank['pivoted_cholesky_3']['pivots'],again['pivoted_cholesky_3']['pivots'])

    def test_original_random_features_are_preserved(self):
        cfg,domain,_,_=fixture();m=cfg['model']['features']
        _,bank=basis_bank(domain,cfg['gp'],cfg['model'],[dict(family='rff',rank=m)])
        scale=np.r_[np.sqrt(cfg['gp']['constant_variance']),np.full(m,cfg['gp']['amplitude'])]
        np.testing.assert_array_equal(bank[f'rff_{m}']['F'],features(domain,cfg['model'])*scale)

    def test_query_space_and_feature_message_posteriors_agree(self):
        cfg,domain,_,_=fixture();n=len(domain.X)
        _,bank=basis_bank(domain,cfg['gp'],cfg['model'],[dict(family='spectral',rank=n)])
        F=bank[f'spectral_{n}']['F'];scale=np.r_[np.sqrt(cfg['gp']['constant_variance']),np.full(n,cfg['gp']['amplitude'])]
        histories=[[dict(action=a,eligible=True,J=j) for a,j in zip(actions,values)] for actions,values in
            [([0,1],[.5,1.2]),([0,2],[1.1,.6]),([],[])]]
        histories[0].append(dict(action=3,eligible=False,J=None))
        clients=np.array([2,0,1,0,2]);actions=np.array([4,3,2,0,1])
        dual=query_posterior(histories,F@F.T,cfg['hierarchical'],cfg['gp'],clients,actions)
        message=cost_posterior(histories,F/scale,cfg['gp'],cfg['hierarchical'])
        np.testing.assert_allclose(dual['mean'],np.asarray(message['mean'])[clients,actions],atol=1e-9)
        np.testing.assert_allclose(dual['variance'],np.asarray(message['variance'])[clients,actions],atol=1e-9)
        np.testing.assert_allclose(dual['sharing_weights'],message['sharing_weights'],atol=1e-9)
        empty=query_posterior([[],[],[]],F@F.T,cfg['hierarchical'],cfg['gp'],clients,actions)
        np.testing.assert_array_equal(empty['mean'],0.)
        np.testing.assert_allclose(empty['variance'],np.diag(F@F.T)[actions],atol=1e-10)

    def test_pivoted_factor_handles_low_rank_and_rejects_invalid_rank(self):
        a=np.array([[1.,2.],[2.,4.]])
        F,pivots=pivoted_factor(a,2)
        self.assertEqual(len(pivots),1)
        np.testing.assert_allclose(F@F.T,a,atol=1e-12)
        with self.assertRaises(ValueError):pivoted_factor(a,3)


class ReferenceMetricTests(unittest.TestCase):
    def test_exact_score_regret_checks_all_numerical_ties_and_zero_error(self):
        acquisition=dict(confidence_multiplier=2.,feasibility_penalty=1.,probability_floor=1e-6)
        exact_mean=np.array([0.,1.]);std=np.zeros(2);p=np.ones(2)
        actual,_,_=accuracy_metrics(exact_mean,std,exact_mean,std,p,acquisition)
        self.assertEqual(actual['exact_score_regret'],0.)
        self.assertEqual(actual['mean_MSE'],0.)
        tied,_,_=accuracy_metrics(np.zeros(2),std,exact_mean,std,p,acquisition)
        self.assertEqual(tied['exact_score_regret'],1.)
        self.assertTrue(tied['top_one_agreement'])

    def test_selection_requires_worst_context_and_worst_history_group_to_pass(self):
        prior=pd.DataFrame([dict(representation='small',feature_width=65,relative_kernel_Frobenius_error=.01,max_relative_diagonal_error=.01),
                            dict(representation='large',feature_width=257,relative_kernel_Frobenius_error=.001,max_relative_diagonal_error=.001)])
        overall=pd.DataFrame([dict(representation=n,mean_RMSE=.01) for n in ['small','large']])
        groups=pd.DataFrame([dict(representation=n,mean_exact_score_regret=.01,p95_exact_score_regret=v)
                             for n,v in [('small',.2),('large',.03)]])
        contexts=pd.DataFrame([dict(representation=n,mean_RMSE=e,std_RMSE=.01)
                               for n,e in [('small',.01),('small',.2),('large',.02)]])
        targets=dict(relative_kernel_Frobenius_error=.02,max_relative_diagonal_error=.05,
            worst_context_mean_RMSE=.05,worst_context_std_RMSE=.05,mean_exact_score_regret=.05,p95_exact_score_regret=.1)
        result=selection_table(prior,overall,groups,contexts,targets).set_index('representation')
        self.assertFalse(result.loc['small','all_accuracy_targets_pass'])
        self.assertTrue(result.loc['large','all_accuracy_targets_pass'])


if __name__=='__main__':unittest.main()
