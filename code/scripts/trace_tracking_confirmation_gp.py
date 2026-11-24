"""Read-only A-fit diagnostics on the eight cross-runtime replay-difference jobs.

Run in the original experiment environment with --label original. Every update
uses the recorded earlier observations; no proposed controller is simulated.
Differences are logged explicitly, then the fixed recorded path is followed.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform

import tracking_confirmation as runner
import numpy as np
import pandas as pd
import scipy

from src.tracking_budget import Posterior,feasibility_probability
from src.tracking_confirmation import METHODS
from src.tracking_online_lcb import acquisition_values
from src.tracking_peer_cost import OBSERVATION_FIELDS
from src.tracking_spectral import fresh_state,plan_round,commit_round

SEEDS={2103,2106,4104,4111,4112,4117,4123,4129}
OUT=runner.REPO/'docs/analysis/spectral_confirmation'


def config_text(function):
    stream=io.StringIO()
    with redirect_stdout(stream):function()
    return stream.getvalue()


def trace_job(state,job):
    result=runner.read(runner.run_path(state,job))
    runner.validate_job(state,job,result,complete=True)
    cfg=dict(state['cfg'],methods=[METHODS[0]])
    domain,clients=state['domain'],result['clients']
    actual=fresh_state(domain,cfg,clients)
    recorded=result['state']['methods'][METHODS[0]]
    rows=[]
    for round_index in range(cfg['queries_per_client']):
        before=list(actual['methods'][METHODS[0]]['last_gp_fit'])
        proposed=plan_round(actual,domain,cfg,clients,job['seed'],state['phis'])
        plans,observations=[],[]
        for plan in proposed:
            i=plan['client'];event=recorded['events'][round_index*len(clients)+i]
            if any(plan[k]!=event[k] for k in plan if k not in ('candidate','action')):
                raise RuntimeError('Non-action protocol difference in GP fit trace.')
            if plan['mode']=='feasibility_penalized_LCB':
                data=actual['methods'][METHODS[0]];history=data['histories'][i]
                params=data['gp_parameters'][i]
                posterior=Posterior(domain,history,np.zeros(len(domain.actions)),cfg['gp'],params,False)
                mean,variance=posterior.predict()
                probability=feasibility_probability(history,domain,np.full(len(mean),.5),cfg['gp'])
                values=acquisition_values(mean,variance,probability,cfg['acquisition'])
                training=[(o['action'],o['J']) for o in history if o['eligible']]
                rows.append(dict(**job,method=METHODS[0],round=round_index+1,client=i,context=plan['context'],
                    computed_candidate=plan['candidate'],recorded_candidate=event['candidate'],
                    exact_candidate_match=plan['action']==event['action'],
                    eligible_training_count=len(training),last_fit_eligible_count=data['last_gp_fit'][i],
                    fitted_this_query=before[i]!=data['last_gp_fit'][i],
                    fitted_log_parameters=json.dumps(params),log_likelihood=float(-posterior._loss(np.asarray(params))),
                    computed_score=float(values[plan['action']]),recorded_score=float(values[event['action']]),
                    recorded_minus_computed_score=float(values[event['action']]-values[plan['action']]),
                    recorded_mean_log_J=float(mean[event['action']]),recorded_latent_variance=float(variance[event['action']]),
                    recorded_eligibility_probability=float(probability[event['action']]),
                    training_fingerprint=hashlib.sha256(json.dumps(training).encode()).hexdigest()))
            # Fixed recorded-path diagnostic, not an alternative policy rollout.
            plans.append({k:event[k] for k in plan})
            observations.append({k:event[k] for k in OBSERVATION_FIELDS if k!='action'})
        commit_round(actual,plans,observations,domain,cfg,clients,state['phis'])
    if actual['methods'][METHODS[0]]['events']!=recorded['events']:
        raise RuntimeError('Recorded-path GP trace changed an event.')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',default='original')
    args=parser.parse_args()
    if not args.label.replace('_','').replace('-','').isalnum():
        parser.error('Use a simple letter/number label.')
    state=runner.setup();rows=[]
    for job in state['jobs']:
        if job['seed'] in SEEDS:
            rows.extend(trace_job(state,job))
            print('Traced '+str(job['seed']),flush=True)
    frame=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/('gp_trace_'+args.label+'.csv');frame.to_csv(path,index=False)
    runner.save(OUT/('gp_trace_'+args.label+'_runtime.json'),dict(
        result_signature=state['signature'],programme_seeds=sorted(SEEDS),recorded_path_only=True,
        new_rollouts=0,bo_decisions=len(frame),candidate_differences=int((~frame.exact_candidate_match).sum()),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
            platform=platform.platform()),numpy_configuration=config_text(np.show_config),
        scipy_configuration=config_text(scipy.show_config),source_sha256=runner.digest(Path(__file__))))
    print('Saved '+str(path),flush=True)


if __name__=='__main__':main()
