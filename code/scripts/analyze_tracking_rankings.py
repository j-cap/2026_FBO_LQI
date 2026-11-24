"""Compare frozen A/D rankings on their saved histories at one difficult context.

Both fits receive the same source history in each comparison. Only the source's
recorded observations advance that history; alternative rankings never reveal
or insert outcomes. Dense validation labels are joined after all model fits.
"""
from pathlib import Path
import json

import analyze_tracking_hierarchical as audit
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.special import logsumexp
from src.tracking_budget import Posterior, feasibility_probability,matern
from src.tracking_hierarchical import SHARED, cost_posterior
from src.tracking_online import features, seed_for
from src.tracking_online_lcb import acquisition_values
from src.tracking_peer_cost import OBSERVATION_FIELDS

runner = audit.runner
OUT = runner.REPO/'docs/analysis/low_frequency_rankings'
CONTEXT = 's30_f0.1'
ARMS = {'A': 'peer_gp_bo', 'D': SHARED}
EXACT = 'exact_kernel_reference'


def exact_kernel_posterior(histories,kernel,cfg,gp):
    """Evaluator-only dense reference for D's intended kernel, no hyper tuning.

    It keeps D's task covariance, sharing grid/prior and noise, replacing only
    the finite random-feature kernel by the exact public Matern kernel.
    """
    observations=[(i,o['action'],np.log(o['J'])) for i,h in enumerate(histories) for o in h if o['eligible']]
    tasks=np.asarray([i for i,_,_ in observations])
    actions=[a for _,a,_ in observations]
    y=np.asarray([v for _,_,v in observations])
    means,variances,evidence=[],[],[]
    for rho in cfg['sharing_fractions']:
        if not len(y):
            means.append(np.zeros((len(histories),len(kernel))))
            variances.append(np.tile(np.diag(kernel),(len(histories),1)));evidence.append(0.)
            continue
        K=kernel[np.ix_(actions,actions)]*(rho+(1-rho)*(tasks[:,None] == tasks))
        K.flat[::len(K)+1] += gp['noise_std_log_J']**2+1e-9
        factor=cho_factor(K,lower=True,check_finite=False)
        alpha=cho_solve(factor,y,check_finite=False)
        evidence.append(float(-.5*(y@alpha+2*np.log(np.diag(factor[0])).sum()+len(y)*np.log(2*np.pi))))
        component_means,component_variances=[],[]
        for i in range(len(histories)):
            cross=kernel[:,actions]*(rho+(1-rho)*(tasks == i))
            component_means.append(cross@alpha)
            component_variances.append(np.maximum(0.,np.diag(kernel)-np.einsum('ij,ji->i',cross,
                cho_solve(factor,cross.T,check_finite=False))))
        means.append(component_means);variances.append(component_variances)
    log_weight=np.log(np.asarray(cfg['prior_weights'])/np.sum(cfg['prior_weights']))+evidence
    weights=np.exp(log_weight-logsumexp(log_weight))
    means,variances=np.asarray(means),np.asarray(variances)
    mean=np.einsum('k,kia->ia',weights,means)
    variance=np.einsum('k,kia->ia',weights,variances+(means-mean[None])**2)
    return dict(mean=mean,variance=variance,sharing_weights=weights,
                expected_sharing_fraction=float(weights@cfg['sharing_fractions']))


def ranked_terms(mean, variance, probability, allowed, cfg):
    """Score-only calculation: no outcome table or oracle is accepted here."""
    mean = np.asarray(mean, float)
    std = np.sqrt(np.maximum(variance, 0.))
    exploration = -cfg['confidence_multiplier']*std
    penalty = -cfg['feasibility_penalty']*np.log(np.maximum(probability, cfg['probability_floor']))
    score = acquisition_values(mean, variance, probability, cfg)
    np.testing.assert_allclose(score, mean+exploration+penalty, atol=1e-12)
    indices = np.flatnonzero(allowed)
    if not len(indices):
        raise RuntimeError('No untried gain at the selected context.')
    best = score[indices].min()
    tolerance = 1e-12*max(1., abs(best))
    ties = indices[score[indices] <= best+tolerance]
    ranks = {}
    for name, values in [('score',score), ('mean',mean), ('penalty',penalty)]:
        rank = np.full(len(mean), np.nan)
        for i in indices:
            rank[i] = 1+np.sum(values[indices] < values[i]-tolerance)
        ranks[name] = rank
    return dict(mean=mean, std=std, probability=np.asarray(probability),
        exploration=exploration, penalty=penalty, score=score, ties=ties, ranks=ranks)


def fit_source(state, job, source):
    """Replay the forced source schedule; fit original A and D on its prefixes."""
    cfg, domain, clients = state['cfg'], state['domain'], job['clients']
    events = job['state']['methods'][ARMS[source]]['events']
    histories = [[] for _ in clients]
    initial = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']])
    params = [initial.copy() for _ in clients]
    last_fit = [0]*len(clients)
    phi = features(domain, cfg['model'])
    kernel=matern(domain.X,domain.X,cfg['gp']['length_scales'],cfg['gp']['amplitude'],cfg['gp']['constant_variance'])
    cases, candidates, parameter_rows = [], [], []
    factual_choices_checked = 0
    for round_id in range(1,cfg['queries_per_client']+1):
        current = events[(round_id-1)*len(clients):round_id*len(clients)]
        joint = cost_posterior(histories,phi,cfg['gp'],cfg['hierarchical']) if round_id > 1 else None
        dense=(exact_kernel_posterior(histories,kernel,cfg['hierarchical'],cfg['gp'])
            if source == 'D' and any(e['context'] == CONTEXT and e['mode'] == 'feasibility_penalized_LCB' for e in current)
            else None)
        for event in current:
            if event['mode'] != 'feasibility_penalized_LCB':
                continue
            i, history = event['client'], histories[event['client']]
            count = sum(o['eligible'] for o in history)
            optimize = count >= 4 and count-last_fit[i] >= cfg['gp']['optimize_every']
            gp = Posterior(domain,history,np.zeros(len(domain.actions)),cfg['gp'],params[i],optimize)
            params[i] = gp.params.copy()
            if optimize:
                last_fit[i] = count
            mean_A, variance_A = gp.predict()
            probability = feasibility_probability(history,domain,np.full(len(domain.actions),.5),cfg['gp'])
            allowed = np.asarray([c == event['context'] for c,_ in domain.actions])
            allowed[[o['action'] for o in history]] = False
            fits = dict(A=ranked_terms(mean_A,variance_A,probability,allowed,cfg['acquisition']),
                D=ranked_terms(np.asarray(joint['mean'][i]),np.asarray(joint['variance'][i]),
                               probability,allowed,cfg['acquisition']))
            rng = np.random.default_rng(seed_for('acquisition',job['seed'],clients[i]['id'],round_id-1))
            actual = int(rng.choice(fits[source]['ties']))
            if actual != event['action']:
                raise RuntimeError('Factual '+source+' choice differs from replay.')
            factual_choices_checked += 1
            if event['context'] != CONTEXT:
                continue
            if dense is not None:
                fits[EXACT]=ranked_terms(dense['mean'][i],dense['variance'][i],probability,allowed,cfg['acquisition'])
            reference_candidate = state['reference_candidates'][clients[i]['plant_id'],CONTEXT]
            reference = domain.action(CONTEXT,reference_candidate)
            base = dict(seed=job['seed'],history_source=source,round=round_id,
                client=clients[i]['id'],plant_id=clients[i]['plant_id'],context=CONTEXT,
                actual_candidate=event['candidate'],actual_eligible=event['eligible'],
                actual_J=event['J'],reference_candidate=reference_candidate,
                reference_available=bool(allowed[reference]),available_candidates=int(allowed.sum()),
                prior_local_cost_labels=count,
                prior_fleet_cost_labels=sum(o['eligible'] for h in histories for o in h),
                prior_peer_failures_at_actual=sum(o['action'] == actual and not o['eligible']
                    for j,h in enumerate(histories) if j != i for o in h),
                prior_peer_eligible_at_actual=sum(o['action'] == actual and o['eligible']
                    for j,h in enumerate(histories) if j != i for o in h),
                sharing_fraction=joint['expected_sharing_fraction'])
            parameter_rows.append(dict(**base,optimized_A_kernel=optimize,
                **dict(zip(['length_Ky','length_Kpsi','length_Kr','length_speed','length_log_frequency','amplitude'],np.exp(params[i])))))
            for model, fit in fits.items():
                row = dict(**base,model=model,factual=model == source,
                    model_expected_sharing_fraction=(dense['expected_sharing_fraction'] if model == EXACT
                        else joint['expected_sharing_fraction'] if model == 'D' else None),
                    reference_score_rank=fit['ranks']['score'][reference],
                    reference_mean_rank=fit['ranks']['mean'][reference],
                    reference_eligibility_penalty_rank=fit['ranks']['penalty'][reference],
                    actual_score_rank=fit['ranks']['score'][actual],
                    reference_mean=fit['mean'][reference],reference_std=fit['std'][reference],
                    reference_probability=fit['probability'][reference],
                    actual_mean=fit['mean'][actual],actual_std=fit['std'][actual],
                    actual_probability=fit['probability'][actual])
                for term in ('mean','exploration','penalty','score'):
                    row['reference_minus_actual_'+term] = fit[term][reference]-fit[term][actual]
                np.testing.assert_allclose(row['reference_minus_actual_score'],sum(
                    row['reference_minus_actual_'+term] for term in ('mean','exploration','penalty')),atol=1e-12)
                best = fit['ties'][0]
                row['lowest_score_candidate'] = domain.actions[best][1]
                row['lowest_score_ties'] = len(fit['ties'])
                cases.append(row)
                for action in np.flatnonzero(np.asarray([c == CONTEXT for c,_ in domain.actions])):
                    candidates.append(dict(seed=job['seed'],history_source=source,round=round_id,
                        client=clients[i]['id'],plant_id=clients[i]['plant_id'],model=model,
                        context=CONTEXT,candidate=domain.actions[action][1],allowed=bool(allowed[action]),
                        is_reference=action == reference,is_actual=action == actual,
                        **{term:float(fit[term][action]) for term in
                           ('mean','std','probability','exploration','penalty','score')},
                        score_rank=fit['ranks']['score'][action],mean_rank=fit['ranks']['mean'][action]))
        # Neither model's alternative ranking changes these paid source outcomes.
        for event in current:
            histories[event['client']].append({k:event[k] for k in OBSERVATION_FIELDS})
    if source == 'A':
        data = job['state']['methods'][ARMS[source]]
        # Finite-difference optimizer refits can differ slightly across BLAS /
        # SciPy platforms. Every factual choice is checked exactly above.
        difference=float(np.max(np.abs(np.asarray(params)-np.asarray(data['gp_parameters']))))
        if difference > 1e-4:
            raise RuntimeError('A refitted log-kernel parameters differ materially.')
        for row in parameter_rows:row['final_A_log_kernel_max_abs_difference']=difference
        if last_fit != data['last_gp_fit']:
            raise RuntimeError('A hyperparameter-update schedule differs.')
    return pd.DataFrame(cases),pd.DataFrame(candidates),pd.DataFrame(parameter_rows),factual_choices_checked


def attach_validation(candidates,tables):
    """Evaluator-only join, after model fits and all score/rank calculations."""
    reference, validation = tables
    out = candidates.copy()
    eligible, costs, gaps, success = [],[],[],[]
    for row in out.itertuples():
        value = validation[row.plant_id,row.context,row.candidate]
        gap = value['J']/reference[row.plant_id,row.context]['J']-1 if value['eligible'] else None
        eligible.append(value['eligible']);costs.append(value['J'] if value['eligible'] else None)
        gaps.append(gap);success.append(gap is not None and gap <= .05)
    out['validation_eligible'],out['validation_J'] = eligible,costs
    out['validation_gap'],out['validation_near_reference'] = gaps,success
    return out


def summaries(cases,candidates):
    rows=[]
    for (source,model),group in cases.groupby(['history_source','model'],sort=False):
        available=group[group.reference_available]
        rows.append(dict(history_source=source,model=model,focus_fallbacks=len(group),
            reference_available=len(available),reference_rank_one=int((available.reference_score_rank == 1).sum()),
            reference_in_top_five=int((available.reference_score_rank <= 5).sum()),
            median_reference_score_rank=float(available.reference_score_rank.median()),
            median_reference_mean_rank=float(available.reference_mean_rank.median()),
            mean_reference_minus_actual_mean=float(available.reference_minus_actual_mean.mean()),
            mean_reference_minus_actual_exploration=float(available.reference_minus_actual_exploration.mean()),
            mean_reference_minus_actual_penalty=float(available.reference_minus_actual_penalty.mean()),
            mean_reference_minus_actual_score=float(available.reference_minus_actual_score.mean())))
    ranked = candidates[candidates.allowed & candidates.score_rank.eq(1)]
    top_summary=ranked.groupby(['history_source','model'],sort=False).agg(
        top_rank_rows=('candidate','size'),validation_eligible_top_ranks=('validation_eligible','sum'),
        near_reference_top_ranks=('validation_near_reference','sum')).reset_index()
    return pd.DataFrame(rows).merge(top_summary,on=['history_source','model'],validate='one_to_one')


def discovery_table(results,cases):
    rows=[]
    for job in results:
        if job['condition'] != 'homogeneous':continue
        for source,method in ARMS.items():
            data=job['state']['methods'][method]
            context=[e for e in data['events'] if e['context'] == CONTEXT]
            fallback=[e for e in context if e['mode'] == 'feasibility_penalized_LCB']
            discoveries=[e for e in context if e['candidate'] == 'g012' and e['mode'] == 'feasibility_penalized_LCB']
            reference=[e for e in context if e['candidate'] == 'g012']
            factual=cases[(cases.seed == job['seed']) & (cases.history_source == source) & cases.factual]
            rows.append(dict(seed=job['seed'],source=source,focus_queries=len(context),
                BO_queries=len(fallback),BO_eligible=sum(e['eligible'] for e in fallback),
                distinct_BO_candidates=len(set(e['candidate'] for e in fallback)),
                unique_failed_BO_candidates=len(set(e['candidate'] for e in fallback if not e['eligible'])),
                BO_with_prior_peer_failure=int((factual.prior_peer_failures_at_actual > 0).sum()),
                reference_BO_discoveries=len(discoveries),
                first_reference_discovery_round=min(e['round'] for e in discoveries) if discoveries else None,
                reference_paid_tests=len(reference),reference_eligible_tests=sum(e['eligible'] for e in reference),
                reference_peer_verifications=sum(e['mode'] == 'peer_gain_verification' for e in reference)))
    return pd.DataFrame(rows)


def figure(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='fedbo-low-frequency-rankings'
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    colors={'A':'#a35b39','D':'#248363',EXACT:'#3478b2'}
    for (source,model),g in cases[cases.reference_available].groupby(['history_source','model'],sort=False):
        ranks=np.sort(g.reference_score_rank.to_numpy())
        axes[0].step(ranks,np.arange(1,len(ranks)+1)/len(ranks),where='post',color=colors[model],
            linestyle='-' if source == 'D' else '--',
            label=f'{"Exact kernel" if model == EXACT else model} on {source} history (n={len(ranks)})')
    axes[0].set(xlabel='Score rank of the only near-reference gain g012',ylabel='Fraction of recorded decisions',ylim=(0,1.02))
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    g=cases[(cases.history_source == 'D') & cases.reference_available]
    term_colors=['#a35b39','#3478b2','#c59a30']
    total=[]
    for x,model in enumerate(['A','D',EXACT]):
        means=g[g.model == model][['reference_minus_actual_'+t for t in ('mean','exploration','penalty')]].mean()
        positive,negative=0.,0.
        for term,color,value in zip(['Predicted cost','Exploration','Eligibility'],term_colors,means):
            bottom=positive if value >= 0 else negative
            axes[1].bar(x,value,bottom=bottom,color=color,label=term if x == 0 else None)
            if value >= 0:positive+=value
            else:negative+=value
        total.append(means.sum())
    axes[1].scatter([0,1,2],total,color='#17232c',s=30,label='Total score margin',zorder=4)
    axes[1].axhline(0,color='#777777',linewidth=.8)
    axes[1].set(xticks=[0,1,2],xticklabels=['A on D history','D on D history','Exact kernel\non D history'],
        ylabel="Mean score margin: g012 minus D's actual gain",
        title='Positive margin rejects g012 relative to D\'s chosen gain')
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    vector=OUT/'rankings_and_score_terms.svg'
    fig.savefig(vector,bbox_inches='tight',metadata={'Date':None})
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
    preview=runner.ROOT/'figures/low_frequency_rankings.png'
    preview.parent.mkdir(parents=True,exist_ok=True);fig.savefig(preview,dpi=140,bbox_inches='tight');plt.close(fig)


def main():
    state,results,tables,*_=audit.verify()
    state=dict(state,reference_candidates={key:value['candidate'] for key,value in tables[0].items()})
    case_parts,candidate_parts,parameter_parts=[],[],[]
    checks=0
    for job in results:
        if job['condition'] != 'homogeneous':continue
        for source in ARMS:
            cases,candidates,parameters,count=fit_source(state,job,source)
            case_parts.append(cases);candidate_parts.append(candidates);parameter_parts.append(parameters);checks+=count
            print(f'Ranked {source} and D/A fits on {source} history, seed {job["seed"]}',flush=True)
    cases=pd.concat(case_parts,ignore_index=True)
    # Evaluation labels were not available to fit_source or ranked_terms.
    candidates=attach_validation(pd.concat(candidate_parts,ignore_index=True),tables)
    summary=summaries(cases,candidates)
    discovery=discovery_table(results,cases)
    example=cases[(cases.history_source == 'D') & cases.reference_available].pivot(
        index=['seed','round','client'],columns='model',values='reference_score_rank').reset_index()
    example['rank_gap']=example.D-example.A
    chosen=example.sort_values(['rank_gap','seed','round','client'],ascending=[False,True,True,True]).iloc[0]
    selected=candidates[(candidates.history_source == 'D') & (candidates.seed == chosen.seed) &
        (candidates['round'] == chosen['round']) & (candidates.client == chosen.client)]
    audit_candidates=candidates[(candidates.score_rank <= 5) | candidates.is_reference | candidates.is_actual]
    reference_rows=candidates[(candidates.seed == candidates.seed.iloc[0]) &
        (candidates.history_source == 'A') & (candidates.model == 'A')]
    reference_rows=reference_rows[reference_rows.client == reference_rows.client.iloc[0]]
    reference_rows=reference_rows[reference_rows['round'] == reference_rows['round'].iloc[0]]
    landscape=reference_rows[['candidate','validation_eligible','validation_J','validation_gap','validation_near_reference']].copy()
    if set(landscape[landscape.validation_near_reference].candidate) != {'g012'}:
        raise RuntimeError('The declared context no longer has a unique near-reference gain.')
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in [('decision_rankings',cases),('ranking_summary',summary),('candidate_terms',audit_candidates),
        ('discovery_and_repetition',discovery),('A_kernel_parameters',pd.concat(parameter_parts,ignore_index=True)),
        ('example_full_rankings',selected),('validation_landscape',landscape)]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    figure(cases)
    paths=[runner.path_for(state,j) for j in runner.jobs(state)]
    paths += [runner.path_for(state['reference_state'],j) for j in runner.jobs(state)]
    paths += [state['archive']/'controlled_evaluations.csv',state['out']/'run_manifest.json']
    runner.save(OUT/'verification.json',dict(result_signature=state['signature'],context=CONTEXT,
        condition='homogeneous',seeds=[1100,1101,1102],history_sources=list(ARMS),
        factual_BO_choices_reproduced=checks,
        focus_decisions=len(cases[['seed','history_source','round','client']].drop_duplicates()),
        new_physical_rollouts=0,full_policy_variants_run=0,alternative_outcomes_fed_back=0,
        validation_labels_joined_after_scoring=True,factual_A_D_definitions_unchanged=True,
        maximum_A_archived_refit_log_kernel_difference=float(pd.concat(parameter_parts,ignore_index=True)
            .final_A_log_kernel_max_abs_difference.max()),
        diagnostic_control='Exact public Matern kernel for the same D task covariance, sharing prior, noise and D histories; no hyperparameter tuning.',
        diagnostic_scope='A/D fitted on identical forced source histories; alternative score rankings, not alternative policy outcomes.',
        analysis_source_sha256=runner.digest(Path(__file__)),
        input_hashes={p.relative_to(runner.REPO).as_posix():runner.digest(p) for p in paths}))
    print('Ranking summary:');print(summary.to_string(index=False))
    print('Discovery and repeated failures:');print(discovery.to_string(index=False))
    print('Example:');print(chosen.to_string())
    print('Analysis: '+str(OUT),flush=True)


if __name__ == '__main__':
    main()
