"""Declared 256/A primary comparison, with 384 as a fixed accuracy control."""
import base64
import io
import json

import pandas as pd

from .tracking_hierarchical import SHARED,LOCAL
from .tracking_online_lcb_report import summarize_times
from .tracking_peer_cost_report import mechanism_tables,summarize
from .tracking_spectral import LEGACY,shared,local


def contrasts(cfg):
    main=cfg['primary_rank'];other=next(r for r in cfg['spectral_ranks'] if r!=main)
    return [(shared(main),LEGACY[0],'primary'),(shared(main),LEGACY[1],'secondary'),
        (shared(other),LEGACY[0],'accuracy_control'),(shared(other),LEGACY[1],'accuracy_control'),
        *[(shared(r),local(r),'sharing_control') for r in cfg['spectral_ranks']],
        (shared(other),shared(main),'representation_size'),
        *[(shared(r),SHARED,'original_D_control') for r in cfg['spectral_ranks']],
        *[(local(r),LOCAL,'original_E_control') for r in cfg['spectral_ranks']]]


def paired_comparisons(trials,cfg,per_seed=False):
    rows=[];keys=['condition','fleet_budget']+(['seed'] if per_seed else [])
    for values,group in trials.groupby(keys,sort=False):
        common=dict(zip(keys,values))
        for method,baseline,role in contrasts(cfg):
            a=group[group.method==method].set_index(['seed','client']).sort_index()
            b=group[group.method==baseline].set_index(['seed','client']).sort_index()
            if a.empty or not a.index.equals(b.index):raise RuntimeError('Spectral comparison is not paired.')
            good=a.fully_feasible & b.fully_feasible
            ma=float(a.loc[good,'J'].mean()) if good.any() else None
            mb=float(b.loc[good,'J'].mean()) if good.any() else None
            rows.append(dict(**common,method=method,baseline=baseline,role=role,
                matched_client_trials=len(a),jointly_feasible_pairs=int(good.sum()),
                eligible_policies_method=int(a.fully_feasible.sum()),eligible_policies_baseline=int(b.fully_feasible.sum()),
                context_success_delta=float((a.context_success-b.context_success).mean()),
                eligible_policy_rate_delta=float((a.fully_feasible.astype(int)-b.fully_feasible.astype(int)).mean()),
                mean_J_method_jointly_feasible=ma,mean_J_baseline_jointly_feasible=mb,
                mean_J_delta_jointly_feasible=ma-mb if good.any() else None,
                relative_reduction_of_joint_mean_J=1-ma/mb if good.any() else None,
                mean_ineligible_tests_per_client_delta=float((a.ineligible_queries-b.ineligible_queries).mean())))
    return pd.DataFrame(rows)


def benchmark_readout(paired,seeds,final_budget):
    rows=[]
    final=paired[(paired.fleet_budget==final_budget)&paired.role.isin(['primary','secondary','accuracy_control'])]
    for _,row in final.iterrows():
        group=seeds[(seeds.fleet_budget==final_budget)&(seeds.condition==row.condition)&
            (seeds.method==row.method)&(seeds.baseline==row.baseline)]
        primary=bool(row.context_success_delta>1e-12)
        eligible=bool(row.eligible_policies_method>=row.eligible_policies_baseline)
        cost=bool(pd.notna(row.mean_J_delta_jointly_feasible) and row.mean_J_delta_jointly_feasible<0)
        rows.append(dict(condition=row.condition,method=row.method,baseline=row.baseline,role=row.role,
            context_success_delta=row.context_success_delta,eligible_policies_method=int(row.eligible_policies_method),
            eligible_policies_baseline=int(row.eligible_policies_baseline),jointly_feasible_pairs=int(row.jointly_feasible_pairs),
            relative_reduction_of_joint_mean_J=row.relative_reduction_of_joint_mean_J,
            primary_context_metric_improved=primary,complete_eligibility_not_reduced=eligible,
            conditional_cost_improved=cost,seeds_with_positive_primary=int((group.context_success_delta>1e-12).sum()),
            seeds_with_negative_primary=int((group.context_success_delta< -1e-12).sum()),
            seeds_with_equal_primary=int((group.context_success_delta.abs()<=1e-12).sum()),
            all_three_aggregate_metrics_support_improvement=primary and eligible and cost))
    return pd.DataFrame(rows)


def labels(cfg):
    out={LEGACY[0]:'A: Peer gains + local GP',LEGACY[1]:'B: Local cost mean',SHARED:'D0: Shared/local RFF64',LOCAL:'E0: Local RFF64'}
    for rank in cfg['spectral_ranks']:
        out[shared(rank)]=f'D{rank}: Shared/local spectral';out[local(rank)]=f'E{rank}: Local spectral'
    return out


def write_report(out,trials,contexts,queries,learning,times,cfg,mode,domain_size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trials=trials.copy()
    trial_basis={m:0 for m in cfg['comparison_methods']}
    nclients=int(trials.client.nunique())
    for rank in cfg['spectral_ranks']:
        for m in (shared(rank),local(rank)):trial_basis[m]=nclients*domain_size*(rank+1)*8
    trials['public_basis_broadcast_estimate_bytes']=trials.method.map(trial_basis)
    trials['payload_with_public_basis_estimate_bytes']=trials.payload_bytes+trials.public_basis_broadcast_estimate_bytes
    summary=summarize(trials)
    basis_summary=trials.groupby(['condition','method','fleet_budget'],sort=False)[
        ['public_basis_broadcast_estimate_bytes','payload_with_public_basis_estimate_bytes']].mean().reset_index()
    summary=summary.merge(basis_summary,on=['condition','method','fleet_budget'],validate='one_to_one')
    paired,seeds=paired_comparisons(trials,cfg),paired_comparisons(trials,cfg,True)
    final_budget=cfg['fleet_budgets'][-1]
    final=paired[paired.fleet_budget==final_budget]
    primary=final[final.role=='primary'];secondary=final[final.role=='secondary']
    primary_seed=seeds[(seeds.fleet_budget==final_budget)&(seeds.role=='primary')]
    control=final[final.role=='accuracy_control']
    readout=benchmark_readout(paired,seeds,final_budget)
    timing=summarize_times(times);modes,peers,transfers=mechanism_tables(queries)
    family=trials[trials.fleet_budget==final_budget].groupby(['condition','family','method'],sort=False).agg(
        client_trials=('client','size'),eligible_policies=('fully_feasible','sum'),context_success_rate=('context_success','mean'),
        conditional_mean_policy_J=('J','mean')).reset_index()
    by_context=contexts[contexts.fleet_budget==final_budget].groupby(['condition','method','context'],sort=False).agg(
        client_context_trials=('client','size'),eligible_contexts=('eligible','sum'),successes=('success','sum'),
        context_success_rate=('success','mean'),conditional_mean_context_J=('J','mean')).reset_index()
    sharing=[]
    for method in [SHARED,*[shared(r) for r in cfg['spectral_ranks']]]:
        for _,row in learning[learning.method==method].iterrows():
            for fraction,weight,evidence in zip(cfg['hierarchical']['sharing_fractions'],row.sharing_weights,row.component_log_evidence):
                sharing.append(dict(condition=row.condition,seed=row.seed,method=method,round=row['round'],
                    used_by_later_query=row['round']<cfg['queries_per_client'],fraction=fraction,weight=weight,
                    log_evidence=evidence,expected_sharing_fraction=row.expected_sharing_fraction))
    sharing=pd.DataFrame(sharing)
    for name,frame in [('online_summary',summary),('paired_comparisons',paired),('per_seed_comparisons',seeds),
        ('primary_comparison',primary),('secondary_comparison',secondary),('primary_per_seed',primary_seed),
        ('accuracy_control_comparison',control),('benchmark_readout',readout),('eligibility_timing_summary',timing),
        ('query_modes',modes),('peer_verifications',peers),('bo_origin_transfers',transfers),
        ('final_family_outcomes',family),('final_context_outcomes',by_context),('sharing_strength',sharing)]:
        frame.to_csv(out/(name+'.csv'),index=False)
    title_labels=labels(cfg)
    main_methods=[LEGACY[0],LEGACY[1],shared(cfg['primary_rank']),shared(next(r for r in cfg['spectral_ranks'] if r!=cfg['primary_rank']))]
    fig,axes=plt.subplots(len(cfg['conditions']),3,figsize=(15,7.5),squeeze=False,constrained_layout=True)
    for row,condition in enumerate(cfg['conditions']):
        for method,color in zip(main_methods,['#a35b39','#3376b0','#248363','#8c61a8']):
            g=summary[(summary.condition==condition)&(summary.method==method)]
            axes[row,0].plot(g.fleet_budget,100*g.eligible_context_coverage,marker='o',color=color,label=title_labels[method])
            axes[row,1].plot(g.fleet_budget,100*g.context_success_rate,marker='o',color=color)
            axes[row,2].plot(g.fleet_budget,g.mean_fleet_ineligible_tests,marker='o',color=color)
        axes[row,0].set(title=condition,ylabel='Contexts with a locally eligible gain [%]',ylim=(-2,102))
        axes[row,1].set(ylabel='Validation contexts within 5% [%]',ylim=(-2,102))
        axes[row,2].set(ylabel='Mean ineligible fleet tests')
        for ax in axes[row]:
            ax.set(xlabel='Fleet calibration tests (log scale)',xscale='log',xticks=cfg['fleet_budgets'])
            ax.set_xticklabels([str(b) for b in cfg['fleet_budgets']]);ax.minorticks_off();ax.grid(alpha=.2)
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,1.09),ncol=2,fontsize=9)
    fig.savefig(out/'spectral_curves.png',dpi=150,bbox_inches='tight')
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=110,bbox_inches='tight');plt.close(fig)
    encoded=base64.b64encode(stream.getvalue()).decode()
    def records(frame):return frame.astype(object).where(pd.notna(frame),None).to_dict('records')
    (out/'summary.json').write_text(json.dumps(dict(mode=mode,development_pilot=True,primary_rank=cfg['primary_rank'],
        primary_baseline='A',secondary_baseline='B',summary=records(summary),benchmarks=records(readout)),
        indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def table(frame):return '<div class="table">'+frame.to_html(index=False,float_format=lambda x:f'{x:.4g}',na_rep='unavailable')+'</div>'
    title='SMOKE: software verification only' if mode=='smoke' else 'Matched online spectral BO: D256 first, D384 accuracy control'
    html='''<!doctype html><html><head><meta charset="utf-8"><title>Spectral online BO</title>
<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#192736}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd5df;padding:6px}
.table{overflow-x:auto}img{max-width:100%}.note{background:#edf3f7;padding:16px}</style></head><body>'''
    html+='<h1>'+title+'</h1><p class="note">D256 is the declared main candidate; D384 is the fixed accuracy '
    html+='control. Each has a matched local E model. Every method pays for one factory test, peer '
    html+='verification, BO and failures within the same local cap. All arms own their observations '
    html+='and catalogues. Public bases contain no cost labels and are pinned before the first test.</p>'
    html+='<h2>Primary: D256 minus A at the final budget</h2>'+table(primary)
    html+='<h2>Primary by seed</h2>'+table(primary_seed)
    html+='<h2>Secondary: D256 minus B</h2>'+table(secondary)
    html+='<h2>Accuracy control: D384 versus A/B</h2>'+table(control)
    html+='<p>Do not promote D384 to the primary candidate after seeing outcomes. Positive context '
    html+='success deltas favor the method. Retain complete-policy counts; J comparisons use jointly '
    html+='eligible policies only. Missing/ineligible contexts remain unsuccessful. These are existing '
    html+='programme/noise seeds on fixed plants, not independent fleets.</p>'
    html+='<h2>Benchmark readout</h2>'+table(readout)
    html+='<img alt="Main candidate and accuracy control against A and B" src="data:image/png;base64,'+encoded+'">'
    html+='<h2>Sharing and representation controls</h2>'+table(final[~final.role.isin(['primary','secondary','accuracy_control'])])
    html+='<p>D/E at each size isolates the cost-model sharing channel under the same marginal prior. '
    html+='D384/D256 changes representation size. Original D0/E0 are frozen random-feature controls. '
    html+='The full run reuses A/B/D0/E0 only for evaluation; workers cannot read their records. Smoke '
    html+='simulates all eight reduced-fleet methods and is not performance evidence.</p>'
    html+='<h2>Final outcomes and communication</h2>'+table(summary[summary.fleet_budget==final_budget])
    html+='<p>Peer and Gaussian information-message payloads are separate. Public-basis broadcast is a '
    html+='separate conservative one-off estimate for both shared and local spectral arms. Unused '
    html+='final-round learning is excluded from decision snapshots. No communication, formal privacy '
    html+='or stability advantage is presumed.</p>'
    html+='<h2>Time to eligibility</h2>'+table(timing)
    html+='<h2>Families</h2>'+table(family)
    html+='<h2>Contexts</h2>'+table(by_context)
    html+='<h2>Query modes</h2>'+table(modes)
    html+='<h2>Sharing before final query</h2>'+table(sharing[sharing['round']==cfg['queries_per_client']-1])
    html+='<p>All new Bayesian choices are replayed before evaluator tables are opened. The complete '
    html+='budget curves and per-seed comparisons are saved. The fixed-history kernel assessment '
    html+='motivated these representations but did not establish their online performance.</p></body></html>'
    (out/'report.html').write_text(html,encoding='utf-8')
