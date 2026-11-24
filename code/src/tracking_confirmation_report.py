"""Predeclared final-budget A/B comparisons and fleet-level replication units."""
import base64
import io

import pandas as pd

from .tracking_confirmation import METHODS

LABELS=dict(zip(METHODS,['A: peer GP','B: local mean','D256: shared/local','E256: local spectral']))


def write_report(out,trials,cells,queries,paired,units,cfg,mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    final=cfg['fleet_budgets'][-1]
    summary=trials.groupby(['condition','method','fleet_budget'],sort=False).agg(
        client_runs=('client','size'),programme_runs=('seed','nunique'),physical_plants=('plant_id','nunique'),
        context_success_rate=('context_success','mean'),eligible_policies=('fully_feasible','sum'),
        conditional_mean_J=('J','mean'),reference_coverage=('reference_coverage','mean'),
        mean_ineligible_tests_per_client=('ineligible_queries','mean')).reset_index()
    counts=cells.groupby(['condition','method','fleet_budget'],sort=False).agg(
        context_trials=('success','size'),successful_contexts=('success','sum'),
        validation_eligible_contexts=('eligible','sum')).reset_index()
    summary=summary.merge(counts,on=['condition','method','fleet_budget'],validate='one_to_one')
    per_run=trials[trials.fleet_budget==final].groupby(['condition','fleet_id','seed','method'],sort=False).agg(
        client_runs=('client','size'),eligible_policies=('fully_feasible','sum'),context_success=('context_success','mean'),
        conditional_mean_J=('J','mean')).reset_index()
    family=cells[cells.fleet_budget==final].groupby(['condition','family','method','context'],sort=False).agg(
        context_trials=('success','size'),successes=('success','sum'),eligible_contexts=('eligible','sum'),
        conditional_mean_J=('J','mean')).reset_index()
    modes=queries.groupby(['condition','fleet_id','seed','method','mode'],sort=False).agg(
        requests=('eligible','size'),eligible_requests=('eligible','sum'),new_incumbents=('became_incumbent','sum')).reset_index()
    for name,frame in [('online_summary',summary),('per_run_outcomes',per_run),('family_context_outcomes',family),('query_modes',modes),
        ('final_comparisons',paired[paired.fleet_budget==final]),('final_replication_units',units[units.fleet_budget==final])]:
        frame.to_csv(out/(name+'.csv'),index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,condition in enumerate(cfg['conditions']):
        for method,color in zip(METHODS,['#a35b39','#3376b0','#248363','#8c61a8']):
            g=summary[(summary.condition==condition)&(summary.method==method)]
            axes[row,0].plot(g.fleet_budget,100*g.context_success_rate,marker='o',color=color,label=LABELS[method])
            axes[row,1].plot(g.fleet_budget,g.eligible_policies/g.client_runs,marker='o',color=color)
        axes[row,0].set(title=condition,ylabel='Near-reference contexts [%]',ylim=(-2,102))
        axes[row,1].set(ylabel='Validation-eligible policies',ylim=(-.02,1.02))
        for ax in axes[row]:
            ax.set(xlabel='Fleet online requests',xscale='log',xticks=cfg['fleet_budgets'])
            ax.set_xticklabels([str(b) for b in cfg['fleet_budgets']]);ax.minorticks_off();ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.08),ncol=2)
    fig.savefig(out/'confirmation_curves.png',dpi=140,bbox_inches='tight')
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=110,bbox_inches='tight');plt.close(fig)
    def table(frame):return '<div class="table">'+frame.to_html(index=False,float_format=lambda x:f'{x:.4g}',na_rep='unavailable')+'</div>'
    title='SMOKE: software verification only' if mode=='smoke' else 'Frozen D256 confirmation: A first, then B'
    html='<!doctype html><html><head><meta charset="utf-8"><title>D256 confirmation</title><style>'
    html+='body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#192736}table{border-collapse:collapse;font-size:12px}'
    html+='td,th{padding:6px;border:1px solid #ccd5df}.table{overflow-x:auto}img{max-width:100%}</style></head><body><h1>'+title+'</h1>'
    html+='<p>All four arms start cold. D256, its public basis, priors, acquisition, peer rule and local cap are frozen. '
    html+='Every generated fleet and failed test remains in the denominator. Held-out evaluation is opened after the online decisions.</p>'
    end=paired[paired.fleet_budget==final]
    html+='<h2>Primary: D256 minus A</h2>'+table(end[end.role=='primary'])
    html+='<h2>Secondary: D256 minus B</h2>'+table(end[end.role=='secondary'])
    html+='<h2>Sharing control: D256 minus E256</h2>'+table(end[end.role=='sharing_control'])
    html+='<p>Intervals resample ten heterogeneous fleets after averaging their three runs, or ten homogeneous programme/noise seeds. '
    html+='They are percentile bootstrap intervals, with no context-level independence assumption. Confirmation flags require a positive '
    html+='success lower bound, complete references, preserved eligible-policy count and lower mean cost on common eligible policies. '
    html+='Missing references remain unsuccessful and prevent a confirmation flag. Smoke never supports confirmation. '
    html+='Per-run and per-unit results retain reversals; conditional cost means always retain their eligible denominators.</p>'
    html+='<img alt="Confirmation budget curves" src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'">'
    html+='<h2>Final outcomes</h2>'+table(summary[summary.fleet_budget==final])
    html+='<h2>Replication units</h2>'+table(units[units.fleet_budget==final])
    html+='<h2>Programme/noise runs</h2>'+table(per_run)
    html+='<h2>Families and contexts</h2>'+table(family)+'</body></html>'
    (out/'report.html').write_text(html,encoding='utf-8')
