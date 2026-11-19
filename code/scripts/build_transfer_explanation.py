"""Build a self-contained evidence brief from frozen v5/v6/v7 results."""
from pathlib import Path
import base64,io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results';OUT=ROOT.parent/'docs/analysis/transfer_explained.html'
v6=pd.read_csv(P/'actuator_transfer_v6/processed/transfers.csv')
v7=pd.read_csv(P/'actuator_mechanism_v7/processed/controllers.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors=['#087f8c','#244b94','#de7c23','#994f91']
def embedded(fig):
    buf=io.BytesIO();fig.savefig(buf,format='svg',bbox_inches='tight');plt.close(fig)
    return 'data:image/svg+xml;base64,'+base64.b64encode(buf.getvalue()).decode()
fig,ax=plt.subplots(figsize=(8.8,2.9))
for (method,budget),color in zip([('sobol',25),('sobol',75),('bo_lower_penalty',25),('bo_lower_penalty',75)],colors):
    g=v6[(v6.method==method)&(v6.budget==budget)].groupby('tau').gap.max()*100
    ax.plot(g.index,g.values,'o-',color=color,label=f'{"Sobol" if method=="sobol" else "BO"}, source budget {budget}')
ax.axhline(5,color='#b13d35',ls='--',label='Primary tolerance: 5%')
ax.set(xlabel='Target actuator time constant τ [s]',ylabel='Worst transfer gap [%]',ylim=(0,5.5),xticks=[.1,.15,.2,.3,.4]);ax.grid(axis='y',alpha=.15)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.28),ncol=3,frameon=False,fontsize=8)
fig1=embedded(fig)
fig,(ax,bx)=plt.subplots(1,2,figsize=(9,3.2),gridspec_kw={'width_ratios':[1.35,1]})
piv=v7.pivot(index=['seed','client','tau'],columns='arm',values='cost')
for arm,color,label in [('target_frozen_tau',colors[2],'Same vehicle; gains frozen at τ=.2'),('donor_frozen',colors[3],'Donor gains frozen at τ=.2')]:
    gaps=(piv[arm]/piv.target_redesign-1)*100
    g=gaps.groupby('tau').agg(['median','min','max']).loc[[.2,.3,.4]]
    ax.plot(g.index,g['median'],'o-',color=color,label=label);ax.fill_between(g.index,g['min'],g['max'],color=color,alpha=.12)
ax.axhline(0,color=colors[0],lw=1.5);ax.axvspan(.085,.17,color='#eff1f4')
ax.text(.1,8,'Not\nfeasible',ha='center',fontsize=8,color='#a33535');ax.text(.15,8,'Numerically\nunresolved',ha='center',fontsize=8,color='#755000')
ax.set(xlabel='Target τ [s]',ylabel='Cost increase vs redesigned gains [%]',xticks=[.1,.15,.2,.3,.4],ylim=(-1,18));ax.grid(axis='y',alpha=.15)
ax.legend(loc='upper left',bbox_to_anchor=(-.02,1.28),frameon=False,fontsize=8)
a=v7[v7.arm=='target_redesign'];vals=np.array([a.true_mse.mean(),a.noise_mse.mean(),a.cross_mse.mean()]);labels=['True error','Measurement noise','Cross term'];bottom=0
for val,col,label in zip(vals,['#087f8c','#b5ccd7','#de7c23'],labels):
    bx.bar([0],[val*1e4],bottom=bottom,color=col,label=label,width=.5);bottom+=val*1e4
bx.set(xticks=[0],xticklabels=['Shared weights + redesign'],ylabel='Mean MSE [10⁻⁴ (rad/s)²]',ylim=(0,2.5));bx.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.27));bx.grid(axis='y',alpha=.15)
fig.tight_layout(w_pad=3);fig2=embedded(fig)
summary=v6.groupby(['method','budget']).agg(n=('gap','size'),success=('success_5','sum'),median=('gap','median'),worst=('gap','max'))
trs=''.join(f'<tr><td>{"Sobol" if m=="sobol" else "BO"}</td><td>{b}</td><td>{int(r.success)}/{int(r.n)}</td><td>{100*r["median"]:.2f}%</td><td>{100*r.worst:.2f}%</td></tr>' for (m,b),r in summary.iterrows())
html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Why LQI weights transfer—and what still adapts</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#203142;font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1030px;margin:28px auto;background:white;padding:38px 48px;border-radius:12px}h1{font-size:32px;line-height:1.15;margin:6px 0 18px}h2{font-size:21px;margin:27px 0 8px}p{margin:9px 0}small,.caption{font-size:13px;color:#526270}.eyebrow{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#087f8c;font-weight:700}.answer{border-left:4px solid #087f8c;padding:12px 18px;background:#edf7f7;margin:18px 0}.equation{padding:15px;text-align:center;background:#f5f7fa;font-size:18px}figure{margin:18px 0}figure img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #dfe5eb}th{background:#f5f7fa}.limits{font-size:14px;background:#f7f7f8;padding:14px 18px;margin-top:22px}a{color:#1f5686}footer{border-top:1px solid #ddd;margin-top:22px;padding-top:12px;font-size:12px;color:#526270}@media(max-width:650px){main{margin:0;padding:24px 18px}h1{font-size:27px}}@media print{body{background:white}main{margin:0;padding:12px;max-width:none}figure,table,.answer{break-inside:avoid}a{color:inherit}}</style></head><body><main>
<div class="eyebrow">FBO–LQI · corrected evidence brief · 19 November 2026</div>
<h1>Why do the LQI weights transfer so well?</h1>
<p><strong>Andreas’s question:</strong> if vehicle calibration landscapes are similar, why optimize again on the target instead of reusing good weights?</p>
<div class="answer"><strong>Current answer:</strong> in this pilot, source-calibrated weights meet the 5% performance criterion on every tested target. The target still adapts through model-based gain computation. Target weight optimization is not needed to meet this criterion.</div>
<h2>1 · We transfer preferences, not a fixed controller</h2>
<div class="equation">Source weights ξ → <strong>target model + target actuator τ</strong> → new LQI gains K and reference gain</div>
<p>The same weights express the same design preferences, while the identified target dynamics determine the feedback law. The simulation retains the same tracking task, controller structure and normalization across vehicles.</p>
<h2>2 · Direct transfer passes without selecting a favorable donor</h2>
<p>Six targets, five other donor vehicles per target, three source-search repeats and five actuator time constants (0.10–0.40 s). Weights were selected only on each donor’s calibration data. Gains were recomputed on the target; no target weight search or best-donor selection occurred.</p>
<figure><img src="FIG1" alt="Worst transfer gap remains below five percent for every source method, budget and target actuator time constant"><figcaption class="caption">Worst gap across all targets, donors and repeats at each τ; all cases satisfy feasibility and numerical checks. Gaps use matched-episode, validated finite-search references.</figcaption></figure>
<table><thead><tr><th>Source search</th><th>Source evaluations</th><th>Within 5%</th><th>Median gap</th><th>Worst gap</th></tr></thead><tbody>ROWS</tbody></table>
<p class="caption">1,800 correlated transfer comparisons, not 1,800 independent systems. The primary arm was Sobol with 25 source evaluations. Historical source calibration and target identification are still required.</p>
<h2>3 · Gain redesign matters; measurement noise is not the whole explanation</h2>
<p>For a <strong>retrospectively identified common weight vector</strong>, we held weights fixed and compared target redesign with gains frozen at τ = 0.20 s. We also froze gains from the same-family vehicle in the other fleet realization. True and measured errors were logged on identical noisy rollouts.</p>
<figure><img src="FIG2" alt="Frozen gains increase measured tracking cost at slower actuator bandwidths; measurement noise contributes about half of redesigned-controller measured MSE"><figcaption class="caption">Left: median and range over six targets; only converged feasible costs are plotted. Right: exact sample MSE decomposition, averaged over 30 common-weight/redesign cases; includes the finite-sample cross term.</figcaption></figure>
<p>Freezing the same vehicle’s gains increases median measured cost by <strong>5.0% at τ = 0.30 s</strong> and <strong>13.2% at 0.40 s</strong>, relative to redesign. At 0.20 s the two controllers are identical. At 0.10 s no frozen-gain episode passes feasibility at both resolutions, predominantly because of the actuator-rate limit. At 0.15 s all frozen-gain banks fail numerical convergence checks; their cost estimates are excluded.</p>
<p>Measurement noise contributes about <strong>50% of measured MSE</strong> for the redesigned common-weight controller. Nevertheless, its worst gap to the frozen reference is only <strong>0.10% in measured RMSE and 0.13% in true-state RMSE</strong>. Shared weights work well even when scoring true error. This check concerns the common vector, not all 1,800 source-only transfers.</p>
<h2>4 · The good-weight regions overlap strongly</h2>
<p>On matched 256-point calibration grids, family-pair rank correlations are at least <strong>0.988</strong>. Each grid has <strong>38–79 points within 5%</strong> of its own best cost; median overlap of these sets is <strong>0.864</strong> (Jaccard). Thus modest source searches can land in a region that is also good for another vehicle. The outer objective rewards yaw tracking, with sideslip and actuator quantities as hard constraints; it does not continuously penalize effort or comfort.</p>
<div class="answer"><strong>Interpretation:</strong> similar tuning preferences, overlapping acceptable weight regions and model-based gain redesign together explain the observed transfer success. These results support direct weight reuse in this setup—not a demonstrated need for fleet-informed target BO.</div>
<div class="limits"><strong>Scope and qualifications.</strong> Six familiar vehicles from two fleet realizations; simulation only. The actuator time constant is known. State feedback uses noise-free simulated states; yaw noise enters the integrator and score. The common-vector diagnostic is retrospective, and the reference was selected for measured error, not reoptimized for true error. Finite-horizon feasibility is not a stability proof. Fixed-gain results at τ = 0.15 s remain numerically unresolved. Earlier anti-windup sign errors were corrected; original ideal-steering results still need their own impact audit.</div>
<footer><strong>Evidence:</strong> <a href="https://github.com/j-cap/2026_FBO_LQI/commit/60b7aaa340265cb875e901fe9bc26d98e43e3067">v6 direct transfer</a>; <a href="https://github.com/j-cap/2026_FBO_LQI/commit/COMMIT">v7 paired diagnostics</a>; v5 matched-grid analysis. V7: all 9,600 episode records present; code/input hashes verified; measured MSE = true MSE + noise MSE + cross term verified. All redesigned/reference cases pass both error-metric integration checks. Frozen cases at τ ≥ 0.20 s pass. No new optimization was performed for this brief.</footer>
</main></body></html>'''
import subprocess
commit=subprocess.check_output(['git','log','-1','--format=%H','--','code/results/actuator_mechanism_v7/protocol.json'],cwd=ROOT.parent,text=True).strip()
html=html.replace('FIG1',fig1).replace('FIG2',fig2).replace('ROWS',trs).replace('COMMIT',commit)
OUT.write_text(html,encoding='utf-8');print(OUT)
