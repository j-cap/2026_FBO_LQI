"""Plot the declared replication-unit effects from the audited saved tables."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=ROOT.parent/'docs/analysis/spectral_confirmation'


def main():
    aggregate=pd.read_csv(OUT/'final_comparisons.csv')
    units=pd.read_csv(OUT/'final_replication_units.csv')
    baselines=[('peer_gp_bo','A'),('peer_local_cost_bo','B'),('peer_spectral_local_256_bo','E256')]
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'fedbo_confirmation','font.size':9})
    fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,constrained_layout=True)
    for row,condition in enumerate(['homogeneous','heterogeneous']):
        for column,(baseline,label) in enumerate(baselines):
            ax=axes[row,column]
            group=units[(units.condition==condition)&(units.baseline==baseline)].sort_values('replication_unit')
            if len(group)!=10:raise RuntimeError('Need all ten declared replication units.')
            record=aggregate[(aggregate.condition==condition)&(aggregate.baseline==baseline)].iloc[0]
            values=100*group.context_success_delta.to_numpy();y=np.arange(10)[::-1]
            colors=['#257a57' if x>1e-10 else '#ae4a40' if x< -1e-10 else '#7b8490' for x in values]
            ax.axvline(0,color='#697683',lw=.8)
            ax.scatter(values,y,c=colors,s=27,zorder=3)
            center=100*record.context_success_delta;lower=100*record.unit_bootstrap_lower;upper=100*record.unit_bootstrap_upper
            ax.errorbar(center,10.8,xerr=[[center-lower],[upper-center]],fmt='D',color='#284f7c',capsize=3,ms=5)
            ax.axhline(10.1,color='#cdd5dd',lw=.7)
            ax.set(title=f'{condition}: D256 minus {label}\nMean {center:+.2f} pp; interval [{lower:+.2f}, {upper:+.2f}]',
                xlim=(-35,60),ylim=(-.6,11.6),yticks=[*range(10),10.8])
            if column==0:ax.set_yticklabels([*group.replication_unit.tolist()[::-1],'Mean'],fontsize=8)
            else:ax.set_yticklabels([])
            ax.grid(axis='x',alpha=.2)
            if row==1:ax.set_xlabel('Near-reference context-success difference [pp]')
    fig.suptitle('Frozen D256 confirmation: declared replication units',fontsize=13)
    fig.supxlabel('Dots: ten programme/noise seeds (homogeneous) or fleet averages of three runs (heterogeneous). Diamond: mean with 95% unit-bootstrap interval.',fontsize=9)
    svg=OUT/'replication_effects.svg'
    fig.savefig(svg,bbox_inches='tight',metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    preview=ROOT/'results/tracking_confirmation_analysis/replication_effects.png'
    preview.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(preview,dpi=140,bbox_inches='tight');plt.close(fig)
    print('Figure: '+str(OUT/'replication_effects.svg'))


if __name__=='__main__':main()
