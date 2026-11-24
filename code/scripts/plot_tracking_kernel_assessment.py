"""Vector accuracy/size figure from the completed read-only kernel assessment."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'docs/analysis/kernel_representation_assessment'


def main():
    data=pd.read_csv(OUT/'representation_comparison.csv')
    matplotlib.rcParams['svg.hashsalt']='fedbo-kernel-assessment'
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
    colors=dict(rff='#a35b39',spectral='#3478b2',pivoted_cholesky='#248363')
    labels=dict(rff='Random Fourier',spectral='Public spectral',pivoted_cholesky='Pivoted Cholesky')
    columns=['relative_kernel_Frobenius_error','worst_context_mean_RMSE','worst_group_p95_exact_score_regret']
    titles=['Public prior approximation','Posterior mean agreement','Acquisition agreement']
    ylabels=['Relative kernel Frobenius error','Worst context log-cost RMSE','Worst group 95th percentile exact-score gap']
    limits=[.02,.05,.10]
    for family,color in colors.items():
        g=data[(data.family==family)&(data.representation!='spectral_426')].sort_values('feature_width')
        for ax,column in zip(axes,columns):
            ax.plot(g.feature_width,g[column],marker='o',color=color,label=labels[family])
    selected=data[data.representation=='spectral_256'].iloc[0]
    for ax,column,title,ylabel,limit in zip(axes,columns,titles,ylabels,limits):
        ax.axhline(limit,color='#666666',linestyle='--',linewidth=1,label='Accuracy target')
        ax.scatter([selected.feature_width],[selected[column]],marker='*',s=150,color='#d99923',edgecolor='#684609',zorder=5)
        ax.set(title=title,xlabel='Coefficients, including intercept',ylabel=ylabel,yscale='log',xticks=[65,129,257,385,513])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='upper right')
    vector=OUT/'accuracy_and_basis_size.svg'
    fig.savefig(vector,bbox_inches='tight',metadata={'Date':None})
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
    preview=ROOT/'figures/kernel_assessment.png';preview.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(preview,dpi=140,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
