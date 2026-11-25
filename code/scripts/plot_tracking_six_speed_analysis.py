"""Plot the audited six-speed pilot tradeoff without touching source results."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / 'docs/analysis/six_speed_contextual_pilot'
METHODS = ['peer_gp_bo', 'peer_local_cost_bo', 'peer_spectral_shared_256_bo', 'peer_spectral_local_256_bo']
LABELS = ['A', 'B', 'D256', 'E256']
COLORS = ['#66748a', '#a77a39', '#007d75', '#7362ad']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path)
    args = parser.parse_args()
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'six_speed_pilot', 'font.size': 10})
    summary = pd.read_csv(OUT / 'final_outcomes.csv').set_index('method').loc[METHODS]
    allocation = pd.read_csv(OUT / 'query_allocation.csv')
    bo = allocation[allocation['mode'] == 'feasibility_penalized_LCB'].set_index('method').loc[METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
    panels = [
        ('Near-reference context success', summary.context_success_percent.to_numpy(),
         [f'{v:.2f}%' for v in summary.context_success_percent], 100),
        ('Complete eligible policies', 100*summary.complete_eligible_on_encountered.to_numpy()/summary.client_runs.to_numpy(),
         [f'{n}/27' for n in summary.complete_eligible_on_encountered], 100),
        ('Ineligible BO requests', bo.ineligible_percent.to_numpy(),
         [f'{n}/{d}' for n, d in zip(bo.ineligible, bo.requests)], 45),
    ]
    for ax, (title, values, labels, maximum) in zip(axes, panels):
        ax.bar(LABELS, values, color=COLORS, width=.65, zorder=3)
        for x, y, label in zip(range(4), values, labels):
            ax.text(x, y + maximum*.025, label, ha='center', va='bottom', fontsize=10)
        ax.set(title=title, ylabel='Percent', ylim=(0, maximum))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2, zorder=0)
        ax.tick_params(axis='x', length=0)
    axes[0].set_xlabel('Higher is better')
    axes[1].set_xlabel('Over encountered contexts; higher is better')
    axes[2].set_xlabel('Lower is better')
    fig.suptitle('Six-speed homogeneous pilot: better context success does not ensure complete calibration', fontsize=12)
    fig.text(.5, .025, 'Three programme/noise seeds on one common plant. Success uses equal vehicle/seed weights. No confirmation claim.', ha='center', fontsize=9, color='#526674')
    fig.subplots_adjust(left=.055, right=.985, top=.81, bottom=.21, wspace=.33)
    fig.savefig(OUT / 'six_speed_tradeoff.svg', metadata={'Date': None})
    svg = OUT / 'six_speed_tradeoff.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=135)
    plt.close(fig)
    print(OUT / 'six_speed_tradeoff.svg')


if __name__ == '__main__':
    main()
