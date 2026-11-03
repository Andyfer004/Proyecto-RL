"""Curvas descriptivas del smoke test; no compara superioridad estadística."""
import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'thruster-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, default=Path('artifacts/rl_smoke_curves.png'))
    args = parser.parse_args()
    budgets = [json.loads((run / 'metadata.json').read_text())['env_steps'] for run in args.runs]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for run in args.runs:
        updates = pd.read_csv(run / 'updates.csv')
        episodes = pd.read_csv(run / 'episodes.csv')
        validation = pd.read_csv(run / 'validation.csv')
        label = run.name
        axes[0, 0].plot(episodes.env_steps, episodes['return'].rolling(10, min_periods=1).mean(), label=label)
        if 'loss' in updates:
            axes[0, 1].plot(updates.env_steps, updates.loss, label=label)
            axes[1, 0].plot(updates.env_steps, updates.mean_abs_q, label=label + ' |Q|')
            axes[1, 0].plot(updates.env_steps, updates.mean_abs_td_error, ls='--', label=label + ' |TD|')
        axes[1, 1].plot(updates.env_steps, updates.epsilon, label=label)
        aggregated = validation.groupby('env_steps').agg(success_rate=('success', 'mean'), mean_return=('return', 'mean'))
        axes[2, 0].plot(aggregated.index, aggregated.success_rate, marker='o', label=label)
        axes[2, 1].plot(aggregated.index, aggregated.mean_return, marker='o', label=label)
    titles = [('Retorno de entrenamiento (media móvil de 10 episodios)', 'Retorno'),
              ('Huber loss media por intervalo', 'Loss'),
              ('Magnitudes Q y error TD', 'Magnitud'), ('Exploración', 'Epsilon'),
              ('Validación greedy (32 escenarios por punto)', 'Fracción de éxitos'),
              ('Retorno en validación greedy', 'Retorno medio')]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Pasos de entrenamiento', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[2, 0].set_ylim(-.05, 1.05)
    fig.suptitle('Diagnóstico con una semilla — pasos por ejecución: ' + ', '.join(f'{budget:,}' for budget in budgets))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
