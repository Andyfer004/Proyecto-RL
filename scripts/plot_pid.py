"""Gráficas desde CSV ya generados; no ejecuta ni ajusta controladores."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thruster-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    episodes = pd.read_csv(args.results)
    traces = pd.read_csv(args.results.with_suffix(".traces.csv"))
    metadata = json.loads(args.results.with_suffix(".metadata.json").read_text())
    config = metadata["config"]
    output = args.output_dir or args.results.parent / (args.results.stem + "_plots")
    output.mkdir(parents=True, exist_ok=True)
    traces = traces[traces.scenario_id.isin(episodes.scenario_id)]
    conditions = list(episodes.condition.unique())
    ncols = min(2, len(conditions))
    nrows = (len(conditions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), squeeze=False)
    indexed = episodes.set_index("scenario_id")
    for ax, condition in zip(axes.flat, conditions):
        group = traces[traces.condition == condition]
        for identifier, trajectory in group.groupby("scenario_id", sort=False):
            success = str(indexed.loc[identifier, "success"]).lower() == "true"
            ax.plot(trajectory.x, trajectory.y, color="tab:green" if success else "tab:red", alpha=0.5, lw=0.9)
            ax.scatter(trajectory.x.iloc[-1], trajectory.y.iloc[-1], color="tab:green" if success else "tab:red", s=10)
        ax.plot([-config["platform_half_width"], config["platform_half_width"]], [0, 0], color="black", lw=4)
        ax.axhline(config["foot_drop"], color="gray", ls=":", lw=0.8)
        cases = episodes[episodes.condition == condition]
        successes = sum(str(value).lower() == "true" for value in cases.success)
        ax.set(title=f"{condition}: {successes}/{len(cases)} éxitos", xlabel="x del centro (m)", ylabel="y del centro (m)")
        ax.set_xlim(min(-3, group.x.min() - 0.2), max(3, group.x.max() + 0.2))
        ax.set_ylim(-0.2, max(13, group.y.max() + 0.2))
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(conditions):]:
        ax.set_visible(False)
    fig.suptitle("Trayectorias del centro de masa — verde: éxito; rojo: fallo")
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=160)
    plt.close(fig)

    # Elección declarada: primer escenario de cada condición, nunca el mejor.
    selected = episodes.groupby("condition", sort=False).head(1)
    fields = [("x", "x (m)", "success_x"), ("y", "y del centro (m)", None),
              ("vx", "vx (m/s)", "success_vx"), ("vy", "vy (m/s)", "success_vy"),
              ("theta", "θ (rad)", "success_angle"), ("omega", "ω (rad/s)", "success_omega")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for row in selected.itertuples():
        trajectory = traces[traces.scenario_id == row.scenario_id]
        for ax, (field, label, limit_key) in zip(axes.flat, fields):
            ax.plot(trajectory.time, trajectory[field], label=f"{row.condition}: {row.outcome}", lw=1)
            ax.set(xlabel="t (s)", ylabel=label)
            ax.grid(alpha=0.2)
    for ax, (field, _, limit_key) in zip(axes.flat, fields):
        if limit_key:
            limit = config[limit_key]
            ax.axhline(-limit, color="black", ls=":", lw=0.8)
            ax.axhline(0 if field == "vy" else limit, color="black", ls=":", lw=0.8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Estados: primer escenario de cada condición; líneas punteadas: umbrales al contacto")
    fig.tight_layout()
    fig.savefig(output / "states.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for row in selected.itertuples():
        trajectory = traces[traces.scenario_id == row.scenario_id]
        axes[0].plot(trajectory.time, config["gas_capacity"] - trajectory.gas, label=row.condition)
        axes[1].step(trajectory.time, trajectory.action, where="pre", label=row.condition, alpha=0.6)
    axes[0].axhline(config["gas_capacity"], color="black", ls=":", label="presupuesto")
    axes[0].set(ylabel="Impulso consumido (N s)")
    axes[0].legend()
    axes[1].set(xlabel="t (s)", ylabel="Acción aplicada en el intervalo previo", yticks=range(8))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Consumo y acciones: primer escenario de cada condición")
    fig.tight_layout()
    fig.savefig(output / "resources.png", dpi=160)
    plt.close(fig)
    print(f"Gráficas: {output}")


if __name__ == "__main__":
    main()
