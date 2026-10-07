"""Create Assignment 2 summary tables, statistical tests and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Circle
from scipy import stats

from evolve_controller import SUCCESS_DISTANCE, TARGET_POSITION, NeuralController, SimulationEvaluator

#order and names used in every table and plot
NAMES = {
    "annealed_mutation": "annealed σ (schedule)",
    "self_adaptive": "self-adaptive σ",
    "fixed_mutation": "fixed σ = 0.12",
    "random_search": "random search",
}
COLOURS = {
    "annealed_mutation": "tab:blue",
    "self_adaptive": "tab:orange",
    "fixed_mutation": "tab:green",
    "random_search": "tab:gray",
}
#main comparison first, then the reference lines
COMPARISONS = [
    ("annealed_mutation", "self_adaptive"),
    ("annealed_mutation", "fixed_mutation"),
    ("self_adaptive", "fixed_mutation"),
    ("annealed_mutation", "random_search"),
    ("self_adaptive", "random_search"),
    ("fixed_mutation", "random_search"),
]


#group the log lines per variant and seed
def load_runs(path):
    runs = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        runs.setdefault(row["variant"], {}).setdefault(row["seed"], []).append(row)
    for seeds in runs.values():
        for rows in seeds.values():
            rows.sort(key=lambda row: row["generation"])
    return runs


#one row per seed, one column per generation
def column(runs, key):
    return np.array([[row[key] for row in runs[seed]] for seed in sorted(runs)], dtype=float)


#vargha-delaney effect size: chance that a run of a is better (lower) than a run of b
def a12(a, b):
    a = np.asarray(a)[:, None]
    b = np.asarray(b)[None, :]
    return float((np.sum(a < b) + 0.5 * np.sum(a == b)) / (a.size * b.size))


#95% bootstrap interval for a12, resampling the runs of both variants
def a12_interval(a, b, repeats=2000):
    rng = np.random.default_rng(0)
    values = [a12(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(repeats)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


#holm correction because we run several tests
def holm(p_values):
    order = np.argsort(p_values)
    corrected = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running_max = max(running_max, value)
        corrected[index] = running_max
    return corrected.tolist()


#simulations until the best robot is within SUCCESS_DISTANCE, nan if never
def first_success(best, evaluations):
    result = []
    for run in best:
        reached = np.nonzero(run <= SUCCESS_DISTANCE)[0]
        result.append(float(evaluations[reached[0]]) if len(reached) else float("nan"))
    return result


#same, but runs that never got there count as slowest (inf) so they can be ranked in a test
def simulations_to_success(best, evaluations):
    return np.nan_to_num(np.array(first_success(best, evaluations)), nan=np.inf)


def replay(genome, evaluator, target_input=TARGET_POSITION):
    """Run one controller again and return the core's (x, y) path, every 0.1 s.

    target_input is the target the network is told about, normally the real one.
    """
    mujoco.mj_resetData(evaluator.model, evaluator.data)
    mujoco.mj_forward(evaluator.model, evaluator.data)
    controller = NeuralController(
        weights=np.asarray(genome, dtype=np.float64),
        output_size=evaluator.output_size,
        core_binding=evaluator.core_binding,
        target=np.asarray(target_input, dtype=np.float64),
    )
    path = [np.asarray(evaluator.core_binding.xpos[:2]).copy()]
    #same loop as thread_safe_runner, so the end point must match the logged fitness
    for step in range(int(evaluator.duration / evaluator.model.opt.timestep)):
        controller.set_control(evaluator.model, evaluator.data)
        mujoco.mj_step(evaluator.model, evaluator.data)
        if (step + 1) % 50 == 0:
            path.append(np.asarray(evaluator.core_binding.xpos[:2]).copy())
    return np.array(path)


#generation in which the best distance went down for the last time
def last_improvement(best):
    result = []
    for run in best:
        improved = np.nonzero(np.diff(run) < -1e-9)[0]
        result.append(int(improved[-1] + 1) if len(improved) else 0)
    return result


#mean over the runs with a band of one standard deviation
def band(ax, x, values, variant):
    mean = values.mean(axis=0)
    spread = values.std(axis=0, ddof=1)
    ax.plot(x, mean, color=COLOURS[variant], label=NAMES[variant])
    ax.fill_between(x, mean - spread, mean + spread, color=COLOURS[variant], alpha=0.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("analysis"))
    #replaying all best controllers takes about half a minute
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.history)
    variants = [variant for variant in NAMES if variant in runs]
    best = {variant: column(runs[variant], "best") for variant in variants}
    final = {variant: best[variant][:, -1] for variant in variants}
    first_seed = sorted(runs[variants[0]])[0]
    generations = np.array([row["generation"] for row in runs[variants[0]][first_seed]])
    evaluations = np.array([row["evaluations"] for row in runs[variants[0]][first_seed]])
    lines = []

    #summary table: quality, reliability and speed per variant
    summary = ["variant,runs,mean_final,std_final,median_final,q25_final,q75_final,min_final,max_final,success_rate,mean_evals_to_success,median_evals_to_success,mean_last_improvement_gen"]
    lines.append(f"success = best robot within {SUCCESS_DISTANCE} m of the target")
    lines.append("final distance in m, sims = simulations, iqr = 25th-75th percentile")
    for variant in variants:
        speed = np.array(first_success(best[variant], evaluations))
        success = ~np.isnan(speed)
        last = last_improvement(best[variant])
        values = final[variant]
        q25, q75 = np.percentile(values, [25, 75])
        mean_speed = np.nanmean(speed) if success.any() else float("nan")
        median_speed = np.nanmedian(speed) if success.any() else float("nan")
        summary.append(
            f"{variant},{len(values)},{values.mean():.4f},{values.std(ddof=1):.4f},{np.median(values):.4f},"
            f"{q25:.4f},{q75:.4f},{values.min():.4f},{values.max():.4f},{success.mean():.2f},"
            f"{mean_speed:.0f},{median_speed:.0f},{np.mean(last):.1f}"
        )
        lines.append(
            f"{variant}: mean {values.mean():.3f}, std {values.std(ddof=1):.3f}, median {np.median(values):.3f}, "
            f"iqr {q25:.3f}-{q75:.3f}, min {values.min():.3f}, max {values.max():.3f}, success {success.sum()}/{len(values)}, "
            f"sims to success (successful runs only) mean {mean_speed:.0f} median {median_speed:.0f}, "
            f"last improvement gen {np.mean(last):.0f}"
        )
    (args.outdir / "summary.csv").write_text("\n".join(summary) + "\n")
    start_best = np.concatenate([best[variant][:, 0] for variant in variants])
    lines.append(f"best distance in generation 0 (all runs): mean {start_best.mean():.3f}, min {start_best.min():.3f}, max {start_best.max():.3f}")

    #does the choice of 0.2 m decide the outcome? success at other thresholds
    lines.append("")
    for variant in variants:
        counts = ", ".join(f"{limit} m {int(np.sum(final[variant] <= limit))}/{len(final[variant])}" for limit in (0.1, 0.2, 0.3))
        lines.append(f"{variant}: success at {counts}")

    #did the runs level off? improvement in the last 50 generations (fewer for short test runs)
    lines.append("")
    back = min(50, len(generations) - 1)
    for variant in variants:
        gain = best[variant][:, -1 - back] - best[variant][:, -1]
        lines.append(
            f"{variant}: still improving in last {back} gens {int(np.sum(gain > 1e-9))}/{len(gain)} runs, "
            f"mean gain {gain.mean() * 100:.1f} cm"
        )

    #statistical tests on the final distance, plus the success rates of the main comparison
    tests = []
    for a, b in COMPARISONS:
        if a in final and b in final:
            result = stats.mannwhitneyu(final[a], final[b], alternative="two-sided")
            tests.append((f"final distance {a} vs {b}", "Mann-Whitney U", float(result.statistic), float(result.pvalue), a12(final[a], final[b]), a12_interval(final[a], final[b])))
    #speed: simulations until within SUCCESS_DISTANCE, runs that never got there rank as slowest
    for a, b in COMPARISONS[:3]:
        if a in best and b in best:
            speed_a = simulations_to_success(best[a], evaluations)
            speed_b = simulations_to_success(best[b], evaluations)
            result = stats.mannwhitneyu(speed_a, speed_b, alternative="two-sided")
            tests.append((f"simulations to success {a} vs {b}", "Mann-Whitney U", float(result.statistic), float(result.pvalue), a12(speed_a, speed_b), a12_interval(speed_a, speed_b)))
    if "annealed_mutation" in final and "self_adaptive" in final:
        success_a = int(np.sum(final["annealed_mutation"] <= SUCCESS_DISTANCE))
        success_b = int(np.sum(final["self_adaptive"] <= SUCCESS_DISTANCE))
        table = [[success_a, len(final["annealed_mutation"]) - success_a], [success_b, len(final["self_adaptive"]) - success_b]]
        result = stats.fisher_exact(table)
        tests.append(("success rate annealed_mutation vs self_adaptive", "Fisher exact", float(result.statistic), float(result.pvalue), float("nan"), (float("nan"), float("nan"))))
    corrected = holm([test[3] for test in tests])
    rows = ["comparison,test,statistic,p_value,p_holm,a12,a12_low,a12_high"]
    lines.append("")
    lines.append(f"tests: {len(tests)}, holm corrected, a12 = chance that the first one is better, with 95% bootstrap interval")
    for (name, test, statistic, p_value, effect, (low, high)), p_holm in zip(tests, corrected):
        rows.append(f"{name},{test},{statistic:.3f},{p_value:.3e},{p_holm:.3e},{effect:.3f},{low:.3f},{high:.3f}")
        lines.append(f"{name}: {test}, stat {statistic:.1f}, p {p_value:.2e}, holm p {p_holm:.2e}, a12 {effect:.2f} ({low:.2f}-{high:.2f})")
    (args.outdir / "tests.csv").write_text("\n".join(rows) + "\n")

    #sensitivity check: the same seed gives every EA variant the same first population, so runs can be paired
    lines.append("")
    lines.append("paired check (wilcoxon signed-rank on final distance, runs paired by seed, not holm corrected)")
    for a, b in COMPARISONS[:3]:
        if a in final and b in final and len(final[a]) == len(final[b]) > 1:
            result = stats.wilcoxon(final[a], final[b])
            better = int(np.sum(final[a] < final[b]))
            lines.append(f"{a} vs {b}: p {result.pvalue:.2e}, {a} closer in {better}/{len(final[a])} seeds")

    #step size and diversity, to explain why the variants behave differently
    lines.append("")
    #start, a sixth, half way and the end of the run (0/50/150/300 for the final runs)
    checkpoints = (0, len(generations) // 6, len(generations) // 2, len(generations) - 1)
    for variant in [v for v in variants if v != "random_search"]:
        sigma = column(runs[variant], "sigma_mean")
        diversity = column(runs[variant], "diversity")
        better = column(runs[variant], "children_better")[:, 1:]
        half = better.shape[1] // 2
        lines.append(
            f"{variant}: sigma at gen {'/'.join(str(generations[g]) for g in checkpoints)} = "
            + "/".join(f"{sigma[:, g].mean():.3f}" for g in checkpoints)
            + f", diversity at end mean {diversity[:, -1].mean():.2f} std {diversity[:, -1].std(ddof=1):.2f}, "
            f"children better than best parent {np.mean(better[:, :half]) * 100:.1f}% (first half) "
            f"{np.mean(better[:, half:]) * 100:.1f}% (second half), "
            f"population mean distance at end {column(runs[variant], 'mean')[:, -1].mean():.2f}"
        )
    if "self_adaptive" in runs:
        sigma_end = column(runs["self_adaptive"], "sigma_mean")[:, -1]
        result = stats.spearmanr(sigma_end, final["self_adaptive"])
        lines.append(f"self_adaptive sigma at end per run: {', '.join(f'{value:.3f}' for value in sigma_end)}")
        inside = column(runs["self_adaptive"], "sigma_std")[:, -1]
        lines.append(f"self_adaptive sigma std inside one population at end: min {inside.min():.3f}, max {inside.max():.3f}")
        lines.append(f"spearman sigma at end vs final distance: r {result.statistic:.2f}, p {result.pvalue:.2e}")

    #the required plot: mean and std of the best distance over generations
    figure, ax = plt.subplots(figsize=(7, 4.5))
    for variant in variants:
        band(ax, generations, best[variant], variant)
    ax.axhline(SUCCESS_DISTANCE, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best distance to target (m), lower is better")
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.outdir / "convergence.png", dpi=180)
    plt.close(figure)

    #final distance of every run
    figure, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([final[variant] for variant in variants], showfliers=False)
    for index, variant in enumerate(variants, start=1):
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(final[variant]))
        ax.scatter(index + jitter, final[variant], s=12, color=COLOURS[variant], zorder=3)
    ax.set_xticks(range(1, len(variants) + 1), [NAMES[variant] for variant in variants])
    ax.axhline(SUCCESS_DISTANCE, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("Final best distance to target (m)")
    figure.tight_layout()
    figure.savefig(args.outdir / "final_boxplot.png", dpi=180)
    plt.close(figure)

    #step size over time, every self-adaptive run as a thin line
    figure, ax = plt.subplots(figsize=(7, 4.5))
    for variant in [v for v in variants if v != "random_search"]:
        band(ax, generations, column(runs[variant], "sigma_mean"), variant)
    if "self_adaptive" in runs:
        for run in column(runs["self_adaptive"], "sigma_mean"):
            ax.plot(generations, run, color=COLOURS["self_adaptive"], linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean mutation step size σ in the population")
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.outdir / "sigma.png", dpi=180)
    plt.close(figure)

    #how spread out the population is
    figure, ax = plt.subplots(figsize=(7, 4.5))
    for variant in [v for v in variants if v != "random_search"]:
        band(ax, generations, column(runs[variant], "diversity"), variant)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity (mean distance to population centre)")
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.outdir / "diversity.png", dpi=180)
    plt.close(figure)

    #mean distance of the whole population (for random search: mean of the samples of that generation)
    figure, ax = plt.subplots(figsize=(7, 4.5))
    for variant in variants:
        band(ax, generations, column(runs[variant], "mean"), variant)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean distance to target in the population (m)")
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.outdir / "mean_fitness.png", dpi=180)
    plt.close(figure)

    #median and interquartile range of the best distance, because the final distances are far from normal
    figure, ax = plt.subplots(figsize=(7, 4.5))
    for variant in variants:
        q25, median, q75 = np.percentile(best[variant], [25, 50, 75], axis=0)
        ax.plot(generations, median, color=COLOURS[variant], label=NAMES[variant])
        ax.fill_between(generations, q25, q75, color=COLOURS[variant], alpha=0.15)
    ax.axhline(SUCCESS_DISTANCE, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best distance to target (m), median and IQR")
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.outdir / "convergence_median.png", dpi=180)
    plt.close(figure)

    #replay the best controller of every run to see where the robot ends up
    if not args.no_replay:
        evaluator = SimulationEvaluator()
        rows = ["variant,seed,final_x,final_y,final_distance,closest_distance,distance_from_start,angle_deg"]
        figure, ax = plt.subplots(figsize=(9, 4.5))
        lines.append("")
        start = replay(runs[variants[0]][sorted(runs[variants[0]])[0]][-1]["best_genome"], evaluator)[0]
        lines.append(
            f"replay of best robot per run: {evaluator.duration:.0f} s, start ({start[0]:.2f}, {start[1]:.2f}), "
            f"target ({TARGET_POSITION[0]:.0f}, {TARGET_POSITION[1]:.0f}), angle 0 = straight at the target"
        )
        largest_gap = 0.0
        timing = []
        for variant in variants:
            ends = []
            for seed in sorted(runs[variant]):
                path = replay(runs[variant][seed][-1]["best_genome"], evaluator)
                #for successful EA runs: where was the robot 1 s before the end, and what happens if
                #the network is told the target is twice as far (does it steer, or just walk for 10 s?)
                if variant != "random_search" and runs[variant][seed][-1]["best"] <= SUCCESS_DISTANCE:
                    moved_target = replay(runs[variant][seed][-1]["best_genome"], evaluator, TARGET_POSITION * np.array([2.0, 1.0, 1.0]))
                    timing.append((variant, float(np.linalg.norm(path[-11] - TARGET_POSITION[:2])), float(path[-1][0]), float(moved_target[-1][0])))
                distance = float(np.linalg.norm(path[-1] - TARGET_POSITION[:2]))
                #the simulation is deterministic, so the replay must give the logged fitness again
                largest_gap = max(largest_gap, abs(distance - runs[variant][seed][-1]["best"]))
                closest = float(np.min(np.linalg.norm(path - TARGET_POSITION[:2], axis=1)))
                travelled = float(np.linalg.norm(path[-1] - path[0]))
                #angle between the direction the robot moved and the direction of the target, seen from the start
                moved = np.arctan2(path[-1][1] - path[0][1], path[-1][0] - path[0][0])
                aimed = np.arctan2(TARGET_POSITION[1] - path[0][1], TARGET_POSITION[0] - path[0][0])
                angle = float(np.degrees((moved - aimed + np.pi) % (2 * np.pi) - np.pi))
                ends.append((distance, closest, travelled, angle))
                rows.append(f"{variant},{seed},{path[-1][0]:.4f},{path[-1][1]:.4f},{distance:.4f},{closest:.4f},{travelled:.4f},{angle:.1f}")
                if variant != "random_search":
                    ax.plot(path[:, 0], path[:, 1], color=COLOURS[variant], linewidth=0.6, alpha=0.5)
                ax.scatter(path[-1][0], path[-1][1], color=COLOURS[variant], s=14, zorder=3)
            ends = np.array(ends)
            stuck = ends[ends[:, 0] > SUCCESS_DISTANCE]
            passed = int(np.sum(stuck[:, 1] <= SUCCESS_DISTANCE)) if len(stuck) else 0
            if len(stuck):
                lines.append(
                    f"{variant}: unsuccessful {len(stuck)}/{len(ends)}, distance from start mean {stuck[:, 2].mean():.2f} m, "
                    f"angle to target direction mean {np.abs(stuck[:, 3]).mean():.0f} deg, walked past the target {passed}"
                )
            else:
                lines.append(f"{variant}: unsuccessful 0/{len(ends)}")
        lines.append(f"largest difference replay vs logged fitness: {largest_gap:.2e} m")
        lines.append("")
        lines.append("successful runs, replayed: distance to target 1 s before the end, and x at the end when the network is told the target is at 2 m")
        for variant in [v for v in variants if v != "random_search"]:
            rows_v = [row for row in timing if row[0] == variant]
            if rows_v:
                late, normal_x, moved_x = np.array([row[1:] for row in rows_v]).T
                lines.append(
                    f"{variant}: {len(rows_v)} runs, distance 1 s before end mean {late.mean():.2f} m, "
                    f"x at end mean {normal_x.mean():.2f} (target told at 1 m) vs {moved_x.mean():.2f} (told at 2 m), "
                    f"max x when told 2 m {moved_x.max():.2f}"
                )
        (args.outdir / "replay.csv").write_text("\n".join(rows) + "\n")
        for variant in variants:
            ax.scatter([], [], color=COLOURS[variant], s=14, label=NAMES[variant])
        ax.scatter(*start, marker="s", color="black", s=30, label="start")
        ax.add_patch(Circle(TARGET_POSITION[:2], SUCCESS_DISTANCE, fill=False, linestyle=":", color="black"))
        ax.scatter(*TARGET_POSITION[:2], marker="*", color="black", s=90, label="target")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        figure.tight_layout()
        figure.savefig(args.outdir / "end_positions.png", dpi=180)
        plt.close(figure)

    (args.outdir / "report_numbers.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written to {args.outdir}")


if __name__ == "__main__":
    main()
