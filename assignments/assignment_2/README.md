# Assignment 2, group 19: neuroevolution

Evolves the weights of a neural network that makes the John Set gecko walk to a target 1 m ahead.
We compare a mutation step size that follows a fixed schedule (annealed) with a step size that evolution sets itself (self-adaptive).
A fixed step size and random search are used as references.

## Files

- `evolve_controller.py`: robot, world, controller, fitness, the three EA variants and random search. Writes one JSON line per generation of every run.
- `analyse_results.py`: tables, statistical tests and plots made from that file. It also replays the best controller of every run (takes about a minute).
- `results/final/history.jsonl`: log of all 80 final runs (4 variants x 20 seeds).
- `results/final/analysis/`: the tables (`summary.csv`, `tests.csv`, `replay.csv`), the figures (`convergence.png`, `convergence_median.png`, `mean_fitness.png`, `final_boxplot.png`, `sigma.png`, `diversity.png`, `end_positions.png`) and `report_numbers.txt` with every number used in the report.

## Setup

The code uses the ariel framework from the course repository
(github.com/AndrzejSzczepura/EvolutionaryComputing2026, commit a435b19).
Put the files of this folder in `assignments/assignment_2/` of that repository and install it with `uv sync`.
Nothing in `src/ariel` was changed.

## Reproduce

Run from the root of the repository.

```bash
# all final runs: 4 variants x 20 seeds, 10 runs at the same time (about 3 hours on 10 cores)
uv run python assignments/assignment_2/evolve_controller.py --workers 10 --output assignments/assignment_2/results/final/history.jsonl

# tables, tests, plots and report_numbers.txt
uv run python assignments/assignment_2/analyse_results.py assignments/assignment_2/results/final/history.jsonl --outdir assignments/assignment_2/results/final/analysis
```

The default settings are the ones used in the report: 10 s simulation, population 20, 300 generations, seeds 1 to 20, starting weights N(0, 0.5).
A run with the same seed gives the same result, also when runs are done in parallel.

Short test run (about a minute):

```bash
uv run python assignments/assignment_2/evolve_controller.py --seeds 1 --generations 5 --population 6 --output assignments/assignment_2/results/test/history.jsonl
uv run python assignments/assignment_2/analyse_results.py assignments/assignment_2/results/test/history.jsonl --outdir assignments/assignment_2/results/test/analysis
```
