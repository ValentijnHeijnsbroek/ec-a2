from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from ariel.ec import Individual, Population
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import thread_safe_runner


ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
SPAWN_POSITION = (0.0, 0.0, 0.1)
TARGET_POSITION = np.array([1.0, 0.0, 0.0], dtype=np.float64)
#with 5 seconds no controller got further than halfway, with 10 seconds the target can be reached
SIMULATION_DURATION = 10.0
HIDDEN_SIZE = 16
INPUT_SIZE = 5
DEFAULT_POPULATION_SIZE = 20
#about 5700 simulations per run, the same fixed budget for every variant
#in test runs most of the improvement happened well before this
DEFAULT_GENERATIONS = 300
DEFAULT_SEEDS = list(range(1, 21))
#scale of the random starting weights, used by the EA and by random search
#same as the course template, with 0.1 no random robot moved more than a few cm
INIT_SCALE = 0.5
#a run counts as successful once the best robot ends within this distance of the target
SUCCESS_DISTANCE = 0.2
#every mutation variant starts with this step size
START_SIGMA = 0.22
#average step size of the annealed schedule, so the fixed variant is not just smaller steps
FIXED_SIGMA = 0.12
#lower bound so a self-adaptive step size can not become zero
MIN_SIGMA = 0.001
VARIANTS = ["fixed_mutation", "annealed_mutation", "self_adaptive", "random_search"]


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def genome_size(output_size: int) -> int:
    """Number of weights and biases in the controller (one hidden layer)."""

    #first layer: input features to hidden neurons.
    first_weights = INPUT_SIZE * HIDDEN_SIZE
    #one bias value is needed for every hidden neuron.
    first_biases = HIDDEN_SIZE
    #second layer: hidden neurons to one output per robot actuator.
    second_weights = HIDDEN_SIZE * output_size
    #one bias value needed for every actuator output.
    second_biases = output_size
    #the complete flattened genome is the sum of all four parameter groups.
    return first_weights + first_biases + second_weights + second_biases


#simulations in one EA run: the first population plus the new children of every generation
def evaluation_budget(population_size: int, generations: int) -> int:
    return population_size + generations * (population_size - 1)


@dataclass
class NeuralController:
    """Convert one evolved weight vector into smooth actuator commands.
    """

    weights: np.ndarray
    output_size: int
    core_binding: object
    target: np.ndarray

    def __post_init__(self):
        #the offsets describe where each parameter group starts in the genome.
        first_weight_end = INPUT_SIZE * HIDDEN_SIZE
        first_bias_end = first_weight_end + HIDDEN_SIZE
        second_weight_end = first_bias_end + HIDDEN_SIZE * self.output_size

        #reshape the flat vector into the first layer's weight matrix.
        self.first_weights = self.weights[:first_weight_end].reshape(
            INPUT_SIZE,
            HIDDEN_SIZE,
        )
        #read the hidden-layer bias values.
        self.first_biases = self.weights[first_weight_end:first_bias_end]
        #reshape the second layer's weights into one row per hidden neuron.
        self.second_weights = self.weights[first_bias_end:second_weight_end].reshape(
            HIDDEN_SIZE,
            self.output_size,
        )
        #the other values are the output-layer biases.
        self.second_biases = self.weights[second_weight_end:]

    def set_control(self, _model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Compute and apply one actuator command during a physics step."""

        #read the current core position from the MuJoCo-bound geometry.
        position = np.asarray(self.core_binding.xpos, dtype=np.float64)
        #use the horizontal distance as the controller's target signal
        #we keep extreme positions from producing huge inputs with clip
        target_delta = np.clip(self.target[:2] - position[:2], -2.0, 2.0)
        #convert time into a smoothsignal that runs periodically so the network can create rhythmic motion instead of only reacting to a static position.
        phase = 2.0 * np.pi * data.time / 1.5
        #the bias, phase, and target direction are the five network inputs.
        observation = np.array(
            [1.0, np.sin(phase), np.cos(phase), target_delta[0], target_delta[1]],
            dtype=np.float64,
        )
        #the hidden layer uses tanh to keep internal activations bounded.
        hidden = np.tanh(observation @ self.first_weights + self.first_biases)
        #the output layer also uses tanh, producing values in [-1, 1].
        normalized_output = np.tanh(hidden @ self.second_weights + self.second_biases)
        target_control = normalized_output * (np.pi / 2.0)
        #small smoothing factor avoids demanding an abrupt physical jump.
        data.ctrl[:] = (0.7 * data.ctrl) + (0.3 * target_control)
        #respect actuatorlimits supplied by MuJoCo.
        if data.model.nu and data.model.actuator_ctrllimited.all():
            data.ctrl[:] = np.clip(
                data.ctrl,
                data.model.actuator_ctrlrange[:, 0],
                data.model.actuator_ctrlrange[:, 1],
            )


class SimulationEvaluator:
    """Build one fixed robot/world and evaluate controller genomes.
    """

    def __init__(self, duration: float = SIMULATION_DURATION):
        #clear everything left
        mujoco.set_mjcb_control(None)
        #setup
        self.world = SimpleFlatWorld()
        self.robot = gecko()
        self.world.spawn(
            self.robot.spec,
            position=SPAWN_POSITION,
            correct_collision_with_floor=True,
        )
        #compile the world
        self.model = self.world.spec.compile()
        self.data = mujoco.MjData(self.model)
        #find core geometry in the world for position data
        geoms = self.world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        core_geom = next(geom for geom in geoms if "core" in geom.name)
        self.core_binding = self.data.bind(core_geom)
        #store number of actuator outputs required
        self.output_size = self.model.nu
        #seconds of simulated time per evaluation
        self.duration = duration

    def evaluate(self, weights: np.ndarray) -> float:
        """Run one simulation and return final distance totarget."""

        #reset
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        #create controller
        controller = NeuralController(
            weights=weights,
            output_size=self.output_size,
            core_binding=self.core_binding,
            target=TARGET_POSITION,
        )

        thread_safe_runner(
            self.model,
            self.data,
            controller,
            duration=self.duration,
        )
        #compare only x and y
        final_position = np.asarray(self.core_binding.xpos, dtype=np.float64)
        return float(np.linalg.norm(final_position[:2] - TARGET_POSITION[:2]))


#random weights, used for the EA's first generation and for random search
def random_genome(rng, size, scale):
    return rng.normal(0.0, scale, size)


def make_individual(weights, sigma, parent_fitness=None):
    individual = Individual()
    individual.genotype = weights.astype(float).tolist()
    #the step size travels with the individual, the self-adaptive variant inherits it
    individual.tags = {"sigma": float(sigma), "parent_fitness": parent_fitness}
    return individual


#read an individuals JSON genome back as a numpy vector
def vector(individual):
    return np.asarray(individual.genotype, dtype=np.float64)


def evaluate_population(population, evaluator):
    #only candidates without a fitness need another simulation
    for individual in population:
        if individual.requires_eval:
            individual.fitness = evaluator.evaluate(vector(individual))


def tournament_parent(population, rng, tournament_size=3):
    #convert the population to a list so numpy can sample indices
    individuals = population.to_list()
    #without replacement
    indices = rng.choice(
        len(individuals),
        size=min(tournament_size, len(individuals)),
        replace=False,
    )
    #lowest score wins.
    return min((individuals[int(index)] for index in indices), key=lambda item: item.fitness)


#intermediate (arithmetic) recombination with a new random weight for every gene
def intermediate_crossover(parent_a, parent_b, rng):
    genome_a = vector(parent_a)
    genome_b = vector(parent_b)
    #one random mixing value per weight so every parameter can inherit differently from the two parents
    alpha = rng.random(genome_a.shape)

    child_a = alpha * genome_a + (1.0 - alpha) * genome_b
    child_b = alpha * genome_b + (1.0 - alpha) * genome_a
    return child_a, child_b


#add gaussian noise to a fraction of the weights
def mutate(weights, rng, sigma, mutation_rate=0.1):
    #which parameters are changed in this mutation event
    mask = rng.random(weights.shape) < mutation_rate
    noise = rng.normal(loc=0.0, scale=sigma, size=weights.shape)
    mutated = weights + mask * noise
    #bounds keep the search from producing too large networks
    return np.clip(mutated, -3.0, 3.0)


#step size for the fixed and annealed variants
def mutation_sigma(variant, generation, generations):
    if variant == "fixed_mutation":
        return FIXED_SIGMA
    #annealed: from 0.22 at the start down to 0.02 at the end of the run
    progress = generation / max(generations, 1)
    return 0.20 * (1.0 - progress) + 0.02


#mutate an individual's own step size, log-normal so it stays positive
def self_adaptive_sigma(sigma, rng, tau):
    return max(sigma * float(np.exp(tau * rng.normal())), MIN_SIGMA)


#average distance of the genomes to the population centre (0 means all the same)
def diversity(population):
    genomes = np.array([vector(individual) for individual in population])
    return float(np.mean(np.linalg.norm(genomes - genomes.mean(axis=0), axis=1)))


def record_population(history, population, variant, seed, generation, evaluations, children_better):
    fitnesses = [individual.fitness for individual in population]
    sigmas = [individual.tags["sigma"] for individual in population]
    history.append(
        {
            "variant": variant,
            "seed": seed,
            "generation": generation,
            "evaluations": evaluations,
            "best": float(min(fitnesses)),
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
            "worst": float(max(fitnesses)),
            "sigma_mean": float(np.mean(sigmas)),
            "sigma_std": float(np.std(sigmas)),
            "diversity": diversity(population),
            #share of last generation's children that beat their better parent
            "children_better": children_better,
        },
    )


def run_ea(
    variant: str,
    seed: int,
    evaluator: SimulationEvaluator,
    population_size: int,
    generations: int,
    init_scale: float = INIT_SCALE,
    crossover_probability: float = 0.8,
) -> list:
    """Run one neural-weight EA and return its generation history.
    """

    #reset
    rng = seed_everything(seed)
    size = genome_size(evaluator.output_size)
    #learning rate of the self-adaptive step size, 1/sqrt(n) as in the course
    tau = 1.0 / np.sqrt(size)
    if variant == "self_adaptive":
        first_sigma = START_SIGMA
    else:
        first_sigma = mutation_sigma(variant, 0, generations)
    initial = [
        make_individual(random_genome(rng, size, init_scale), first_sigma)
        for _ in range(population_size)
    ]
    population = Population(initial)
    #evaluate the initial generation and count those physics simulations
    evaluate_population(population, evaluator)
    evaluations = len(population)
    history = []
    children_better = None

    #log every generation, generation zero included
    for generation in range(generations + 1):
        record_population(history, population, variant, seed, generation, evaluations, children_better)
        if generation == generations:
            break
        #keep the best individual
        elite = min(population.to_list(), key=lambda item: item.fitness)
        #create enough children
        children = []
        while len(children) < population_size - 1:
            #select two parents
            parent_a = tournament_parent(population, rng)
            parent_b = tournament_parent(population, rng)
            #remember the better parent to check later if the child improved on it
            parent_fitness = min(parent_a.fitness, parent_b.fitness)
            sigma_a = parent_a.tags["sigma"]
            sigma_b = parent_b.tags["sigma"]
            #apply crossover
            if rng.random() < crossover_probability:
                child_a, child_b = intermediate_crossover(parent_a, parent_b, rng)
                #mixed children start from the average step size of both parents
                sigma_a = sigma_b = (sigma_a + sigma_b) / 2.0
            else:
                child_a, child_b = vector(parent_a).copy(), vector(parent_b).copy()
            if variant == "self_adaptive":
                #mutate the step size first, then the weights with the new step size
                sigma_a = self_adaptive_sigma(sigma_a, rng, tau)
                sigma_b = self_adaptive_sigma(sigma_b, rng, tau)
            else:
                sigma_a = sigma_b = mutation_sigma(variant, generation, generations)
            child_a = mutate(child_a, rng, sigma_a)
            child_b = mutate(child_b, rng, sigma_b)
            children.extend([
                make_individual(child_a, sigma_a, parent_fitness),
                make_individual(child_b, sigma_b, parent_fitness),
            ])
        #remove extra child when the population size is odd
        children = children[: population_size - 1]
        next_population = Population([elite, *children])
        #evaluate
        evaluate_population(next_population, evaluator)
        #count only the newly evaluated offspring
        evaluations += len(children)
        children_better = float(np.mean([
            child.fitness < child.tags["parent_fitness"] for child in children
        ]))
        #replace the old population
        population = next_population

    #save the best controller so it can be replayed or filmed later
    best = min(population.to_list(), key=lambda item: item.fitness)
    history[-1]["best_genome"] = best.genotype
    history[-1]["best_sigma"] = best.tags["sigma"]
    return history


def run_random_search(
    seed: int,
    evaluator: SimulationEvaluator,
    population_size: int,
    generations: int,
    init_scale: float = INIT_SCALE,
) -> list:
    """Run random controller-weight search at the EA's evaluation budget."""

    rng = seed_everything(seed)
    size = genome_size(evaluator.output_size)
    #exactly as many simulations as one EA run
    total_evaluations = evaluation_budget(population_size, generations)
    best = float("inf")
    best_weights = None
    block = []
    history = []

    for evaluation in range(1, total_evaluations + 1):
        weights = random_genome(rng, size, init_scale)
        fitness = evaluator.evaluate(weights)
        block.append(fitness)
        if fitness < best:
            best = fitness
            best_weights = weights
        #log at the same simulation counts as the EA generations
        if evaluation >= population_size and (evaluation - population_size) % (population_size - 1) == 0:
            history.append(
                {
                    "variant": "random_search",
                    "seed": seed,
                    "generation": (evaluation - population_size) // (population_size - 1),
                    "evaluations": evaluation,
                    #best so far, the mean/std/worst are of the samples since the last log line
                    "best": float(best),
                    "mean": float(np.mean(block)),
                    "std": float(np.std(block)),
                    "worst": float(max(block)),
                    "sigma_mean": None,
                    "sigma_std": None,
                    "diversity": None,
                    "children_better": None,
                },
            )
            block = []
    history[-1]["best_genome"] = best_weights.tolist()
    return history


#run one variant with one seed, every process builds its own simulation
def run_job(job):
    variant, seed, population_size, generations, duration, init_scale = job
    evaluator = SimulationEvaluator(duration)
    if variant == "random_search":
        rows = run_random_search(seed, evaluator, population_size, generations, init_scale)
    else:
        rows = run_ea(variant, seed, evaluator, population_size, generations, init_scale)
    print(f"completed {variant} seed={seed}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--duration", type=float, default=SIMULATION_DURATION)
    parser.add_argument("--init-scale", type=float, default=INIT_SCALE)
    #number of runs at the same time, every run uses its own seed so results do not change
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "history.jsonl")
    args = parser.parse_args()
    jobs = [
        (variant, seed, args.population, args.generations, args.duration, args.init_scale)
        for variant in args.variants
        for seed in args.seeds
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.workers) if args.workers > 1 else None
    results = pool.imap(run_job, jobs) if pool else map(run_job, jobs)
    with args.output.open("w", encoding="utf-8") as handle:
        for rows in results:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    if pool:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
