import sys
import signal
from datetime import datetime
import os
import random
from typing import List, Tuple
import time
import slimevolleygym
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
import gym
gym.logger.set_level(40)


class Network:
    def __init__(self, param_size=None):
        # Network architecture constants matching baseline
        # Full state: [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy]
        self.nGameInput = 12
        self.nGameOutput = 3  # 3 buttons (forward, backward, jump)
        self.nRecurrentState = 4  # extra recurrent states for feedback
        self.nOutput = self.nGameOutput + self.nRecurrentState
        self.nInput = self.nGameInput + self.nOutput

        # Calculate param_size if not provided
        if param_size is None:
            param_size = (self.nInput * self.nOutput) + self.nOutput  # weights + biases

        # Initialize network parameters randomly if none provided
        self.params = np.random.randn(param_size)

        # store current inputs and outputs
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

        # Split parameters into weights and biases
        self.weight = self.params[:-self.nOutput].reshape(self.nOutput, self.nInput)
        self.bias = self.params[-self.nOutput:]

    def reset_hidden(self):
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

    def _setInputState(self, obs):
        # Use full observation vector directly
        self.inputState[0:self.nGameInput] = np.array(obs)
        self.inputState[self.nGameInput:] = self.outputState

    def _forward(self):
        self.prevOutputState = self.outputState
        self.outputState = np.tanh(np.dot(self.weight, self.inputState) + self.bias)

    def _getAction(self):
        forward = 0
        backward = 0
        jump = 0

        if self.outputState[0] > 0.75:
            forward = 1
        if self.outputState[1] > 0.75:
            backward = 1
        if self.outputState[2] > 0.75:
            jump = 1

        return [forward, backward, jump]

    def forward(self, obs):
        self._setInputState(obs)
        self._forward()
        return self._getAction()

    def mutate(self, mutation_rate=0.1, mutation_std=0.1):
        mask = np.random.random(len(self.params)) < mutation_rate
        noise = np.random.normal(0, mutation_std, len(self.params))
        self.params[mask] += noise[mask]

        self.weight = self.params[:-self.nOutput].reshape(self.nOutput, self.nInput)
        self.bias = self.params[-self.nOutput:]

        return self

    def crossover(self, other_network, crossover_rate=0.2):
        mask = np.random.random(len(self.params)) < crossover_rate
        child_params = np.where(mask, self.params, other_network.params)

        child = Network(param_size=len(self.params))
        child.params = child_params
        child.weight = child.params[:-child.nOutput].reshape(child.nOutput, child.nInput)
        child.bias = child.params[-child.nOutput:]

        return child


def evaluate_match(networks: Tuple[Network, Network]) -> Tuple[float, float, int, int]:
    """Evaluate a single match between two networks"""
    right_net, left_net = networks
    env = gym.make('SlimeVolley-v0')
    env.multiagent = True
    env.reset()

    obs, _, _, info = env.step([0, 0, 0], [0, 0, 0])
    right_obs = obs
    left_obs = info['otherObs']

    right_net.reset_hidden()
    left_net.reset_hidden()

    done = False
    total_reward = 0
    crosses = 0
    frames = 0
    MAX_FRAMES = 1200

    while not done and frames < MAX_FRAMES:
        left_action = left_net.forward(left_obs)
        right_action = right_net.forward(right_obs)

        prev_ball_pos = obs[4]
        obs, reward, done, info = env.step(right_action, left_action)
        curr_ball_pos = obs[4]

        right_obs = obs
        left_obs = info['otherObs']
        total_reward += reward
        frames += 1

        # Count ball crosses
        ignore = prev_ball_pos == 0 or curr_ball_pos == 0
        if np.sign([prev_ball_pos]) != np.sign([curr_ball_pos]) and not ignore:
            crosses += 1

    env.close()
    return total_reward, -total_reward, crosses, frames


def evaluate_network(eval_tuple: Tuple[Network, List[Network], int]):
    """Evaluate a network against multiple opponents"""
    network, opponents, matches_per_opponent = eval_tuple
    total_fitness = 0
    total_crosses = 0
    total_frames = 0
    total_matches = 0

    # Play against each opponent multiple times
    for opponent in opponents:
        for _ in range(matches_per_opponent):
            # Randomly assign sides
            if random.random() < 0.5:
                right_reward, _, crosses, frames = evaluate_match((network, opponent))
                total_fitness += right_reward
            else:
                _, left_reward, crosses, frames = evaluate_match((opponent, network))
                total_fitness += left_reward
            total_crosses += crosses
            total_frames += frames
            total_matches += 1

    # Return average fitness, crosses, and frames
    return (total_fitness / total_matches,
            total_crosses / total_matches,
            total_frames / total_matches)


class GeneticAlgorithm:
    def __init__(self,
                 population_size=1024,
                 mutation_rate=0.1,
                 mutation_std=0.1,
                 elite_size=32,
                 evaluation_matches=10,
                 opponents_per_eval=5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elite_size = elite_size
        self.evaluation_matches = evaluation_matches
        self.opponents_per_eval = opponents_per_eval

        self.population = [Network() for _ in range(population_size)]

    def select_parents(self, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        selected = []

        for _ in range(2):
            idx = np.random.choice(len(fitness_scores), tournament_size)
            winner_idx = idx[np.argmax(fitness_scores[idx])]
            selected.append(self.population[winner_idx])

        return selected[0], selected[1]

    def evolve(self, fitness_scores):
        """Create next generation through evolution"""
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]

        new_population = sorted_population[:self.elite_size]

        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_scores)
            child = parent1.crossover(parent2)
            child.mutate(self.mutation_rate, self.mutation_std)
            new_population.append(child)

        self.population = new_population
        return self.population


gym.logger.set_level(40)


def expand_population(current_population: list, target_size: int) -> list:
    """Expand population to target size by creating variants of existing individuals"""
    if target_size <= len(current_population):
        return current_population

    expanded_population = current_population.copy()

    # Keep adding new variants until we reach target size
    while len(expanded_population) < target_size:
        # Randomly select a parent from existing population
        parent = random.choice(current_population)

        # Create a new network with the same architecture
        child = Network(param_size=len(parent.params))

        # Copy parent's parameters with some mutation
        child.params = parent.params.copy()
        child.mutate(mutation_rate=0.2, mutation_std=0.1)  # Higher mutation for diversity

        expanded_population.append(child)

    return expanded_population


def load_population(population_dir: str, target_size: int = None) -> tuple:
    """Load a saved population and optionally resize it"""
    # Load metadata
    with open(os.path.join(population_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    # Get all network files sorted by rank
    network_files = [f for f in os.listdir(population_dir) if f.startswith("network_rank")]
    network_files.sort(key=lambda x: int(x.split("rank")[1].split("_")[0]))

    # Load all networks
    population = []
    for network_file in network_files:
        with open(os.path.join(population_dir, network_file), "rb") as f:
            network = pickle.load(f)
            population.append(network)

    # Expand population if target_size is specified
    if target_size and target_size > len(population):
        print(f"Expanding population from {len(population)} to {target_size} individuals...")
        population = expand_population(population, target_size)

    return population, metadata


def save_population(population: list, fitness_scores: np.ndarray, num_generations: int):
    """Save the current population and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"population_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "population_size": len(population),
        "num_generations": num_generations,
        "final_fitness_scores": fitness_scores.tolist(),
        "best_fitness": float(np.max(fitness_scores)),
        "average_fitness": float(np.mean(fitness_scores)),
        "max_frames": 1200
    }
    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    # Save best network separately
    best_idx = np.argmax(fitness_scores)
    with open("best_network.pkl", "wb") as f:
        pickle.dump(population[best_idx], f)

    # Save all networks
    sorted_indices = np.argsort(fitness_scores)[::-1]
    for rank, idx in enumerate(sorted_indices):
        network = population[idx]
        fitness = fitness_scores[idx]
        filename = f"network_rank{rank:04d}_fitness{fitness:.3f}.pkl"
        with open(os.path.join(save_dir, filename), "wb") as f:
            pickle.dump(network, f)

    print(f"\nPopulation saved in directory: {save_dir}")
    print(f"Best network saved as: best_network.pkl")


def train_genetic_algorithm(num_generations=10, pop_size=512, workers=32,
                            resume_from=None, start_gen=0):
    if resume_from:
        population, metadata = load_population(resume_from, target_size=pop_size)
        ga = GeneticAlgorithm(population_size=pop_size)
        ga.population = population
        print(f"Loaded population from {resume_from}")
        print(f"Previous best fitness: {metadata['best_fitness']}")
        if len(population) > metadata['population_size']:
            print(
                f"Population expanded from {metadata['population_size']} to {len(population)} individuals")
    else:
        ga = GeneticAlgorithm(population_size=pop_size)

    try:
        with Pool(workers) as pool:
            for gen in range(start_gen, num_generations):
                gen_st = time.time()

                # Create evaluation pairs
                eval_tasks = []
                for network in ga.population:
                    opponents = random.sample([n for n in ga.population if n != network],
                                              ga.opponents_per_eval)
                    eval_tasks.append((network, opponents, ga.evaluation_matches))

                # Evaluate population
                results = pool.map(evaluate_network, eval_tasks)
                fitness_scores, crosses, frames = zip(*results)
                fitness_scores = np.array(fitness_scores)
                crosses = np.array(crosses)
                frames = np.array(frames)

                # Print progress
                best_fitness = np.max(fitness_scores)
                avg_fitness = np.mean(fitness_scores)
                avg_crosses = np.mean(crosses)
                avg_frames = np.mean(frames)
                print(f'Generation {gen}: Best fitness = {best_fitness:.3f}, ' +
                      f'Avg fitness = {avg_fitness:.3f}, Avg crosses = {avg_crosses:.1f}, ' +
                      f'Avg frames = {avg_frames:.1f}, Time: {time.time() - gen_st:.2f}s')

                # Evolve population
                ga.evolve(fitness_scores)

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current population...")
        save_population(ga.population, fitness_scores, gen + 1)
        sys.exit(0)

    # Final evaluation and save
    with Pool(workers) as pool:
        final_eval_tasks = [(n, random.sample(ga.population, ga.opponents_per_eval),
                            ga.evaluation_matches) for n in ga.population]
        final_results = pool.map(evaluate_network, final_eval_tasks)
        final_fitness, final_crosses, final_frames = zip(*final_results)
        final_fitness = np.array(final_fitness)
        final_frames = np.array(final_frames)

    save_population(ga.population, final_fitness, num_generations)

    print(f"\nFinal Statistics:")
    print(f"Best fitness: {np.max(final_fitness):.3f}")
    print(f"Average fitness: {np.mean(final_fitness):.3f}")
    print(f"Average frames per game: {np.mean(final_frames):.1f}")

    return ga.population[np.argmax(final_fitness)]


if __name__ == "__main__":
    best_solution = train_genetic_algorithm(
        num_generations=1024,  # Increased generations
        pop_size=112 * 10,        # Double the population size
        workers=112,
    )
