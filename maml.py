from multiprocessing import Pool
import random
import os
from datetime import datetime
from typing import List, Tuple
import time
import slimevolleygym
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from ad_selfplay_orig import Network
import gym
gym.logger.set_level(40)


class MAMLRewardNetwork:
    def __init__(self, base_params=None, base_architecture=None):
        # Match original Network architecture
        self.base_architecture = {
            'nGameInput': 8,  # Only 8 filtered states used
            'nGameOutput': 3,  # 3 buttons (forward, backward, jump)
            'nRecurrentState': 4  # Match original network's recurrent states
        } if base_architecture is None else base_architecture

        # Total output includes both actions and recurrent state
        self.nBaseOutput = self.base_architecture['nGameOutput'] + \
            self.base_architecture['nRecurrentState']  # 7

        # Input includes filtered game input and full previous output
        self.nBaseInput = self.base_architecture['nGameInput'] + \
            self.nBaseOutput  # 8 + 7 = 15

        # Reward network dimensions - takes full state for richer reward signal
        self.nFullGameInput = 12  # Full game state
        self.nRewardHidden = 4    # Reward network's recurrent state

        # Initialize states
        self.reset()

        # Initialize base network parameters
        if base_params is not None:
            expected_params = (self.nBaseInput * self.nBaseOutput) + \
                self.nBaseOutput  # Should be 112
            actual_params = len(base_params)

            if actual_params != expected_params:
                raise ValueError(
                    f"Parameter count mismatch! Expected {expected_params} but got {actual_params}."
                )

            self.base_params = base_params.copy()
        else:
            n_base_params = (self.nBaseInput * self.nBaseOutput) + self.nBaseOutput
            self.base_params = np.random.randn(n_base_params) * 0.1

        # Set up base network weights and biases
        n_weights = self.nBaseInput * self.nBaseOutput
        self.base_weight = self.base_params[:-
                                            self.nBaseOutput].reshape(self.nBaseOutput, self.nBaseInput)
        self.base_bias = self.base_params[-self.nBaseOutput:]

        # Initialize reward network - takes full state + recurrent states
        reward_input_size = self.nFullGameInput + \
            self.base_architecture['nRecurrentState'] + self.nRewardHidden
        reward_output_size = 1 + self.nRewardHidden  # Reward value + new hidden state

        # Initialize reward network weights with smaller values
        self.reward_w = np.random.randn(
            reward_output_size, reward_input_size) * 0.01
        self.reward_b = np.random.randn(reward_output_size) * 0.01

        # Store reward network parameters
        self.reward_params = np.concatenate([
            self.reward_w.flatten(),
            self.reward_b.flatten()
        ])

        # Initialize learning rate as an evolvable parameter
        # Use sigmoid to keep it between 0 and 1
        # Initialize learning rate parameter more conservatively
        self.lr_param = np.random.randn() - 4  # Center around -4 instead of 0
        self.inner_lr = self._sigmoid(self.lr_param) * 0.001  # Scale to 0-0.001 instead of 0-0.1

        # Combined parameter array for evolution - now includes learning rate
        self.params = np.concatenate([
            self.base_params,
            self.reward_params,
            [self.lr_param]  # Add learning rate parameter
        ])

    def _sigmoid(self, x):
        """Sigmoid function to bound learning rate"""
        return 1 / (1 + np.exp(-x))

    def reset(self):
        """Reset all episode-specific states"""
        self.base_input_state = np.zeros(self.nBaseInput)
        self.base_output_state = np.zeros(self.nBaseOutput)
        self.base_prev_output_state = np.zeros(self.nBaseOutput)
        self.reward_hidden_state = np.zeros(self.nRewardHidden)
        self.reward_prev_hidden_state = np.zeros(self.nRewardHidden)

        # Clear adaptation tracking
        if hasattr(self, '_prev_obs'):
            del self._prev_obs
        if hasattr(self, '_prev_action'):
            del self._prev_action
        if hasattr(self, '_prev_base_output'):
            del self._prev_base_output

    def _base_network_forward(self, obs):
        """Forward pass through base network with filtered inputs"""
        # Extract only the 8 used observations, matching original network
        filtered_obs = np.array([
            obs[0],  # x
            obs[1],  # y
            obs[2],  # vx
            obs[3],  # vy
            obs[4],  # ball_x
            obs[5],  # ball_y
            obs[6],  # ball_vx
            obs[7]   # ball_vy
        ])

        # Set filtered game input portion
        self.base_input_state[:self.base_architecture['nGameInput']] = filtered_obs

        # Set full previous output as input (both actions and recurrent states)
        self.base_input_state[self.base_architecture['nGameInput']:] = self.base_output_state

        # Store previous output state before update
        self.base_prev_output_state = self.base_output_state.copy()

        # Forward pass
        self.base_output_state = np.tanh(
            np.dot(self.base_weight, self.base_input_state) + self.base_bias
        )

        # Convert to action
        game_outputs = self.base_output_state[:self.base_architecture['nGameOutput']]
        return [
            int(game_outputs[0] > 0.75),
            int(game_outputs[1] > 0.75),
            int(game_outputs[2] > 0.75)
        ]

    def _compute_reward(self, obs):
        """Compute dense reward signal using full observation and both hidden states"""
        # Combine full game input, base hidden state, and reward hidden state
        reward_input = np.concatenate([
            obs,  # Full 12-dim game state
            # Base network's recurrent state
            self.base_output_state[self.base_architecture['nGameOutput']:],
            self.reward_hidden_state  # Reward's own hidden state
        ])

        # Forward pass through reward network
        reward_output = np.tanh(np.dot(self.reward_w, reward_input) + self.reward_b)

        # Split into reward value and new hidden state
        self.reward_prev_hidden_state = self.reward_hidden_state.copy()
        self.reward_hidden_state = reward_output[1:]  # Update hidden state
        return float(reward_output[0])  # Return reward value

    def _adapt_base_network(self, reward_signal):
        """Adapt base network parameters using reward signal"""
        # Compute gradients for base network parameters
        action_grads = np.zeros_like(self.base_prev_output_state)

        # Only apply gradients to action outputs
        action_grads[:self.base_architecture['nGameOutput']] = \
            (np.array(self._prev_action, dtype=float) -
             self.base_prev_output_state[:self.base_architecture['nGameOutput']]) * reward_signal

        # Backpropagate through tanh
        output_grads = action_grads * (1 - self.base_prev_output_state**2)

        # Compute weight and bias gradients
        weight_grads = np.outer(output_grads, self.base_input_state)

        # Update parameters
        self.base_weight += self.inner_lr * weight_grads
        self.base_bias += self.inner_lr * output_grads

        # Update parameter arrays
        self.base_params = np.concatenate([
            self.base_weight.flatten(),
            self.base_bias.flatten()
        ])
        self.params = np.concatenate([self.base_params, self.reward_params])

    def forward(self, obs):
        """Forward pass with continuous adaptation"""
        # Get action from current base network parameters
        action = self._base_network_forward(obs)

        # Get continuous reward signal from reward network
        reward_signal = self._compute_reward(obs)

        # Adapt base network if we have previous experience
        if hasattr(self, '_prev_obs'):
            self._adapt_base_network(reward_signal)

        # Store current state for next frame's adaptation
        self._prev_obs = obs.copy()
        self._prev_action = action
        self._prev_base_output = self.base_output_state.copy()

        return action

    def mutate(self, mutation_rate=0.1, mutation_std=0.1):
        """Mutate both base and reward parameters"""
        mask = np.random.random(len(self.params)) < mutation_rate
        noise = np.random.normal(0, mutation_std, len(self.params))
        self.params[mask] += noise[mask]
        self._update_all_params()
        return self

    def crossover(self, other_network, crossover_rate=0.2):
        """Crossover both base and reward parameters"""
        mask = np.random.random(len(self.params)) < crossover_rate
        child_params = np.where(mask, self.params, other_network.params)

        child = MAMLRewardNetwork(
            base_architecture=self.base_architecture,
        )
        child.params = child_params
        child._update_all_params()
        return child

    def _update_all_params(self):
        """Update all parameters including learning rate from flat parameter array"""
        # Split parameters
        n_base_params = (self.nBaseInput * self.nBaseOutput) + self.nBaseOutput
        self.base_params = self.params[:n_base_params]

        # Get reward network parameters
        reward_input_size = self.nFullGameInput + \
            self.base_architecture['nRecurrentState'] + self.nRewardHidden
        reward_output_size = 1 + self.nRewardHidden
        n_reward_params = (reward_input_size * reward_output_size) + reward_output_size

        self.reward_params = self.params[n_base_params:-1]  # Exclude learning rate
        self.lr_param = self.params[-1]  # Get learning rate parameter

        # Update learning rate through sigmoid
        self.inner_lr = self._sigmoid(self.lr_param) * 0.1  # Scale to 0-0.1 range

        # Update base network
        n_base_weights = self.nBaseInput * self.nBaseOutput
        self.base_weight = self.base_params[:-self.nBaseOutput].reshape(
            self.nBaseOutput, self.nBaseInput)
        self.base_bias = self.base_params[-self.nBaseOutput:]

        # Update reward network
        n_reward_weights = reward_input_size * reward_output_size
        self.reward_w = self.reward_params[:n_reward_weights].reshape(
            reward_output_size, reward_input_size)
        self.reward_b = self.reward_params[n_reward_weights:]


def expand_population(current_population: list, target_size: int) -> list:
    """Expand population to target size by creating variants of existing individuals"""
    if target_size <= len(current_population):
        return current_population

    expanded_population = current_population.copy()

    while len(expanded_population) < target_size:
        parent = random.choice(current_population)

        # Create new network with same architecture
        child = MAMLRewardNetwork(
            base_architecture=parent.base_architecture,
        )

        # Copy all parameters and mutate
        child.params = parent.params.copy()
        child.mutate(mutation_rate=0.2, mutation_std=0.1)
        child._update_all_params()

        expanded_population.append(child)

    return expanded_population


def create_maml_population(base_population_path: str) -> List[MAMLRewardNetwork]:
    """Create initial MAML population from existing population"""
    with open(base_population_path, "rb") as f:
        population_data = pickle.load(f)

    if isinstance(population_data, dict):
        base_networks = population_data['population']
    else:
        base_networks = population_data

    maml_population = []
    for base_net in base_networks:
        # Extract base network parameters
        base_params = np.concatenate([
            base_net.weight.flatten(),
            base_net.bias.flatten()
        ])

        # Create MAML network initialized with base network parameters
        maml_net = MAMLRewardNetwork(
            base_params=base_params,
        )
        maml_population.append(maml_net)

    return maml_population


class OpponentPool:
    """Maintains the full pool of traditionally evolved networks"""

    def __init__(self, population: List):
        self.base_networks = population

    def get_random_opponents(self, n: int) -> List[Network]:
        """Get n random opponents from the full pool"""
        return random.sample(self.base_networks, min(n, len(self.base_networks)))


def evaluate_match(match_tuple: Tuple[MAMLRewardNetwork, Network]) -> Tuple[float, int, int]:
    """Evaluate a match between a MAML network and a traditional opponent"""
    maml_net, base_opponent = match_tuple

    env = gym.make('SlimeVolley-v0')
    env.multiagent = True
    env.reset()

    obs, _, _, info = env.step([0, 0, 0], [0, 0, 0])
    maml_obs = obs
    opp_obs = info['otherObs']

    maml_net.reset()

    done = False
    total_reward = 0
    crosses = 0
    frames = 0
    MAX_FRAMES = 3000  # Increased to allow for longer adaptation

    while not done and frames < MAX_FRAMES:
        maml_action = maml_net.forward(maml_obs)
        opp_action = base_opponent.forward(opp_obs)

        prev_ball_pos = obs[4]
        obs, reward, done, info = env.step(maml_action, opp_action)
        curr_ball_pos = obs[4]

        maml_obs = obs
        opp_obs = info['otherObs']
        total_reward += reward
        frames += 1

        if np.sign([prev_ball_pos]) != np.sign([curr_ball_pos]) and not (
                prev_ball_pos == 0 or curr_ball_pos == 0):
            crosses += 1

    env.close()
    return total_reward, crosses, frames


def evaluate_network(eval_tuple: Tuple[MAMLRewardNetwork, List[Network], int]):
    """Evaluate a MAML network against multiple fixed opponents"""
    maml_network, opponents, matches_per_opponent = eval_tuple
    total_fitness = 0
    total_crosses = 0
    total_frames = 0
    total_matches = 0

    for opponent in opponents:
        for _ in range(matches_per_opponent):
            reward, crosses, frames = evaluate_match(
                (maml_network, opponent)
            )

            total_fitness += reward
            total_crosses += crosses
            total_frames += frames
            total_matches += 1

    avg_fitness = total_fitness / total_matches
    avg_crosses = total_crosses / total_matches
    avg_frames = total_frames / total_matches

    return avg_fitness, avg_crosses, avg_frames


class GeneticAlgorithm:
    def __init__(self,
                 opponent_pool: OpponentPool,
                 population_size=1024,
                 mutation_rate=0.1,
                 mutation_std=0.1,
                 elite_size=32,
                 evaluation_matches=1,  # Reduced matches per opponent
                 opponents_per_eval=10):  # Reduced number of opponents
        self.opponent_pool = opponent_pool
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elite_size = elite_size
        self.evaluation_matches = evaluation_matches
        self.opponents_per_eval = opponents_per_eval

        self.population = [MAMLRewardNetwork() for _ in range(population_size)]

    def select_parents(self, fitness_scores):
        tournament_size = 3
        selected = []

        for _ in range(2):
            idx = np.random.choice(len(fitness_scores), tournament_size)
            winner_idx = idx[np.argmax(fitness_scores[idx])]
            selected.append(self.population[winner_idx])

        return selected[0], selected[1]

    def evolve(self, fitness_scores):
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


def load_traditional_networks(population_dir: str) -> tuple:
    """Load just the traditional networks without converting to MAML"""
    with open(os.path.join(population_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    network_files = [f for f in os.listdir(population_dir) if f.startswith("network_rank")]
    network_files.sort(key=lambda x: int(x.split("rank")[1].split("_")[0]))

    traditional_networks = []
    for network_file in network_files:
        with open(os.path.join(population_dir, network_file), "rb") as f:
            network = pickle.load(f)
            traditional_networks.append(network)

    return traditional_networks, metadata


def convert_to_maml(traditional_networks: list, target_size: int = None) -> list:
    """Convert traditional networks to MAML networks and optionally expand population"""
    population = []
    for base_net in traditional_networks:
        base_params = np.concatenate([
            base_net.weight.flatten(),
            base_net.bias.flatten()
        ])
        maml_net = MAMLRewardNetwork(base_params=base_params)
        population.append(maml_net)

    if target_size and target_size > len(population):
        population = expand_population(population, target_size)

    return population


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


def train_genetic_algorithm(resume_from: str,
                            num_generations=64,
                            pop_size=512,
                            workers=32,
                            start_gen=0):
    if resume_from.endswith(".pkl"):
        with open(resume_from, "rb") as f:
            data = pickle.load(f)
        population = data["population"]
        metadata = data[[i for i in data.keys() if i != "population"]]
        # Need to load traditional networks separately for opponents
        traditional_networks = load_traditional_networks(resume_from)
    else:
        # Load and keep traditional networks separate from MAML population
        traditional_networks, metadata = load_traditional_networks(resume_from)
        population = convert_to_maml(traditional_networks, target_size=pop_size)

    # Initialize opponent pool with all networks
    opponent_pool = OpponentPool(traditional_networks)
    ga = GeneticAlgorithm(opponent_pool, population_size=pop_size)
    ga.population = population
    print(f"Loaded population from {resume_from}")

    with Pool(workers) as pool:
        try:
            for gen in range(start_gen, num_generations):
                gen_st = time.time()

                # Create evaluation tasks - each network gets different random opponents
                eval_tasks = []
                for network in ga.population:
                    opponents = opponent_pool.get_random_opponents(ga.opponents_per_eval)
                    eval_tasks.append((network, opponents, ga.evaluation_matches))

                # Evaluate population
                results = pool.map(evaluate_network, eval_tasks)
                fitness_scores = np.array([r[0] for r in results])
                crosses = np.array([r[1] for r in results])
                frames = np.array([r[2] for r in results])

                # Get elite indices and compute their statistics
                elite_indices = np.argsort(fitness_scores)[-ga.elite_size:]
                elite_fitness = fitness_scores[elite_indices]
                elite_crosses = crosses[elite_indices]
                elite_frames = frames[elite_indices]

                # Print progress with elite statistics
                print(f'Generation {gen}:')
                print(f'  Elite Performance vs Traditional:')
                print(f'    Elite Avg   = {np.mean(elite_fitness):.3f}')
                print(f'    Elite Std   = {np.std(elite_fitness):.3f}')
                print(f'  Population Performance:')
                print(f'    Pop Avg     = {np.mean(fitness_scores):.3f}')
                print(f'  Elite Game Stats:')
                print(
                    f'    Crosses     = {np.mean(elite_crosses):.2f} ± {np.std(elite_crosses):.2f}')
                print(f'    Frames      = {np.mean(elite_frames):.2f} ± {np.std(elite_frames):.2f}')
                print(f'  Time: {time.time() - gen_st:.2f}s')
                print()

                # Evolve population
                ga.evolve(fitness_scores)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving final state...")

        # Save final state using elite statistics
        elite_indices = np.argsort(fitness_scores)[-ga.elite_size:]
        elite_fitness = fitness_scores[elite_indices]
        best_network = ga.population[elite_indices[-1]]

        with open("best_network.pkl", "wb") as f:
            pickle.dump(best_network, f)

        population_data = {
            'population': ga.population,
            'fitness_scores': fitness_scores,
            'num_generations': gen + 1,
            'population_size': pop_size,
            'total_opponents': len(opponent_pool.base_networks),
            'elite_performance': {
                'mean_fitness': float(np.mean(elite_fitness)),
                'std_fitness': float(np.std(elite_fitness)),
                'size': ga.elite_size
            }
        }
        with open("final_population.pkl", "wb") as f:
            pickle.dump(population_data, f)

        print(f"\nTraining complete!")
        print(f"Elite performance vs traditional opponents:")
        print(f"  Fitness: {np.mean(elite_fitness):.3f} ± {np.std(elite_fitness):.3f}")
        print(f"\nPopulation average:")
        print(f"  Fitness: {np.mean(fitness_scores):.3f}")
        return best_network


if __name__ == "__main__":
    best_solution = train_genetic_algorithm(
        resume_from="population_oldmodel_1000gen",
        num_generations=2048,
        pop_size=112 * 10,
        workers=112
    )
