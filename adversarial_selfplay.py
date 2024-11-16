import random
from typing import List, Tuple
import time
import slimevolleygym
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
import gym
gym.logger.set_level(40)


class MAMLNetwork:
    def __init__(self, param_size=299):
        # Network architecture constants
        self.nGameInput = 12  # Full obs vector
        self.nGameOutput = 3  # 3 buttons (forward, backward, jump)
        self.nRecurrentState = 20  # Recurrent states for feedback
        self.nOutput = self.nGameOutput + self.nRecurrentState
        self.nInput = self.nGameInput + self.nOutput

        # Reward network architecture - single layer
        self.nRewardHidden = 12  # Reduced size

        # Calculate parameter sizes
        self.policy_params_size = (self.nInput * self.nOutput) + self.nOutput
        reward_input_size = self.nGameInput + self.nRecurrentState
        self.reward_params_size = (reward_input_size * 1) + 1  # Single layer weights + bias

        # Initialize all parameters (policy + reward)
        self.params = np.random.randn(self.policy_params_size + self.reward_params_size)

        # Split parameters between policy and reward networks
        self.policy_params = self.params[:self.policy_params_size]
        self.reward_params = self.params[self.policy_params_size:]

        # Initialize policy network
        self.weight = self.policy_params[:-self.nOutput].reshape(self.nOutput, self.nInput)
        self.bias = self.policy_params[-self.nOutput:]

        # Initialize reward network weights and biases
        reward_input_size = self.nGameInput + self.nRecurrentState
        self.reward_w = self.reward_params[:-1].reshape(reward_input_size, 1)
        self.reward_b = self.reward_params[-1:]

        # Store current states
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

        # Pre-allocate arrays for reward computation
        self.reward_input = np.zeros(self.nGameInput + self.nRecurrentState)

        # MAML specific
        self.inner_lr = 0.01
        self.trajectory_buffer = []

    def reset(self):
        """Complete reset between games"""
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)
        self.trajectory_buffer = []
        # Reset policy network
        self.weight = self.policy_params[:-self.nOutput].reshape(self.nOutput, self.nInput)
        self.bias = self.policy_params[-self.nOutput:]
        if hasattr(self, '_prev_obs'):
            del self._prev_obs
        if hasattr(self, '_prev_action'):
            del self._prev_action

    def _setInputState(self, obs):
        """Use full observation vector"""
        self.inputState[0:self.nGameInput] = obs
        self.inputState[self.nGameInput:] = self.outputState

    def _forward(self):
        """Forward pass through the policy network"""
        self.prevOutputState = self.outputState
        self.outputState = np.tanh(np.dot(self.weight, self.inputState) + self.bias)

    def _getAction(self):
        """Convert network output to discrete actions"""
        forward = int(self.outputState[0] > 0.75)
        backward = int(self.outputState[1] > 0.75)
        jump = int(self.outputState[2] > 0.75)
        return [forward, backward, jump]

    def forward(self, obs):
        """Full forward pass with experience storage and adaptation"""
        # Regular forward pass
        self._setInputState(obs)
        self._forward()
        action = self._getAction()

        # Store experience and adapt
        if hasattr(self, '_prev_obs') and hasattr(self, '_prev_action'):
            self.store_transition(
                self._prev_obs,
                self._prev_action,
                self._compute_reward(obs),
                obs
            )
            self.adapt_from_experience()

        # Store current state for next transition
        self._prev_obs = obs.copy()
        self._prev_action = action

        return action

    def store_transition(self, obs, action, reward, next_obs):
        """Store transition in replay buffer"""
        self.trajectory_buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs
        })
        if len(self.trajectory_buffer) > 10:  # Keep buffer small
            self.trajectory_buffer.pop(0)

    def compute_action_grads(self, desired_action):
        """Compute gradients for action adaptation"""
        current_output = self.outputState[:self.nGameOutput]
        action_grads = np.zeros(self.nOutput)
        action_grads[:self.nGameOutput] = (desired_action - current_output)

        pre_tanh = np.dot(self.weight, self.inputState) + self.bias
        tanh_deriv = 1 - np.tanh(pre_tanh)**2

        weight_grads = np.zeros_like(self.weight)
        weight_grads[:self.nGameOutput] = np.outer(
            action_grads[:self.nGameOutput] * tanh_deriv[:self.nGameOutput],
            self.inputState
        )
        bias_grads = np.zeros_like(self.bias)
        bias_grads[:self.nGameOutput] = action_grads[:self.nGameOutput] * \
            tanh_deriv[:self.nGameOutput]

        return weight_grads, bias_grads

    def adapt_from_experience(self):
        """Adapt policy using stored experience"""
        if len(self.trajectory_buffer) < 2:
            return

        good_transitions = [t for t in self.trajectory_buffer if t['reward'] > 0]
        if not good_transitions:
            return

        weight_update = np.zeros_like(self.weight)
        bias_update = np.zeros_like(self.bias)

        for transition in good_transitions:
            obs = transition['obs']
            action = transition['action']

            desired_output = np.array([
                1.0 if action[0] else -1.0,
                1.0 if action[1] else -1.0,
                1.0 if action[2] else -1.0
            ])

            self._setInputState(obs)
            w_grads, b_grads = self.compute_action_grads(desired_output)

            weight_update += w_grads
            bias_update += b_grads

        n_good = len(good_transitions)
        self.weight += self.inner_lr * (weight_update / n_good)
        self.bias += self.inner_lr * (bias_update / n_good)

    def _compute_reward(self, obs):
        """Optimized reward computation with single layer"""
        # Use pre-allocated array and in-place operations
        self.reward_input[:self.nGameInput] = obs
        self.reward_input[self.nGameInput:] = self.outputState[-self.nRecurrentState:]

        # Single layer
        return (np.dot(self.reward_input, self.reward_w) + self.reward_b)[0]

    def mutate(self, mutation_rate=0.1, mutation_std=0.1):
        """Mutate both policy and reward network parameters"""
        mask = np.random.random(len(self.params)) < mutation_rate
        noise = np.random.normal(0, mutation_std, len(self.params))
        self.params[mask] += noise[mask]

        # Update policy parameters
        self.policy_params = self.params[:self.policy_params_size]
        self.weight = self.policy_params[:-self.nOutput].reshape(self.nOutput, self.nInput)
        self.bias = self.policy_params[-self.nOutput:]

        # Update reward parameters
        self.reward_params = self.params[self.policy_params_size:]
        reward_input_size = self.nGameInput + self.nRecurrentState
        self.reward_w = self.reward_params[:-1].reshape(reward_input_size, 1)
        self.reward_b = self.reward_params[-1:]

        return self

    def crossover(self, other_network, crossover_rate=0.2):
        """Crossover both policy and reward network parameters"""
        mask = np.random.random(len(self.params)) < crossover_rate
        child_params = np.where(mask, self.params, other_network.params)

        child = MAMLNetwork()
        child.params = child_params

        # Update policy parameters
        child.policy_params = child.params[:child.policy_params_size]
        child.weight = child.policy_params[:-child.nOutput].reshape(child.nOutput, child.nInput)
        child.bias = child.policy_params[-child.nOutput:]

        # Update reward parameters
        child.reward_params = child.params[child.policy_params_size:]
        reward_input_size = child.nGameInput + child.nRecurrentState
        child.reward_w = child.reward_params[:-1].reshape(reward_input_size, 1)
        child.reward_b = child.reward_params[-1:]

        return child


def evaluate_match(networks: Tuple[MAMLNetwork, MAMLNetwork],
                   enable_rl: Tuple[bool, bool]) -> Tuple[float, float, int, int]:
    """
    Evaluate a match between two networks with configurable RL learning.

    Args:
        networks: Tuple of (right_net, left_net)
        enable_rl: Tuple of (right_rl_enabled, left_rl_enabled)
    """
    right_net, left_net = networks
    right_rl, left_rl = enable_rl
    env = gym.make('SlimeVolley-v0')
    env.multiagent = True
    env.reset()

    obs, _, _, info = env.step([0, 0, 0], [0, 0, 0])
    right_obs = obs
    left_obs = info['otherObs']

    right_net.reset()  # Reset adaptations
    left_net.reset()   # Reset adaptations

    done = False
    total_reward = 0
    crosses = 0
    frames = 0
    MAX_FRAMES = 1200

    while not done and frames < MAX_FRAMES:
        # Get actions using appropriate method based on RL flag
        left_action = left_net.forward(left_obs) if left_rl else left_net._getAction()
        right_action = right_net.forward(right_obs) if right_rl else right_net._getAction()

        prev_ball_pos = obs[4]
        obs, reward, done, info = env.step(right_action, left_action)
        curr_ball_pos = obs[4]

        right_obs = obs
        left_obs = info['otherObs']
        total_reward += reward
        frames += 1

        if np.sign([prev_ball_pos]) != np.sign([curr_ball_pos]) and not (
                prev_ball_pos == 0 or curr_ball_pos == 0):
            crosses += 1

    env.close()
    return total_reward, -total_reward, crosses, frames


def evaluate_network(eval_tuple: Tuple[MAMLNetwork, List[MAMLNetwork], int, float]):
    """
    Evaluate a network against multiple opponents with weighted static vs RL performance.

    Args:
        eval_tuple: (network, opponents, matches_per_opponent, rl_weight)
        rl_weight: Weight for RL performance (0.0 to 0.5)
    """
    network, opponents, matches_per_opponent, rl_weight = eval_tuple
    total_static_fitness = 0
    total_rl_fitness = 0
    total_crosses = 0
    total_frames = 0
    total_matches = 0

    for opponent in opponents:
        for _ in range(matches_per_opponent):
            # Game 1: Static vs Static
            static_right, _, crosses1, frames1 = evaluate_match(
                (network, opponent),
                enable_rl=(False, False)
            )
            # Game 2: RL vs Static
            rl_right, _, crosses2, frames2 = evaluate_match(
                (network, opponent),
                enable_rl=(True, False)
            )

            total_static_fitness += static_right
            total_rl_fitness += rl_right

            total_crosses += (crosses1 + crosses2)
            total_frames += (frames1 + frames2)
            total_matches += 1

    avg_static_fitness = total_static_fitness / total_matches
    avg_rl_fitness = total_rl_fitness / total_matches

    # Weight the fitness scores
    static_weight = 1.0 - rl_weight
    avg_fitness = (avg_static_fitness * static_weight) + (avg_rl_fitness * rl_weight)
    avg_crosses = total_crosses / (2 * total_matches)
    avg_frames = total_frames / (2 * total_matches)

    return avg_fitness, avg_static_fitness, avg_rl_fitness, avg_crosses, avg_frames


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

        self.population = [MAMLNetwork() for _ in range(population_size)]

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


def train_genetic_algorithm(num_generations=64, pop_size=512, workers=32):
    ga = GeneticAlgorithm(population_size=pop_size)

    with Pool(workers) as pool:
        try:
            for gen in range(num_generations):
                gen_st = time.time()

                # Calculate current RL weight (linear progression from 0.1 to 0.5)
                if gen < 64:
                    rl_weight = (0.5 * (gen / (64)))
                else:
                    rl_weight = 0.5

                # Create evaluation pairs
                eval_tasks = []
                for network in ga.population:
                    opponents = random.sample([n for n in ga.population if n != network],
                                              ga.opponents_per_eval)
                    eval_tasks.append((network, opponents, ga.evaluation_matches, rl_weight))

                # Evaluate population
                results = pool.map(evaluate_network, eval_tasks)
                fitness_scores = np.array([r[0] for r in results])
                static_scores = np.array([r[1] for r in results])
                rl_scores = np.array([r[2] for r in results])
                crosses = np.array([r[3] for r in results])
                frames = np.array([r[4] for r in results])

                # Print progress with detailed statistics
                best_idx = np.argmax(fitness_scores)
                print(f'Generation {gen}:')
                print(f'  Current RL Weight: {rl_weight:.3f}')
                print(f'  Overall Fitness:')
                print(f'    Best  = {fitness_scores[best_idx]:.3f}')
                print(f'    Avg   = {np.mean(fitness_scores):.3f}')
                print(f'  Static Performance:')
                print(f'    Best  = {static_scores[best_idx]:.3f}')
                print(f'    Avg   = {np.mean(static_scores):.3f}')
                print(f'  RL Performance:')
                print(f'    Best  = {rl_scores[best_idx]:.3f}')
                print(f'    Avg   = {np.mean(rl_scores):.3f}')
                print(f'  Game Stats:')
                print(f'    Crosses = {np.mean(crosses):.2f}')
                print(f'    Frames  = {np.mean(frames):.2f}')
                print(f'  Time: {time.time() - gen_st:.2f}s')
                print()

                # Evolve population
                ga.evolve(fitness_scores)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current population and best network...")

        # Final evaluation with equal weighting
        final_eval_tasks = [(n, random.sample(ga.population, ga.opponents_per_eval),
                            ga.evaluation_matches, 0.5) for n in ga.population]
        final_results = pool.map(evaluate_network, final_eval_tasks)
        final_fitness = np.array([r[0] for r in final_results])
        final_static = np.array([r[1] for r in final_results])
        final_rl = np.array([r[2] for r in final_results])

        # Save best network
        best_idx = np.argmax(final_fitness)
        best_network = ga.population[best_idx]
        with open("best_network.pkl", "wb") as f:
            pickle.dump(best_network, f)

        # Save entire population
        population_data = {
            'population': ga.population,
            'fitness_scores': final_fitness,
            'static_scores': final_static,
            'rl_scores': final_rl
        }
        with open("population.pkl", "wb") as f:
            pickle.dump(population_data, f)

        print(f"\nTraining complete!")
        print(f"Best network performance:")
        print(f"  Overall fitness: {final_fitness[best_idx]:.3f}")
        print(f"  Static fitness: {final_static[best_idx]:.3f}")
        print(f"  RL fitness:     {final_rl[best_idx]:.3f}")
        print(f"\nPopulation averages:")
        print(f"  Overall fitness: {np.mean(final_fitness):.3f}")
        print(f"  Static fitness: {np.mean(final_static):.3f}")
        print(f"  RL fitness:     {np.mean(final_rl):.3f}")
        print(f"\nSaved population to 'population.pkl' and best network to 'best_network.pkl'")
        return best_network


if __name__ == "__main__":
    best_solution = train_genetic_algorithm(
        num_generations=1024,
        pop_size=112 * 4,
        workers=112
    )
