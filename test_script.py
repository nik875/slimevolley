import gym
import slimevolleygym
import numpy as np


def print_state(step, obs, info):
    agent_x = obs[0]
    ball_x = obs[4]
    opponent_x = info['otherObs'][0]
    ball_x_from_opponent = info['otherObs'][4]

    print(f"\nStep {step}")
    print(f"{'='*50}")
    print(f"Agent perspective:")
    print(f"  My x position: {agent_x:.3f}")
    print(f"  Ball x position: {ball_x:.3f}")
    print(f"  Opponent x position: {opponent_x:.3f}")
    print(f"\nOpponent perspective:")
    print(f"  Their x position: {info['otherObs'][0]:.3f}")
    print(f"  Ball x position: {ball_x_from_opponent:.3f}")
    print(f"  Their opponent (me) x position: {info['otherObs'][8]:.3f}")


def test_coordinates():
    env = gym.make('SlimeVolley-v0')
    env.multiagent = True

    # Test both sides
    for episode in range(2):
        obs = env.reset()
        print(f"\nEPISODE {episode + 1}")
        print(f"{'='*50}")

        # Get initial state
        obs, reward, done, info = env.step([0, 0, 0], [0, 0, 0])

        # Print first few steps
        for i in range(5):
            print_state(i, obs, info)
            obs, reward, done, info = env.step([0, 0, 0], [1, 0, 0])

        env.close()

        # Swap sides by reinitializing environment
        if episode == 0:
            print("\nSwapping sides...\n")


if __name__ == "__main__":
    test_coordinates()
