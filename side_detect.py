import pickle
import numpy as np


class SmartAgent:
    def __init__(self, network):
        self.network = network
        self.reset()

        # Constants
        self.HIT_THRESHOLD = 2.2  # Slightly above ball.r + agent.r (0.5 + 1.5)
        self.WALL_THRESHOLD = 23  # Just inside the walls at ±24
        self.NET_THRESHOLD = 2    # Width around net to ignore hits

        # Track distances over multiple hits to ensure consistency
        self.hit_distances = []
        self.MIN_HITS_NEEDED = 3

    def reset(self):
        self.network.reset_hidden()
        self.prev_ball_vx = None
        self.prev_ball_x = None
        self.side_detected = False
        self.is_left_side = None
        self.hit_distances = []

    def reset_hidden(self):
        self.reset()

    def detect_side(self, obs):
        """
        Detect which side we're on based on ball hits, accounting for the environment's
        coordinate system bug where our x is inverted when we're on the left but ball_x isn't.
        """
        if self.side_detected:
            return

        x, _, _, _, ball_x, _, ball_vx, _, op_x, _, _, _ = obs

        # Initialize previous states on first call
        if self.prev_ball_vx is None:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Detect hit by checking for change in ball velocity direction
        velocity_changed_sign = (np.sign(ball_vx) != np.sign(self.prev_ball_vx))
        if not velocity_changed_sign:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Ignore hits near walls
        if abs(self.prev_ball_x) > self.WALL_THRESHOLD:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Ignore hits near net
        if abs(self.prev_ball_x) < self.NET_THRESHOLD:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Calculate both possible distances
        right_dist = abs(x - self.prev_ball_x)
        left_dist = abs(-x - self.prev_ball_x)

        # Store the distances for analysis
        self.hit_distances.append((right_dist, left_dist))

        # Only make determination after seeing enough hits
        if len(self.hit_distances) >= self.MIN_HITS_NEEDED:
            # Count hits where right distance is very small (< 0.2)
            right_side_hits = sum(1 for r, l in self.hit_distances if r < 0.2)
            # Count hits where left distance is around 2.0-2.2
            left_side_hits = sum(1 for r, l in self.hit_distances if 2.0 <= l <= 2.2)

            total_hits = len(self.hit_distances)

            # Only make determination if pattern is very clear
            if right_side_hits > total_hits * 0.8:
                self.is_left_side = True  # Consistent small right_dist means we're on left
                self.side_detected = True
                print(
                    f"Left side confirmed after {total_hits} hits - {right_side_hits} consistent hits")
            elif left_side_hits > total_hits * 0.8:
                self.is_left_side = False  # Consistent 2.1ish left_dist means we're on right
                self.side_detected = True
                print(
                    f"Right side confirmed after {total_hits} hits - {left_side_hits} consistent hits")

        # Keep last 10 hits to avoid overflow
        if len(self.hit_distances) > 10:
            self.hit_distances = self.hit_distances[-10:]

        self.prev_ball_vx = ball_vx
        self.prev_ball_x = ball_x

    def transform_observation(self, obs):
        """
        Transform the observation by inverting ball coordinates if we're on the left.
        """
        if not self.is_left_side:
            return obs

        # Create copy to avoid modifying original
        transformed = obs.copy()

        # Invert ball x position and velocity
        transformed[4] *= -1  # ball_x
        transformed[6] *= -1  # ball_vx

        return transformed

    def forward(self, obs):
        self.detect_side(obs)
        transformed_obs = self.transform_observation(obs)
        return self.network.forward(transformed_obs)


def load_smart_agent(filepath):
    with open(filepath, 'rb') as f:
        network = pickle.load(f)
    return SmartAgent(network)
