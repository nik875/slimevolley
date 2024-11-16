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

    def reset(self):
        self.network.reset_hidden()
        self.prev_ball_vx = None
        self.prev_ball_x = None
        self.side_detected = False
        self.is_left_side = None

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

        # Key insight: x is already inverted if we're on the left side!
        # So we need to try both possibilities and see which makes sense

        # First possibility: we're on right (x is not inverted)
        right_dist = abs(x - self.prev_ball_x)

        # Second possibility: we're on left (x is inverted)
        # We need to un-invert x to get true distance
        left_dist = abs(-x - self.prev_ball_x)

        # If we're closer with normal x, we're on right
        # If we're closer with inverted x, we're on left
        potential_left_hit = left_dist < self.HIT_THRESHOLD
        potential_right_hit = right_dist < self.HIT_THRESHOLD

        # Only set side if exactly one distance indicates a hit
        if potential_left_hit and not potential_right_hit:
            self.is_left_side = True
            print(f"Left side hit detected - dist: {left_dist:.2f}")
            self.side_detected = True
        elif potential_right_hit and not potential_left_hit:
            self.is_left_side = False
            print(f"Right side hit detected - dist: {right_dist:.2f}")
            self.side_detected = True

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
