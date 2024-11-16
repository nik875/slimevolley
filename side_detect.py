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
        self.MIN_FRAMES = 3       # Minimum frames to wait before detecting hits

    def reset(self):
        self.network.reset_hidden()
        self.prev_ball_vx = None
        self.prev_ball_x = None
        self.side_detected = False
        self.is_left_side = None
        self.frame_count = 0

    def reset_hidden(self):
        self.reset()

    def detect_side(self, obs):
        if self.side_detected:
            return

        x, _, _, _, ball_x, _, ball_vx, _, op_x, _, _, _ = obs
        self.frame_count += 1

        # Initialize previous states on first frame
        if self.prev_ball_vx is None:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Wait for minimum frames to avoid false initial hit detection
        if self.frame_count < self.MIN_FRAMES:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Only proceed if ball velocity changed direction
        if not (np.sign(ball_vx) != np.sign(self.prev_ball_vx)):
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Skip if near walls or net
        if abs(self.prev_ball_x) > self.WALL_THRESHOLD or abs(
                self.prev_ball_x) < self.NET_THRESHOLD:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Compare our position (always positive) to absolute ball position
        our_dist = abs(x - abs(self.prev_ball_x))

        print(f"\nBall hit detected:")
        print(f"agent_x: {x:.2f}, ball_x: {self.prev_ball_x:.2f}")
        print(f"our_dist to abs(ball_x): {our_dist:.2f}")

        if our_dist < self.HIT_THRESHOLD:
            self.is_left_side = (x < 1.0)  # Left if we're near 0.20, Right if near 2.25
            self.side_detected = True
            print(f"DETERMINATION: {'Left' if self.is_left_side else 'Right'} side (our hit)")
        else:
            print("Opponent hit - ignoring for side detection")

        self.prev_ball_vx = ball_vx
        self.prev_ball_x = ball_x

    def transform_observation(self, obs):
        if not self.is_left_side:
            return obs

        transformed = obs.copy()
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
