import pickle
import numpy as np


class SmartAgent:
    def __init__(self, network):
        self.network = network
        self.reset()

        # Constants
        self.HIT_THRESHOLD = 1  # Slightly above ball.r + agent.r (0.5 + 1.5)
        self.WALL_THRESHOLD = 23  # Just inside the walls at ±24
        self.NET_THRESHOLD = 2    # Width around net to ignore hits
        self.MIN_FRAMES = 3       # Minimum frames to wait before detecting hits
        self.prev_ball_vx = None
        self.prev_ball_x = None
        self.side_detected = False
        self.is_left_side = None
        self.frame_count = 0

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
        prev_ball_vx = self.prev_ball_vx
        prev_ball_x = self.prev_ball_x
        self.prev_ball_vx = ball_vx
        self.prev_ball_x = ball_x
        self.frame_count += 1

        # Initialize previous states on first frame
        if prev_ball_vx is None:
            return

        # Wait for minimum frames to avoid false initial hit detection
        if self.frame_count < self.MIN_FRAMES:
            return

        # Only proceed if ball velocity changed direction
        if np.sign(ball_vx) == np.sign(prev_ball_vx):
            return

        # Skip if near walls or net
        if abs(self.prev_ball_x) > self.WALL_THRESHOLD or abs(
                self.prev_ball_x) < self.NET_THRESHOLD:
            return

        # Compare our position (always positive) to absolute ball position
        our_dist = abs(x - abs(prev_ball_x))
        opp_dist = abs(op_x - abs(prev_ball_x))

        print(f"\nBall hit detected:")
        print(f"agent_x: {x:.2f}, ball_x: {prev_ball_x:.2f}, opp_x: {op_x:.2f}")
        print(f"our_dist to abs(ball_x): {our_dist:.2f}")
        print(f"opp_dist to abs(ball_x): {opp_dist:.2f}")

        if our_dist < self.HIT_THRESHOLD and opp_dist > self.HIT_THRESHOLD:
            self.is_left_side = np.sign(ball_x) != np.sign(x)
            self.side_detected = True
            print(f"DETERMINATION: {'Left' if self.is_left_side else 'Right'} side (our hit)")
        elif our_dist > self.HIT_THRESHOLD and opp_dist < self.HIT_THRESHOLD:
            self.is_left_side = np.sign(ball_x) == np.sign(x)
            self.side_detected = True
            print(f"DETERMINATION: {'Left' if self.is_left_side else 'Right'} side (opp hit)")
        else:
            print("Ambiguous hit - ignoring for side detection")

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
