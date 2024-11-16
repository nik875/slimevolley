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
        if self.side_detected:
            return

        x, _, _, _, ball_x, _, ball_vx, _, op_x, _, _, _ = obs

        # Initialize previous states on first call
        if self.prev_ball_vx is None:
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

        # Calculate distances for both players
        our_dist = abs(x - self.prev_ball_x)
        op_dist = abs(op_x - self.prev_ball_x)

        print(f"\nBall hit detected:")
        print(f"agent_x: {x:.2f}, op_x: {op_x:.2f}, ball_x: {self.prev_ball_x:.2f}")
        print(f"our_dist: {our_dist:.2f}, op_dist: {op_dist:.2f}")

        # Determine who hit the ball
        we_hit = our_dist < op_dist and our_dist < self.HIT_THRESHOLD

        if we_hit:
            # If we hit the ball, our x coordinate tells us our side
            # If x is near 0, we're left, if x is near 2, we're right
            self.is_left_side = abs(x) < 1.0
            self.side_detected = True
            print(f"DETERMINATION: {'Left' if self.is_left_side else 'Right'} side (our hit)")

        # If opponent hit, don't use it for side detection
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
