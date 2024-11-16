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
        self.VELOCITY_CHANGE_THRESHOLD = 0.1

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
        Detect which side we're on based only on our hits.
        When we're on the left side, our x coordinate is inverted but the ball's isn't.
        """
        if self.side_detected:
            return

        x, _, _, _, ball_x, _, ball_vx, _, _, _, _, _ = obs

        # Initialize previous states on first call
        if self.prev_ball_vx is None:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Check for ball hit
        velocity_change = abs(ball_vx - self.prev_ball_vx)
        direction_changed = (np.sign(ball_vx) != np.sign(self.prev_ball_vx))
        is_hit = (velocity_change > self.VELOCITY_CHANGE_THRESHOLD) and direction_changed

        if not is_hit:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Ignore hits near walls
        if abs(ball_x) > self.WALL_THRESHOLD:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Ignore hits near net
        if abs(ball_x) < self.NET_THRESHOLD:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Calculate our distance to the hit
        our_dist_to_hit = abs(abs(x) - abs(self.prev_ball_x))

        # Only look at hits we made
        if our_dist_to_hit < self.HIT_THRESHOLD:
            # If we're hitting the ball and our x is large positive (~2.2),
            # we must be on the left side since our x is inverted
            self.is_left_side = (x > 1.0)

            self.side_detected = True
            print(f"Side detected: {'left' if self.is_left_side else 'right'}")
            print(f"Our hit at x={self.prev_ball_x:.2f}")
            print(f"Agent x: {x:.2f}, Ball x: {self.prev_ball_x:.2f}")
            print(f"Agent distance: {our_dist_to_hit:.2f}")
            print(f"Velocity change: {velocity_change:.2f}")

        self.prev_ball_vx = ball_vx
        self.prev_ball_x = ball_x

    def transform_observation(self, obs):
        """
        Transform the observation by inverting ball coordinates if we're on the left.
        Our x coordinate is already inverted by the environment when we're on the left.
        """
        if not self.is_left_side:
            return obs

        # Create copy to avoid modifying original
        transformed = obs.copy()

        # Only invert ball x position and velocity since our position is already inverted
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
