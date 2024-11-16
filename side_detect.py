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
        Detect which side we're on based on ball hits, with detailed debugging.
        """
        if self.side_detected:
            return

        x, _, _, _, ball_x, _, ball_vx, _, op_x, _, _, _ = obs

        print("\n=== New Frame ===")
        print(f"Current state:")
        print(f"x: {x:.2f}, ball_x: {ball_x:.2f}, ball_vx: {ball_vx:.2f}")
        print(f"prev_ball_x: {self.prev_ball_x if self.prev_ball_x is not None else 'None'}")
        print(f"prev_ball_vx: {self.prev_ball_vx if self.prev_ball_vx is not None else 'None'}")

        # Initialize previous states on first call
        if self.prev_ball_vx is None:
            print("Initializing previous states")
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            return

        # Detect hit by checking for change in ball velocity direction
        velocity_changed_sign = (np.sign(ball_vx) != np.sign(self.prev_ball_vx))
        print(f"\nHit detection:")
        print(f"Ball vx changed from {self.prev_ball_vx:.2f} to {ball_vx:.2f}")
        print(f"Velocity sign changed: {velocity_changed_sign}")

        if not velocity_changed_sign:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            print("No velocity sign change - updating previous states and returning")
            return

        # Wall hit check
        near_wall = abs(self.prev_ball_x) > self.WALL_THRESHOLD
        print(f"\nWall check:")
        print(f"Ball x position: {self.prev_ball_x:.2f}")
        print(f"Wall threshold: {self.WALL_THRESHOLD}")
        print(f"Near wall: {near_wall}")

        if near_wall:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            print("Hit was near wall - updating previous states and returning")
            return

        # Net hit check
        near_net = abs(self.prev_ball_x) < self.NET_THRESHOLD
        print(f"\nNet check:")
        print(f"Ball x position: {self.prev_ball_x:.2f}")
        print(f"Net threshold: {self.NET_THRESHOLD}")
        print(f"Near net: {near_net}")

        if near_net:
            self.prev_ball_vx = ball_vx
            self.prev_ball_x = ball_x
            print("Hit was near net - updating previous states and returning")
            return

        # Distance calculations
        right_dist = abs(x - self.prev_ball_x)
        left_dist = abs(-x - self.prev_ball_x)

        print(f"\nDistance calculations:")
        print(f"Agent x: {x:.2f}")
        print(f"Ball x at hit: {self.prev_ball_x:.2f}")
        print(f"Distance if on right: {right_dist:.2f}")
        print(f"Distance if on left: {left_dist:.2f}")
        print(f"Hit threshold: {self.HIT_THRESHOLD}")

        potential_left_hit = left_dist < self.HIT_THRESHOLD
        potential_right_hit = right_dist < self.HIT_THRESHOLD

        print(f"\nHit analysis:")
        print(f"Could be left hit: {potential_left_hit}")
        print(f"Could be right hit: {potential_right_hit}")

        # Make determination
        if potential_left_hit and not potential_right_hit:
            self.is_left_side = True
            self.side_detected = True
            print("\nDETERMINATION: Left side - only left distance within threshold")
        elif potential_right_hit and not potential_left_hit:
            self.is_left_side = False
            self.side_detected = True
            print("\nDETERMINATION: Right side - only right distance within threshold")
        else:
            print("\nNo clear determination - both or neither distance within threshold")

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
