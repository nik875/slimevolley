import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import gym
import slimevolleygym
import pickle

class GameHistoryCollector:
    def __init__(self):
        self.history = []
        
    def record_state(self, obs, info):
        # Record the state from the perspective of the right agent
        otherObs = info['otherObs']
        state = {
            'right_agent': (obs[0], obs[1]),
            'ball': (obs[4], obs[5]),
            'left_agent': (otherObs[0], otherObs[1]),
        }
        self.history.append(state)
        
    def get_history(self):
        return self.history

class GameRenderer:
    def __init__(self, history, fps=30):
        self.history = history
        self.fps = fps
        
        # Set up the figure and animation
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.setup_plot()
        
        # Initialize artists with explicit labels
        self.left_agent = Circle((0, 0), radius=0.15, color='red', label='Agent 1')
        self.right_agent = Circle((0, 0), radius=0.15, color='blue', label='Agent 2')
        self.ball = Circle((0, 0), radius=0.05, color='green', label='Ball')
        
        # Add artists to plot
        self.ax.add_patch(self.left_agent)
        self.ax.add_patch(self.right_agent)
        self.ax.add_patch(self.ball)
        
        # Add trajectory lines with labels
        self.left_line, = self.ax.plot([], [], 'r:', alpha=0.3, label='Agent 1 Trail')
        self.right_line, = self.ax.plot([], [], 'b:', alpha=0.3, label='Agent 2 Trail')
        self.ball_line, = self.ax.plot([], [], 'g:', alpha=0.3, label='Ball Trail')
        
        # Draw the ground and center wall
        self.ground = plt.Rectangle((-2.5, 0), 5.0, 0.15, color='gray', alpha=0.5)
        self.wall = plt.Rectangle((-0.05, 0.15), 0.1, 0.35, color='brown', alpha=0.5)
        self.ax.add_patch(self.ground)
        self.ax.add_patch(self.wall)
        
        # Create legend
        self.ax.legend(loc='upper right')
        
    def setup_plot(self):
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(0, 3)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Slime Volley Game Replay')
        
    def mirror_coordinates(self, x, y):
        """Mirror the x-coordinates while keeping y-coordinates the same"""
        return -x, y
    
    def animate(self, frame):
        state = self.history[frame]

        # Mirror the coordinates for the left agent
        left_x, left_y = self.mirror_coordinates(*state['left_agent'])
        right_x, right_y = state['right_agent']
        ball_x, ball_y = state['ball']

        # Update circle positions
        self.left_agent.center = (left_x, left_y)
        self.right_agent.center = (right_x, right_y)
        self.ball.center = (ball_x, ball_y)

        # Update trajectory lines
        history_slice = self.history[:frame+1]

        # Get trajectories for the correct sides
        left_positions = np.array([self.mirror_coordinates(*s['left_agent']) for s in history_slice])
        right_positions = np.array([s['right_agent'] for s in history_slice])
        ball_positions = np.array([s['ball'] for s in history_slice])

        self.left_line.set_data(left_positions[:, 0], left_positions[:, 1])
        self.right_line.set_data(right_positions[:, 0], right_positions[:, 1])
        self.ball_line.set_data(ball_positions[:, 0], ball_positions[:, 1])

        return self.left_agent, self.right_agent, self.ball, self.left_line, self.right_line, self.ball_line

    def create_animation(self):
        anim = FuncAnimation(
            self.fig, 
            self.animate,
            frames=len(self.history),
            interval=1000/self.fps,
            blit=True
        )
        return anim

def play_and_record_game(left_network, right_network, max_steps=3000):
    """Play a game and record the history of positions"""
    env = gym.make('SlimeVolley-v0')
    env.multiagent = True
    env.reset()
    collector = GameHistoryCollector()
    
    for i in [left_network, right_network]:
        if hasattr(i, "reset"):
            i.reset()
        elif hasattr(i, "reset_hidden"):
            i.reset_hidden()
    
    # Initialize the game
    obs, _, _, info = env.step([0, 0, 0], [0, 0, 0])
    right_obs = obs
    left_obs = info['otherObs']
    collector.record_state(obs, info)
    
    # Play the game
    done = False
    step = 0
    while not done and step < max_steps:
        right_action = right_network.forward(right_obs)
        left_action = left_network.forward(left_obs)
        
        obs, reward, done, info = env.step(right_action, left_action)
        
        right_obs = obs
        left_obs = info['otherObs']
        collector.record_state(obs, info)
        step += 1
        
    env.close()
    return collector.get_history()

def evaluate_average(left_network, right_network, max_steps=1000, n_games=100):
    """Evaluate a match between a MAML network and a traditional opponent"""
    avg_reward = 0
    avg_crosses = 0
    avg_frames = 0
    for _ in range(n_games):
        env = gym.make('SlimeVolley-v0')
        env.multiagent = True
        env.reset()

        obs, _, _, info = env.step([0, 0, 0], [0, 0, 0])
        left_obs = obs
        right_obs = info['otherObs']

        for i in [left_network, right_network]:
            if hasattr(i, "reset"):
                i.reset()
            elif hasattr(i, "reset_hidden"):
                i.reset_hidden()

        done = False
        total_reward = 0
        crosses = 0
        frames = 0
        while not done and frames < max_steps:
            right_action = right_network.forward(right_obs)
            left_action = left_network.forward(left_obs)

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
        avg_reward += total_reward
        avg_crosses += crosses
        avg_frames += frames
    print("Average reward for right: ", avg_reward / n_games)
    print("Average crosses per game: ", avg_crosses / n_games)
    print("Average frames: ", avg_frames / n_games)

def create_game_replay(left_network, right_network, save_path=None, fps=30):
    """Create and optionally save an animation of a game between two agents"""
    history = play_and_record_game(left_network, right_network)
    renderer = GameRenderer(history, fps=fps)
    anim = renderer.create_animation()
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow')
        else:
            html_path = save_path.rsplit('.', 1)[0] + '.html'
            plt.rcParams['animation.embed_limit'] = 50
            anim.save(html_path, writer='html')
    
    return anim, renderer.fig

# Load the parameters
#from ad_selfplay_orig import Network
#with open("oldmodel_1000gen_best.pkl", 'rb') as f:
#    left_params = pickle.load(f)
#from ad_selfplay_new import Network
#with open("fullmodel_600gen_best.pkl", 'rb') as f:
#    right_params = pickle.load(f)

#from ad_selfplay_orig import Network
#with open("oldmodel_1000gen_best.pkl", 'rb') as f:
#    left_params = pickle.load(f)

# Create the game replay
#anim, fig = create_game_replay(left_params, right_params, 'game_replay3.gif')
#evaluate_average(left_params, right_params, max_steps=2400, n_games=1024)
#evaluate_average(right_params, left_params, max_steps=2400, n_games=1024)

class Pretrained:
    def __init__(self):
        self.nGameInput = 8 # 8 states for agent
        self.nGameOutput = 3 # 3 buttons (forward, backward, jump)
        self.nRecurrentState = 4 # extra recurrent states for feedback.

        self.nOutput = self.nGameOutput+self.nRecurrentState
        self.nInput = self.nGameInput+self.nOutput

        # store current inputs and outputs
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

        self.weight = np.array(
        [7.5719, 4.4285, 2.2716, -0.3598, -7.8189, -2.5422, -3.2034, 0.3935, 1.2202, -0.49, -0.0316, 0.5221, 0.7026, 0.4179, -2.1689,
        1.646, -13.3639, 1.5151, 1.1175, -5.3561, 5.0442, 0.8451, 0.3987, -2.9501, -3.7811, -5.8994, 6.4167, 2.5014, 7.338, -2.9887,
        2.4586, 13.4191, 2.7395, -3.9708, 1.6548, -2.7554, -1.5345, -6.4708, 9.2426, -0.7392, 0.4452, 1.8828, -2.6277, -10.851, -3.2353,
        -4.4653, -3.1153, -1.3707, 7.318, 16.0902, 1.4686, 7.0391, 1.7765, -1.155, 2.6697, -8.8877, 1.1958, -3.2839, -5.4425, 1.6809,
        7.6812, -2.4732, 1.738, 0.3781, 0.8718, 2.5886, 1.6911, 1.2953, -9.0052, -4.6038, -6.7447, -2.5528, 0.4391, -4.9278, -3.6695,
        -4.8673, -1.6035, 1.5011, -5.6124, 4.9747, 1.8998, 3.0359, 6.2983, -4.8568, -2.1888, -4.1143, -3.9874, -0.0459, 4.7134, 2.8952,
        -9.3627, -4.685, 0.3601, -1.3699, 9.7294, 11.5596, 0.1918, 3.0783, 0.0329, -0.1362, -0.1188, -0.7579, 0.3278, -0.977, -0.9377])

        self.bias = np.array([2.2935,-2.0353,-1.7786,5.4567,-3.6368,3.4996,-0.0685])

        # unflatten weight, convert it into 7x15 matrix.
        self.weight = self.weight.reshape(self.nGameOutput+self.nRecurrentState,
        self.nGameInput+self.nGameOutput+self.nRecurrentState)

    def reset(self):
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

    def _forward(self):
        self.prevOutputState = self.outputState
        self.outputState = np.tanh(np.dot(self.weight, self.inputState)+self.bias)

    def _setInputState(self, obs):
        # obs is: (op is opponent). obs is also from perspective of the agent (x values negated for other agent)
        [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = obs
        self.inputState[0:self.nGameInput] = np.array([x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy])
        self.inputState[self.nGameInput:] = self.outputState

    def _getAction(self):
        forward = 0
        backward = 0
        jump = 0
        if (self.outputState[0] > 0.75):
            forward = 1
        if (self.outputState[1] > 0.75):
            backward = 1
        if (self.outputState[2] > 0.75):
            jump = 1
        return [forward, backward, jump]

    def predict(self, obs):
        self._setInputState(obs)
        self._forward()
        return self._getAction()

    def forward(self, obs):
        return self.predict(obs)


pretrained = Pretrained()

from ad_selfplay_orig import Network
from side_detect import load_smart_agent
our_agent = load_smart_agent("oldmodel_1000gen_best.pkl")
print("Our agent left")
evaluate_average(our_agent, pretrained, max_steps=2400, n_games=1024)
print("Our agent right")
evaluate_average(pretrained, our_agent, max_steps=2400, n_games=1024)
anim, fig = create_game_replay(our_agent, pretrained, 'game_replay.gif', code=True)
