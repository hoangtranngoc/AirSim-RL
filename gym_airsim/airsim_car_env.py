from configparser import ConfigParser
import gym
from gym import spaces
import numpy as np
from numpy.linalg import norm
from .car_agent import CarAgent
from os.path import dirname, abspath, join
import sys
sys.path.append('..')
import utils

class AirSimCarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))

        # Using discrete actions
        #TODO Compute number of actions from other settings
        self.action_space = spaces.Discrete(int(config['car_agent']['actions']))
    
        self.image_height = int(config['airsim_settings']['image_height'])
        self.image_width = int(config['airsim_settings']['image_width'])
        self.image_channels = int(config['airsim_settings']['image_channels'])
        image_shape = (self.image_height, self.image_width, self.image_channels)

        self.track_width = float(config['airsim_settings']['track_width'])

        # Using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

        self.car_agent = CarAgent()

    def step(self, action):
        # move the car according to the action
        self.car_agent.move(action)

        # compute reward
        car_state= self.car_agent.getCarState()
        reward = self._compute_reward(car_state)

        # check if the episode is done
        car_controls = self.car_agent.getCarControls()
        done = self._isDone(car_state, car_controls, reward)

        # log info
        info = {}
        #info = {"x_pos" :x_val, "y_pos" : y_val}

        # get observation
        observation = self.car_agent.observe()

        return observation, reward, done, info

    def reset(self):
        self.car_agent.restart()
        observation = self.car_agent.observe()

        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        return #nothing

    def close (self):
        self.car_agent.reset()
        return

    def _compute_reward(self, car_state):
        way_point1, way_point2 = self.car_agent.simGet2ClosestWayPoints()
        car_pos = car_state.kinematics_estimated.position
        car_point = np.array([car_pos.x_val, car_pos.y_val])

        # perpendicular  distance to the line connecting 2 closest way points,
        # this distance is approximate to distance to center
        distance_p1_to_p2p3 = lambda p1, p2, p3: abs(np.cross(p2-p3, p3-p1))/norm(p2-p3)
        distance_to_center = distance_p1_to_p2p3(car_point, way_point1, way_point2)

        reward = utils.compute_reward_distance_to_center(distance_to_center, self.track_width)
        return reward

    def _isDone(self, car_state, car_controls, reward):
        if reward < 0:
            return True

        car_pos = car_state.kinematics_estimated.position
        car_point = ([car_pos.x_val, car_pos.y_val])
        destination = self.car_agent.simGetWayPoints()[-1]
        distance = norm(car_point-destination)
        if distance < 5: # 5m close to the destination, stop
            return True

        return False