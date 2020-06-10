from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Concatenate
from gym import spaces
import numpy as np
from PIL import Image
from configparser import ConfigParser
import os
from os.path import join, pardir, exists

from gym_airsim.airsim_car_env import AirSimCarEnv

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  #dynamically grow the memory used on the GPU
set_session(tf.Session(config=config))

class AirSimCarProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

config = ConfigParser()
config.read('config.ini')
num_actions = int(config['car_agent']['actions'])
                    
WINDOW_LENGTH = 4
INPUT_SHAPE = (84, 84)

env = AirSimCarEnv()
np.random.seed(123)

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_actions))
model.add(Activation('linear'))
print(model.summary())


def build_callbacks(env_name):
    log_dir = 'logs'
    if not exists(log_dir):
        os.makedirs(log_dir)
    
    checkpoint_weights_filename = join(log_dir, 'dqn_' + env_name + '_weights_{step}.h5f')
    log_filename = join(log_dir,'dqn_{}_log.json'.format(env_name))
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=25000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks

memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),  attr='eps', value_max=1., 
                              value_min=.1, value_test=.05, nb_steps=1000000)
processor = AirSimCarProcessor()

dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.0001), metrics=['mae'])

callbacks = build_callbacks('AirSimCarRL')

dqn.fit(env, nb_steps=2000000,
        visualize=False,
        verbose=2,
        callbacks=callbacks)