#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import sys

import gym
from gym import wrappers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, Input)
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import deeprl_p2 as tfrl
from deeprl_p2.dqn import DQNAgent
from deeprl_p2.objectives import mean_huber_loss
from deeprl_p2.preprocessors import PreprocessorSequence
from deeprl_p2.policy import UniformRandomPolicy, GreedyPolicy, GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from deeprl_p2.core import ReplayMemory

def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    return None, None

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def train(args):
    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    env = gym.make(args.env)
    num_actions = env.action_space.n

    network_model, q_values_func = create_model(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions, model_name='q_network')
    preprocessor = PreprocessorSequence(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions)
    memory = ReplayMemory(args.memsize, args.stack_frames)
    policy = {
                'init':    UniformRandomPolicy(num_actions),
                'train':   GreedyEpsilonPolicy(num_actions),
                'test':    GreedyPolicy(),
    }

    print("Generate Model...")
    dqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output)

    print("Compile Model...")
    dqn_agent.compile(optimizer=Adam(lr=args.learning_rate), loss_func=mean_huber_loss)

    print("Fit Model...")
    sys.stdout.flush()

    dqn_agent.fit(env, args.num_iterations, args.max_episode_length)

def test(args):
    if not os.path.isfile(args.model_path):
        print("The model path: {} doesn't exist in the system.".format(args.model_path))
        print("Hints: python dqn_atari.py --mode test --model_path Path_to_your_model_weigths")
        return

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    env = gym.make(args.env)
    num_actions = env.action_space.n
    network_model, q_values_func = create_model(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions, model_name='q_network')


    rewards = []
    lens = []
    tries = 0
    while True:
        env = gym.make(args.env)
        env = wrappers.Monitor(env, 'videos', force=True)

        #network_model.load_weights(args.output + '/model_weights_%d.h5' % m)

        preprocessor = PreprocessorSequence(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions)
        memory = ReplayMemory(args.memsize, args.stack_frames)
        policy = {
                  'init':   UniformRandomPolicy(num_actions),
                  'train':  GreedyEpsilonPolicy(num_actions),
                  'test':   GreedyPolicy(),
        }

        dqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output)

        dqn_agent.load_weights(args.model_path)

        cumulative_reward, std, average_episode_length = dqn_agent.evaluate(env, 1, None)
        tries += 1

        # Sometime the model is not very stable.
        if tries > 100 or cumulative_reward > 350:
            break

        print ('average reward = %f, std = %f, average_epis_length = %d' % (cumulative_reward, std, average_episode_length))
        rewards.append(cumulative_reward)
        lens.append(average_episode_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--output', default='model-weights', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--memsize', default=1000000, type=int, help='Replay Memory Size')
    parser.add_argument('--mode', default='train', type=str, help='Train or test')
    parser.add_argument('--stack_frames', default=4, type=int, help='The number of stacked frames')
    parser.add_argument('--cropped_size', default=84, type=int, help='The size of the cropped windows')
    parser.add_argument('--max_episode_length', default=10000, type=int, help='the maximum of episode to be ran')
    parser.add_argument('--gamma', default=0.99, type=float, help='The reward discount parameter')
    parser.add_argument('--target_update_freq', default=10000, type=int, help='how many steps to update target network')
    parser.add_argument('--num_burn_in', default=12000, type=int, help='how many frames to burn in the memory before traiing')
    parser.add_argument('--train_freq', default=10, type=int, help='How often you actually update your Q-Network. Sometimes stability is improved if you collect a couple samples for your replay memory, for every Q-network update that you run.')
    parser.add_argument('--batch_size', default=32, type=int, help='size of each training batch')
    parser.add_argument('--learning_rate', default=0.00025, type=float, help='size of each training batch')
    parser.add_argument('--num_iterations', default=5000000, type=int, help='the number of iteration to run')
    parser.add_argument('--model_path', default='', type=str, help='path to the model')

    args = parser.parse_args()

    if args.mode == "train":
        args.output = get_output_folder(args.output, args.env)
        train(args)
    elif args.mode == "test":
        test(args)
