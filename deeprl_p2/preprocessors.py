"""Suggested Preprocessors."""

import numpy as np
import copy
from collections import deque
from PIL import Image
from deeprl_p2 import utils

class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass

class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.scale = new_size

    def crop_image(self, image, new_height, new_width):
        width, height = image.size
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        return image.crop((left, top, right, bottom))

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.

        Huiting: the API of PIL is below:
        https://pillow.readthedocs.io/en/4.0.x/reference/Image.html
        """
        state = Image.fromarray(state, 'RGB')
        state = state.convert(mode='L')
        short_side = min(state.width, state.height)
        long_side = max(state.width, state.height)
        long_scale = int(long_side * float(self.scale[0]) / short_side)
        state = state.resize((self.scale[0], long_scale))
        state = self.crop_image(state, self.scale[0], self.scale[1])
        state = np.asarray(state)

        return state

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """

        state = Image.fromarray(state, 'RGB')
        state = state.convert(mode='F')
        short_side = min(state.width, state.height)
        long_side = max(state.width, state.height)
        long_scale = int(long_side * float(self.scale[0]) / short_side)
        state = state.resize((self.scale[0], long_scale))
        state = self.crop_image(state, self.scale[0], self.scale[1])
        state = np.asarray(state) / 255.

        return state

    def process_batch(self, states, new_states):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        preprocessed_states = states.astype('float32') / 255.
        preprocessed_new_states = new_states.astype('float32') / 255.
        return preprocessed_states, preprocessed_new_states

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        if reward > 0:
            return 1.0
        elif reward < 0:
            return -1.0
        else:
            return 0.0

class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    """
    def __init__(self, preprocessors):
        self.atari = preprocessors['atari']
        self.history = preprocessors['history']
    """

    def __init__(self, history_length, new_size, num_actions):
        self.atari = AtariPreprocessor(new_size)
        self.history_length = history_length
        self.history = [None] * self.history_length
        self.num_actions = num_actions

    def process_state_for_network(self, state):
        state = self.atari.process_state_for_network(state)
        if len(self.history) and self.history[0] is None:
            for i in range(self.history_length):
                self.history[i] = state
        else:
            self.history[0:self.history_length-1] = self.history[1:self.history_length]
            self.history[self.history_length-1] = state

        return np.expand_dims(np.asarray(self.history), 0)

    def process_batch(self, states, new_states):
        return self.atari.process_batch(states, new_states)

    def process_state_for_memory(self, state):
        return self.atari.process_state_for_memory(state)

    def process_reward(self, reward):
        return self.atari.process_reward(reward)

    def process_action(self, actions):
        one_hot = np.zeros((len(actions), self.num_actions), dtype='float32')
        one_hot[np.arange(len(actions), dtype='int'), actions] = 1.

        return one_hot

    def reset(self):
        self.history = [None] * self.history_length
