"""Core classes."""
import random
from collections import deque
import numpy as np

class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    pass


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


class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, window_length):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.psize = 84
        self.max_size = max_size
        self.window_length = window_length
        self.mem_size = (max_size + window_length-1);
        self.mem_state = np.ones((self.mem_size, self.psize, self.psize), dtype=np.uint8)
        self.mem_action = np.ones(self.mem_size, dtype=np.int8)
        self.mem_reward = np.ones(self.mem_size, dtype=np.float32)
        self.mem_terminal = np.ones(self.mem_size, dtype=np.bool)
        self.start = 0
        # End point to the next position.
        # The content doesn't change when end points at it but change
        # when end points move forward from it.
        self.end = 0
        self.full = False

    def append(self, state, action, reward, is_terminal):
        if self.start == 0 and self.end == 0: # the first frame
            # 1 2 3 S E
            for i in range(self.window_length-1):
                self.mem_state[i] = state
                self.start = (self.start + 1) % self.mem_size
            self.mem_state[self.start] = state
            self.mem_action[self.start] = action
            self.mem_reward[self.start] = reward
            self.mem_terminal[self.start] = is_terminal
            self.end = (self.start + 1) % self.mem_size
        else:
            # Case 1:  1 2 3 S ... E
            # Case 2:  ... E 1 2 3 S ...
            self.mem_state[self.end] = state
            self.mem_action[self.end] = action
            self.mem_reward[self.end] = reward
            self.mem_terminal[self.end] = is_terminal
            self.end = (self.end + 1) % self.mem_size
            if self.end > 0 and self.end < self.start:
                self.full = True

            if self.full:
                self.start = (self.start + 1) % self.mem_size

    def sample(self, batch_size, indexes=None):
        if self.end == 0 and self.start == 0:
            # state, action, reward, next_state, is_terminal
            return None, None, None, None, None
        else:
            count = 0
            if self.end > self.start:
                count = self.end - self.start
            else:
                count = self.max_size

            if count <= batch_size:
                indices = np.arange(0, count-1)
            else:
                #indices range is 0 ... count-2
                indices = np.random.randint(0, count-1, size=batch_size)

            # 4 is the current state frame because of our design
            indices_5 = (self.start + indices + 1) % self.mem_size
            indices_4 = (self.start + indices) % self.mem_size
            indices_3 = (self.start + indices - 1) % self.mem_size
            indices_2 = (self.start + indices - 2) % self.mem_size
            indices_1 = (self.start + indices - 3) % self.mem_size
            frame_5 = self.mem_state[indices_5]
            frame_4 = self.mem_state[indices_4]
            frame_3 = self.mem_state[indices_3]
            frame_2 = self.mem_state[indices_2]
            frame_1 = self.mem_state[indices_1]

            state_list = np.array([frame_1, frame_2, frame_3, frame_4])
            state_list = np.transpose(state_list, [1,0,2,3])

            next_state_list = np.array([frame_2, frame_3, frame_4, frame_5])
            next_state_list = np.transpose(next_state_list, [1,0,2,3])

            action_list = self.mem_action[indices_4]
            reward_list = self.mem_reward[indices_4]
            terminal_list = self.mem_terminal[indices_4]

            return state_list, action_list, reward_list, next_state_list, terminal_list

    def clear(self):
        self.start = 0
        self.end = 0
        self.full = False
