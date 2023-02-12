import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray


# deque is a Doubly Ended Queue, provides O(1) complexity for pop and append actions
# namedtuple is a tuple that can be accessed by both its index and attributes


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int, device: torch.device, seed: int):
        """
        Buffer to store experience tuples. Each experience has the following structure:
        (state, action, reward, next_state, done)
        Args:
            buffer_size: Maximum size for the buffer. Higher buffer size imply higher RAM consumption.
            batch_size: Number of experiences to be retrieved from the ReplayBuffer per batch.
            device: CUDA device.
            seed: Random seed.
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state: NDArray, action: NDArray, reward: float, next_state: NDArray, done: int) -> None:
        """
        Add a new experience to the Replay Buffer.
        Args:
            state: NDArray of the current state.
            action: NDArray of the action taken from state {state}.
            reward: Reward obtained after performing action {action} from state {state}.
            next_state: NDArray of the state reached after performing action {action} from state {state}.
            done: Integer (0 or 1) indicating whether the next_state is a terminal state.
        """
        # Create namedtuple object from the experience
        exp = self.experience(state, action, reward, next_state, done)
        # Add the experience object to memory
        self.memory.append(exp)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.
        Returns:
            Tuple of 5 elements, which are (states, actions, rewards, next_states, dones). Each element
            in the tuple is a torch Tensor composed of {batch_size} items.
        """
        # Randomly sample a batch of experiences
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state if not isinstance(e.state, tuple) else e.state[0] for e in experiences])).float().to(
            self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Return the current size of the Replay Buffer
        Returns:
            Size of Replay Buffer
        """
        return len(self.memory)
