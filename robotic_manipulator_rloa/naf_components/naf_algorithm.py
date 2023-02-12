import json
import os
import random
import time
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch.nn.utils import clip_grad_norm_

from robotic_manipulator_rloa.utils.logger import get_global_logger
from robotic_manipulator_rloa.environment.environment import Environment
from robotic_manipulator_rloa.utils.exceptions import MissingWeightsFile
from robotic_manipulator_rloa.naf_components.naf_neural_network import NAF
from robotic_manipulator_rloa.utils.replay_buffer import ReplayBuffer


logger = get_global_logger()


class NAFAgent:

    MODEL_PATH = 'model.p'  # Filename where the parameters of the trained torch neural network are stored

    def __init__(self,
                 environment: Environment,
                 state_size: int,
                 action_size: int,
                 layer_size: int,
                 batch_size: int,
                 buffer_size: int,
                 learning_rate: float,
                 tau: float,
                 gamma: float,
                 update_freq: int,
                 num_updates: int,
                 checkpoint_frequency: int,
                 device: torch.device,
                 seed: int) -> None:
        """
        Interacts with and learns from the environment via the NAF algorithm.
        Args:
            environment: Instance of Environment class.
            state_size: Dimension of the states.
            action_size: Dimension of the actions.
            layer_size: Size for the hidden layers of the neural network.
            batch_size: Number of experiences to train with per training batch.
            buffer_size: Maximum number of experiences to be stored in Replay Buffer.
            learning_rate: Learning rate for neural network's optimizer.
            tau: Hyperparameter for soft updating the target network.
            gamma: Discount factor.
            update_freq: Number of timesteps after which the main neural network is updated.
            num_updates: Number of updates performed when learning.
            checkpoint_frequency: Number of episodes after which a checkpoint is generated.
            device: Device used (CPU or CUDA).
            seed: Random seed.
        """
        # Create required parent directory
        os.makedirs('checkpoints/', exist_ok=True)

        self.environment = environment
        self.state_size = state_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        random.seed(seed)
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.update_freq = update_freq
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.checkpoint_frequency = checkpoint_frequency

        # Initalize Q-Networks
        self.qnetwork_main = NAF(state_size, action_size, layer_size, seed, device).to(device)
        self.qnetwork_target = NAF(state_size, action_size, layer_size, seed, device).to(device)

        # Define Adam as optimizer
        self.optimizer = optim.Adam(self.qnetwork_main.parameters(), lr=learning_rate)

        # Initialize Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device, seed)

        # Initialize update time step counter (for updating every {update_freq} steps)
        self.update_t_step = 0

    def initialize_pretrained_agent_from_episode(self, episode: int) -> None:
        """
        Loads the previously trained weights into the main and target neural networks.
        The pretrained weights are retrieved from the checkpoints generated on a training execution, so
        the episode provided must be present in the checkpoints/ folder.
        Args:
            episode: Episode from which to retrieve the pretrained weights.
        Raises:
            MissingWeightsFile: The weights.p file is not present in the checkpoints/{episode}/ folder provided.
        """
        # Check if file is present in checkpoints/{episode}/ directory
        if not os.path.isfile(f'checkpoints/{episode}/weights.p'):
            raise MissingWeightsFile

        logger.debug(f'Loading naf_components weights from trained naf_components on episode {episode}...')
        self.qnetwork_main.load_state_dict(torch.load(f'checkpoints/{episode}/weights.p'))
        self.qnetwork_target.load_state_dict(torch.load(f'checkpoints/{episode}/weights.p'))
        logger.info(f'Loaded weights from trained naf_components on episode {episode}')

    def initialize_pretrained_agent_from_weights_file(self, weights_path: str) -> None:
        """
        Loads the previously trained weights into the main and target neural networks.
        The pretrained weights are retrieved from a .p file containing the weights, located in 
        the {weights_path} path.
        Args:
            weights_path: Path to the .p file containing the network's weights.
        Raises:
            MissingWeightsFile: The file path provided does not exist.
        """
        # Check if file is present
        if not os.path.isfile(weights_path):
            raise MissingWeightsFile

        logger.debug('Loading naf_components weights from trained naf_components...')
        self.qnetwork_main.load_state_dict(torch.load(weights_path))
        self.qnetwork_target.load_state_dict(torch.load(weights_path))
        logger.info('Loaded pre-trained weights for the NN')

    def step(self, state: NDArray, action: NDArray, reward: float, next_state: NDArray, done: int) -> None:
        """
        Stores in the ReplayBuffer the new experience composed by the parameters received,
        and learns only if the Buffer contains enough experiences to fill a batch. The
        learning will occur if the update frequency {update_freq} is reached, in which case it
        will learn {num_updates} times.
        Args:
            state: Current state.
            action: Action performed from state {state}.
            reward: Reward obtained after performing action {action} from state {state}.
            next_state: New state reached after performing action {action} from state {state}.
            done: Integer (0 or 1) indicating whether a terminal state have been reached.
        """

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learning will be performed every {update_freq}} time-steps.
        self.update_t_step = (self.update_t_step + 1) % self.update_freq  # Update time step counter
        if self.update_t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                for _ in range(self.num_updates):
                    # Pick random batch of experiences from memory
                    experiences = self.memory.sample()

                    # Learn from experiences and get loss
                    self.learn(experiences)

    def act(self, state: NDArray) -> NDArray:
        """
        Extracts the action which maximizes the Q-Function, by getting the output of the mu layer
        of the main neural network.
        Args:
            state: Current state from which to pick the best action.
        Returns:
            Action which maximizes Q-Function.
        """
        state = torch.from_numpy(state).float().to(self.device)

        # Set evaluation mode on naf_components for obtaining a prediction
        self.qnetwork_main.eval()
        with torch.no_grad():
            # Get the action with maximum Q-Value from the local network
            action, _, _ = self.qnetwork_main(state.unsqueeze(0))

        # Set training mode on naf_components for future use
        self.qnetwork_main.train()

        return action.cpu().squeeze().numpy()

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """
        Calculate the Q-Function estimate from the main neural network, the Target value from
        the target neural network, and calculate the loss with both values, all by feeding the received
        batch of experience tuples to both networks. After loss is calculated, backpropagation is performed on the
        main network from the given loss, so that the weights of the main network are updated.
        Args:
            experiences: Tuple of five elements, where each element is a torch.Tensor of length {batch_size}.
        """
        # Set gradients of all optimized torch Tensors to zero
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # Get the Value Function for the next state from target naf_components (no_grad() disables gradient calculation)
        with torch.no_grad():
            _, _, V_ = self.qnetwork_target(next_states)

        # Compute the target Value Functions for the given experiences.
        # The target value is calculated as target_val = r + gamma * V(s')
        target_values = rewards + (self.gamma * V_)

        # Compute the expected Value Function from main network
        _, q_estimate, _ = self.qnetwork_main(states, actions)

        # Compute loss between target value and expected Q value
        loss = F.mse_loss(q_estimate, target_values)

        # Perform backpropagation for minimizing loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_main.parameters(), 1)
        self.optimizer.step()

        # Update the target network softly with the local one
        self.soft_update(self.qnetwork_main, self.qnetwork_target)

        # return loss.detach().cpu().numpy()

    def soft_update(self, main_nn: NAF, target_nn: NAF) -> None:
        """
        Soft update naf_components parameters following this formula:\n
                    θ_target = τ*θ_local + (1 - τ)*θ_target
        Args:
            main_nn: Main torch neural network.
            target_nn: Target torch neural network.
        """
        for target_param, main_param in zip(target_nn.parameters(), main_nn.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1. - self.tau) * target_param.data)

    def run(self, frames: int = 1000, episodes: int = 1000, verbose: bool = True) -> Dict[int, Tuple[float, int]]:
        """
        Execute training flow of the NAF algorithm on the given environment.
        Args:
            frames: Number of maximum frames or timesteps per episode.
            episodes: Number of episodes required to terminate the training.
            verbose: Boolean indicating whether many or few logs are shown.
        Returns:
            Returns the score history generated along the training.
        """
        logger.info('Training started')
        # Initialize 'scores' dictionary to store rewards and timesteps executed for each episode
        scores = {episode: (0, 0) for episode in range(1, episodes + 1)}

        # Iterate through every episode
        for episode in range(episodes):
            logger.info(f'Running Episode {episode + 1}')
            start = time.time()  # Timer to measure execution time per episode
            state = self.environment.reset(verbose)
            score, mean = 0, list()

            for frame in range(1, frames + 1):
                if verbose: logger.info(f'Running frame {frame} in episode {episode + 1}')

                # Pick action according to current state
                if verbose: logger.info(f'Current State: {state}')
                action = self.act(state)
                if verbose: logger.info(f'Action chosen for the given state is: {action}')

                # Perform action on environment and get new state and reward
                next_state, reward, done = self.environment.step(action)

                # Save the experience in the ReplayBuffer, and learn from previous experiences if applicable
                self.step(state, action, reward, next_state, done)

                state = next_state  # Update state to next state
                score += reward
                mean.append(reward)

                if verbose: logger.info(f'Reward: {reward}  -  Cumulative reward: {score}\n')

                if done:
                    break

            # Updates scores history
            scores[episode + 1] = (score, frame)  # save most recent score and last frame
            logger.info(f'Reward:                             {score}')
            logger.info(f'Number of frames:                   {frame}')
            logger.info(f'Mean of rewards on this episode:    {sum(mean) / frames}')
            logger.info(f'Time taken for this episode:        {round(time.time() - start, 3)} secs\n')

            # Save the episode's performance if it is a checkpoint episode
            if (episode + 1) % self.checkpoint_frequency == 0:
                # Create parent directory for current episode
                os.makedirs(f'checkpoints/{episode + 1}/', exist_ok=True)
                # Save naf_components weights
                torch.save(self.qnetwork_main.state_dict(), f'checkpoints/{episode + 1}/weights.p')
                # Save naf_components's performance metrics
                with open(f'checkpoints/{episode + 1}/scores.txt', 'w') as f:
                    f.write(json.dumps(scores))

        torch.save(self.qnetwork_main.state_dict(), self.MODEL_PATH)
        logger.info(f'Model has been successfully saved in {self.MODEL_PATH}')

        return scores
