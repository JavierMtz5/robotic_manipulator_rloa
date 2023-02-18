import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import torch

from robotic_manipulator_rloa.environment.environment import Environment, EnvironmentConfiguration
from robotic_manipulator_rloa.utils.exceptions import (
    EnvironmentNotInitialized,
    NAFAgentNotInitialized,
    ConfigurationIncomplete,
    InvalidHyperParameter,
    InvalidNAFAgentParameter
)
from robotic_manipulator_rloa.utils.logger import get_global_logger, Logger
from robotic_manipulator_rloa.naf_components.naf_algorithm import NAFAgent

# Mute matplotlib and PIL logs
plt.set_loglevel('critical')
logging.getLogger('PIL').setLevel(logging.WARNING)
# Initialize framework logger
logger = get_global_logger()
Logger.set_logger_setup()


@dataclass
class HyperParameters:
    """
    Data Class for the storage of the required hyper-parameters.
    """
    buffer_size: int = 100000
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.001
    learning_rate: float = 0.001
    update_freq: int = 1
    num_updates: int = 1


class ManipulatorFramework:

    def __init__(self) -> None:
        """
        Main class of the package. Initializes the required hyper-parameters to their default value.
        """
        self.env: Union[Environment, None] = None
        self.naf_agent: Union[NAFAgent, None] = None
        self._hyperparameters: Union[HyperParameters, None] = None

        self._initialize_hyperparameters()
        logger.info('The Framework has been initialized with the default hyperparameters configuration')
        logger.debug('* Custom hyperparameters can be set via the set_hyperparameter() method')
        logger.debug('* All the required hyperparameters can be printed via the get_required_hyperparameters() method')
        logger.debug('* Load a manipulator via the initialize_environment() method to start with the training '
                     'configuration')

    def _initialize_hyperparameters(self) -> None:
        """
        Initializes the hyper-parameters with their default values.
        """
        self._hyperparameters = HyperParameters(buffer_size=100000,
                                                batch_size=128,
                                                gamma=0.99,
                                                tau=0.001,
                                                learning_rate=0.001,
                                                update_freq=1,
                                                num_updates=1)

    @staticmethod
    def set_log_level(log_level: int) -> None:
        """
        Set Log Level for the Logger.
        Args:
            log_level: Log level to be set.\n
                Valid values: 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL)
        """
        level_name_mappings = {10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}
        # Set Log level
        if log_level in [10, 20, 30, 40, 50]:
            logger.info(f'Log Level has been set to {logger.level} ({level_name_mappings[logger.level]})')
            logger.setLevel(log_level)
        else:
            logger.error(
                f'The Log level provided is invalid, so the previous Log Level is maintained ({logger.level}))')
            logger.error('Valid values: 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL)')

    @staticmethod
    def get_required_hyperparameters() -> None:
        """
        Returns a list with the required hyper-parameters. This function does not imply
        any logic, as it is only intended to help the user to know what hyper-parameters can be set.
        The required hyper-parameters are shown as DEBUG logs, so if the Log level is set to INFO or higher
        the function will not return anything.
        """
        if logger.level > 10:
            logger.error('get_required_hyperparameters() only shows information for DEBUG log level. '
                         'Try running this method after setting the log level to DEBUG by calling '
                         'set_log_level(10) class method')
            return

        hyperparameters_info_map = {
            'Buffer Size': 'https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial?hl=es-419',
            'Batch Size': 'https://www.kaggle.com/general/276990',
            'Gamma (discount factor)': 'https://arxiv.org/pdf/2007.02040.pdf',
            'Tau': 'https://arxiv.org/abs/1603.00748',
            'Learning Rate': 'https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep'
                             '-learning-neural-networks/',
            'Update Frequency': 'https://medium.com/towards-data-science/applied-reinforcement-learning-v-normalized'
                                '-advantage-function-naf-for-continuous-control-62ad143d3095',
            'Number of Updates': 'https://medium.com/towards-data-science/applied-reinforcement-learning-v-normalized'
                                 '-advantage-function-naf-for-continuous-control-62ad143d3095'}

        logger.debug('Required Hyperparameters:')
        for hyperparam, info in hyperparameters_info_map.items():
            logger.debug('{:<25} (see {:<10})'.format(hyperparam, info))

    @staticmethod
    def plot_training_rewards(episode: int, mean_range: int = 50) -> None:
        """
        Plots the mean reward of each batch of {mean_range} episodes for the "scores.txt" file stored
        in the checkpoints/ folder, corresponding to the episode received as parameter.
        Args:
            episode: Episode number from which to plot the results. The episode provided must be one of the
                checkpoints generated in the /checkpoints directory.
            mean_range: Range of episodes on which the mean is calculated. If the execution lasted for
                200 episodes, and the mean_range is set to 50, 4 metric points will be generated.
        Raises:
            FileNotFoundError: Raises if the episode provided is not present as a directory in the /checkpoints
                directory generated after executing a training.
        """

        try:
            with open(f'checkpoints/{episode}/scores.txt', 'r') as f:
                file = f.read()
                scores = json.loads(file)
        except FileNotFoundError as err:
            logger.error(f'File "scores.txt" located in checkpoints/{episode}/ folder was not found')
            raise err

        counter, cummulative_reward, values_to_plot = 0, list(), list()
        for episode, result in scores.items():
            cummulative_reward.append(result[0])
            counter += 1
            if counter % mean_range == 0:
                mean = sum(cummulative_reward) / len(cummulative_reward)
                values_to_plot.append(mean)
                cummulative_reward = list()

        plt.plot(range(len(values_to_plot)), values_to_plot)
        plt.show()

    def set_hyperparameter(self, hyperparameter: str, value: Union[float, int]) -> None:
        """
        Sets the specified value on the given hyper-parameter. Checking are performed to
        ensure that the new value for the hyper-parameter is a valid value.
        Args:
            hyperparameter: Name of the hyper-parameter to be updated.
                Allowed names are:\n
                - buffer_size/buffersize/BUFFER_SIZE/BUFFERSIZE\n
                - batch_size/batchsize/BATCH_SIZE/BATCHSIZE\n
                - gamma/GAMMA\n
                - tau/TAU\n
                - learning_rate/learningrate/LEARNING_RATE/LEARNINGRATE\n
                - update_freq/updatefreq/UPDATE_FREQ/UPDATEFREQ\n
                - num_update/numupdate/NUMUPDATE/NUM_UPDATE\n
            value: New value for the given hyper-parameter.
        Raises:
            InvalidHyperParameter: The hyperparameter received has an invalid value/type, or the hyperparameter
                name received is not one of the accepted values.
        """
        # Set BUFFER SIZE
        if re.match(r'^(buffer_size|buffersize|BUFFER_SIZE|BUFFERSIZE)$', hyperparameter):
            if not (isinstance(value, int) and value > 0):
                raise InvalidHyperParameter('Buffer Size is not an int or has a value lower than 0')
            self._hyperparameters.buffer_size = value
            logger.info(f'Hyperparameter {hyperparameter} has been set to {value}')

        # Set BATCH SIZE
        elif re.match(r'^(batch_size|batchsize|BATCH_SIZE|BATCHSIZE)$', hyperparameter):
            if not (isinstance(value, int) and value > 0):
                raise InvalidHyperParameter('Batch Size is not an int or has a value lower than 0')
            self._hyperparameters.batch_size = value
            logger.info(f'Hyperparameter {hyperparameter} has been set to {value}')

        # Set GAMMA
        elif re.match(r'^(gamma|GAMMA)$', hyperparameter):
            if not (isinstance(value, (int, float)) and 0 < value < 1):
                raise InvalidHyperParameter('Gamma is not a float or its value is out of range (0, 1)')
            self._hyperparameters.gamma = value
            logger.info(f'Hyperparameter {hyperparameter} has been set to {value}')

        # Set TAU
        elif re.match(r'^(tau|TAU)$', hyperparameter):
            if not (isinstance(value, (int, float)) and 0 <= value <= 1):
                raise InvalidHyperParameter('Tau is not a float or its value is out of range [0, 1]')
            self._hyperparameters.tau = value
            logger.info(f'Hyperparameter {hyperparameter} has been set to {value}')

        # Set LEARNING RATE
        elif re.match(r'^(learning_rate|learningrate|LEARNING_RATE|LEARNINGRATE)$', hyperparameter):
            if not (isinstance(value, (int, float)) and value > 0):
                raise InvalidHyperParameter('Learning Rate is not a float or has a value lower than 0')
            self._hyperparameters.learning_rate = value
            logger.info(f'Hyperparameter {hyperparameter} has been set to {value}')

        # Set UPDATE EVERY
        elif re.match(r'^(update_freq|updatefreq|UPDATE_FREQ|UPDATEFREQ)$', hyperparameter):
            if not (isinstance(value, int) and value > 0):
                raise InvalidHyperParameter('Update Frequency is not an int or has a value lower than 0')
            self._hyperparameters.update_freq = value
            logger.info(f'Hyperparameter {hyperparameter} has been set to {value}')

        # Set NUPDATE
        elif re.match(r'^(num_update|numupdate|NUMUPDATE|NUM_UPDATE)$', hyperparameter):
            if not (isinstance(value, int) and value > 0):
                raise InvalidHyperParameter('Buffer Size is not an int or has a value lower than 0')
            self._hyperparameters.num_updates = value
            logger.info(f'Hyperparameter {hyperparameter} has been set to {value}')

        else:
            raise InvalidHyperParameter(
                'The hyperparameter name passed as parameter is not valid. Valid hyperparameters are: '
                '["buffer_size", "batch_size", "gamma", "tau", "learning_rate", "update_freq", "num_update"]')

    def load_pretrained_parameters_from_weights_file(self, parameters_file_path: str) -> None:
        """
        Loads a pretrained network's weights into the current networks. The weights are loaded from the path
        provided in the {parameters_file_path} parameter. As the weights are loaded in the neural networks
        contained in the NAFAgent class, the method will raise an error if either the Environment or the NAFAgent
        class are not initialized.
        Args:
            parameters_file_path: Path to the .p file where the weights are stored.
        Raises:
            EnvironmentNotInitialized: The Environment class has not been initialized.
            NAFAgentNotInitialized: The NAFAgentNotInitialized class has not been initialized.
            MissingWeightsFile: The .p file path provided does not exist.
        """
        if not self.env:
            raise EnvironmentNotInitialized

        if not self.naf_agent:
            raise NAFAgentNotInitialized

        self.naf_agent.initialize_pretrained_agent_from_weights_file(parameters_file_path)

    def load_pretrained_parameters_from_episode(self, episode: int) -> None:
        """
        Loads previously trained weights into the current networks.
        The pretrained weights are retrieved from the checkpoints generated on a training execution, so
        the episode provided must be present in the checkpoints/ folder. As the weights are loaded in the
        neural networks contained in the NAFAgent class, the method will raise an error if either
        the Environment or the NAFAgent class are not initialized.
        Args:
            episode: Episode in the /checkpoints folder from which to retrieve the pretrained weights.
        Raises:
            EnvironmentNotInitialized: The Environment class has not been initialized.
            NAFAgentNotInitialized: The NAFAgentNotInitialized class has not been initialized.
            MissingWeightsFile: The weights.p file does not exist in the folder provided.
        """
        if not self.env:
            raise EnvironmentNotInitialized

        if not self.naf_agent:
            raise NAFAgentNotInitialized

        self.naf_agent.initialize_pretrained_agent_from_episode(episode)

    def get_environment_configuration(self) -> None:
        """
        Shows the Environment configuration as logs on stdout.
        """
        if not self.env:
            logger.error("Environment is not initialized yet, can't show configuration")
            return

        logger.info('Environment Configuration:')
        logger.info(f'* Manipulator File:                    {self.env.manipulator_file}')
        logger.info(f'* End Effector index:                  {self.env.endeffector_index}')
        logger.info(f'* List of fixed Joints:                {self.env.fixed_joints}')
        logger.info(f'* List of Joints involved in training: {self.env.involved_joints}')
        logger.info(f'* Position of the Target:              {self.env.target_pos}')
        logger.info(f'* Position of the Obstacle:            {self.env.obstacle_pos}')
        logger.info(f'* Initial position of joints:          {self.env.initial_joint_positions}')
        logger.info(f'* Initial variation range of joints:   {self.env.initial_positions_variation_range}')
        logger.info(f'* Max Force to be applied on joints:   {self.env.max_force}')
        logger.info(f'* Visualize mode:                      {self.env.visualize}')
        logger.info(f'* Instance of the Environment:         {self.env}')

    def get_nafagent_configuration(self) -> None:
        """
        Shows the NAFAgent configuration as logs on stdout.
        """
        if not self.naf_agent:
            logger.error("NAFAgent is not initialized yet, can't show configuration")
            return

        logger.info('NAFAgent Configuration:')
        logger.info(f'* Environment Instance:                    {self.naf_agent.environment}')
        logger.info(f'* State Size:                              {self.naf_agent.state_size}')
        logger.info(f'* Action Size:                             {self.naf_agent.action_size}')
        logger.info(f'* Size of layers of the Neural Network:    {self.naf_agent.layer_size}')
        logger.info(f'* Batch Size:                              {self.naf_agent.batch_size}')
        logger.info(f'* Buffer Size:                             {self.naf_agent.buffer_size}')
        logger.info(f'* Learning Rate:                           {self.naf_agent.learning_rate}')
        logger.info(f'* Tau:                                     {self.naf_agent.tau}')
        logger.info(f'* Gamma:                                   {self.naf_agent.gamma}')
        logger.info(f'* Update Frequency:                        {self.naf_agent.update_freq}')
        logger.info(f'* Number of Updates:                       {self.naf_agent.num_updates}')
        logger.info(f'* Checkpoint frequency:                    {self.naf_agent.checkpoint_frequency}')
        logger.info(f'* Device:                                  {self.naf_agent.device}')

    def test_trained_model(self, n_episodes: int, frames: int) -> None:
        """
        Tests a previously trained agent through the execution of {n_episodes} test episodes,
        for {frames} timesteps each. When the test concludes, the results of the test are logged on terminal.
        Args:
            n_episodes: Number of test episodes to execute.
            frames: Number of timesteps per test episode.
        Raises:
            ConfigurationIncomplete: Either the Environment class or the NAFAgent class are not initialized.
        """
        # Check if Environment and NAFAgent initialized
        if not self.naf_agent or not self.env:
            raise ConfigurationIncomplete

        # Initialize Test result's history
        results, num_collisions = list(), 0

        for _ in range(n_episodes):
            state = self.env.reset()

            for frame in range(frames):
                action = self.naf_agent.act(state)
                next_state, reward, done = self.env.step(action)
                state = next_state
                if done:
                    if reward == 250:
                        results.append((True, frame))
                        break
                    else:
                        results.append((False, frame))
                        num_collisions += 1
                        break

                if frame == frames - 1 and not done:
                    results.append((False, frame))
                    break

            logger.info('Test Episode number {ep} completed\n'.format(ep=_ + 1))

        logger.info('RESULTS OF THE TEST:')
        for i, result in enumerate(results):
            logger.info(f'Results of Iteration {i + 1}: COMPLETED: {result[0]}. FRAMES: {result[1]}')

        logger.info(f'Number of successful executions: '
                    f'{[res[0] for res in results].count(True)}/{len(results)}  '
                    f'({([res[0] for res in results].count(True) / len(results)) * 100}%)')
        logger.info(f'Average number of frames required to complete an episode: '
                    f'{np.mean(np.array([res[1] for res in results if res[0]]))}')
        logger.info(f'Number of episodes terminated because of collisions: {num_collisions}')

    def initialize_environment(
            self,
            manipulator_file: str,
            endeffector_index: int,
            fixed_joints: List[int],
            involved_joints: List[int],
            target_position: List[float],
            obstacle_position: List[float],
            initial_joint_positions: List[float] = None,
            initial_positions_variation_range: List[float] = None,
            max_force: float = 200.,
            visualize: bool = True) -> None:
        """
        Initialize the Environment by creating an instance of the Environment class.
        Args:
            manipulator_file: Path to the manipulator's URDF or SDF file.
            endeffector_index: Index of the manipulator's end-effector.
            fixed_joints: List containing the indices of every joint not involved in the training.
            involved_joints: List containing the indices of every joint involved in the training.
            target_position: List containing the position of the target object, as 3D Cartesian coordinates.
            obstacle_position: List containing the position of the obstacle, as 3D Cartesian coordinates.
            initial_joint_positions: List containing as many items as the number of joints of the manipulator.
                Each item in the list corresponds to the initial position wanted for the joint with that same index.
            initial_positions_variation_range: List containing as many items as the number of joints of the manipulator.
                Each item in the list corresponds to the variation range wanted for the joint with that same index.
            max_force: Maximum force to be applied on the joints.
            visualize: Visualization mode.
        Raises:
            InvalidEnvironmentParameter: One or more parameters provided for the Environment initialization are invalid.
            InvalidManipulatorFile: The URDF/SDF file provided does not exist or cannot be loaded
                into the Pybullet Environment.
        """
        logger.debug('Initializing Pybullet Environment...')

        environment_config = EnvironmentConfiguration(
            endeffector_index=endeffector_index,
            fixed_joints=fixed_joints,
            involved_joints=involved_joints,
            target_position=target_position,
            obstacle_position=obstacle_position,
            initial_joint_positions=initial_joint_positions,
            initial_positions_variation_range=initial_positions_variation_range,
            max_force=max_force,
            visualize=visualize)
        self.env = Environment(manipulator_file=manipulator_file,
                               environment_config=environment_config)

        logger.info('Pybullet Environment successfully initialized')
        logger.debug(f'* The NAF Agent can now be initialized via the initialize_naf_agent() method')

    def delete_environment(self) -> None:
        """
        Deletes the existing Environment instance, and disconnects the current Pybullet instance.
        """
        if not self.env:
            logger.error('No existing instance of Environment found')
            return

        p.disconnect(self.env.physics_client)
        self.env = None
        logger.info('Environment instance has been successfully removed')

    def initialize_naf_agent(self, checkpoint_frequency: int = 500, seed: int = 0) -> None:
        """
        Initialize the NAF Agent by creating an instance of the NAFAgent class.
        Args:
            checkpoint_frequency: Number of episodes required to generate a checkpoint.
            seed: Random seed.
        Raises:
            EnvironmentNotInitialized: Environment class has not been initialized.
            InvalidNAFAgentParameter: Either "checkpoint_frequency" or "seed" parameters have an invalid value/type.
        """
        if not self.env:
            raise EnvironmentNotInitialized

        if not isinstance(checkpoint_frequency, int) or not isinstance(seed, int):
            raise InvalidNAFAgentParameter('Checkpoint Frequency or Seed received is not an integer')

        logger.debug('Initializing NAF Agent...')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.naf_agent = NAFAgent(environment=self.env,
                                  state_size=self.env.observation_space.shape[0],
                                  action_size=self.env.action_space.shape[0],
                                  layer_size=256,
                                  batch_size=self._hyperparameters.batch_size,
                                  buffer_size=self._hyperparameters.buffer_size,
                                  learning_rate=self._hyperparameters.learning_rate,
                                  tau=self._hyperparameters.tau,
                                  gamma=self._hyperparameters.gamma,
                                  update_freq=self._hyperparameters.update_freq,
                                  num_updates=self._hyperparameters.num_updates,
                                  checkpoint_frequency=checkpoint_frequency,
                                  device=device,
                                  seed=seed)

        logger.info('NAF Agent successfully initialized')
        logger.debug('* The Robotic Manipulator training can now be launched via the run_training() method')

    def delete_naf_agent(self) -> None:
        """
        Deletes the existing NAFAgent instance.
        """
        if not self.naf_agent:
            logger.error('No existing instance of NAFAgent found')
            return

        self.naf_agent = None
        logger.info('NAFAgent instance has been successfully removed')

    def run_training(self, episodes: int, frames: Optional[int] = 500, verbose: bool = True):
        """
        Execute a training on the configured Environment with the configured NAF Agent. The training is
        executed for {episodes}, and for {frames} timesteps per episode.
        Args:
            episodes: Maximum number of episodes to execute.
            frames: Maximum number of timesteps to execute per episode.
            verbose: Verbose mode.\n
            - If set to True, each timestep will log information about the current state,
            the action chosen from that state, the current reward and the cummulative reward until that timestep.
            In addition, each time an episode ends information about the total reward obtained, the number of
            frames required to complete the episode, the mean reward obtained along the episode and the
            execution time of the episode are logged.\n
            - If set to False, each time an episode ends information about the total reward obtained, the number of
            frames required to complete the episode, the mean reward obtained along the episode and the
            execution time of the episode are logged.\n
            It is recommended to use the verbose mode only in a development/debugging context, since logging
            information for each timestep greatly reduces the visibility of what is happening during the training.
        Raises:
            ConfigurationIncomplete: Either NAFAgent or Environment has not been initialized.
        """
        if not self.naf_agent or not self.env:
            raise ConfigurationIncomplete
        self.naf_agent.run(frames, episodes, verbose)

    def run_demo_training(self, demo_type: str, verbose: bool = False) -> None:
        """
        Run a demo from a preconfigured Environment and NAFAgent, which shows how the framework works.
        The training is executed for 20 episodes. Do not expect good results, as this is just a demo of the training
        configuration process and the number of episodes is not enough to achieve a good performance in the manipulator.
        Args:
            demo_type: Valid values:\n
            - 'kuka_training': training with the KUKA IIWA Robotic Manipulator.\n
            - 'xarm6_training': training with the XArm6 Robotic Manipulator.\n
            verbose: Verbose mode. Verbose mode functionality is applied on the same way as for the
                run_training() method.
        """
        logger.warning('Both the demo testing and the demo training are executed with the Log level '
                       'set to DEBUG, so that the framework can be understood at a low level.')
        old_level = logger.level
        logger.setLevel(10)

        # CHECK EXISTENT ENVIRONMENT AND NAF AGENT

        # Overwrite previous Environment configuration if present
        if self.env:

            overwrite_env = input('Environment instance found. Overwrite? [Y/n] ').lower() == 'y'
            if not overwrite_env:
                logger.info('Demo could not run due to the presence of a user-configured Environment instance')
                return

            self.delete_environment()

        # Overwrite previous NAFAgent configuration if present
        if self.naf_agent:

            overwrite_naf_agent = input('NAFAgent instance found. Overwrite? [Y/n] ').lower() == 'y'
            if not overwrite_naf_agent:
                logger.info('Demo could not run due to the presence of a user-configured NAFAgent instance')
                return

            self.delete_naf_agent()

        # START DEMO

        if demo_type == 'kuka_training':

            logger.info('Initializing demo Environment instance...')
            self.initialize_environment(manipulator_file='kuka_iiwa/kuka_with_gripper2.sdf',
                                        endeffector_index=13,
                                        fixed_joints=[6, 7, 8, 9, 10, 11, 12, 13],
                                        involved_joints=[0, 1, 2, 3, 4, 5],
                                        target_position=[0.4, 0.85, 0.71],
                                        obstacle_position=[0.45, 0.55, 0.55],
                                        initial_joint_positions=[0.9, 0.45, 0, 0, 0, 0],
                                        initial_positions_variation_range=[0, 0, 0, 0, 0, 0],
                                        visualize=True)

            logger.info('Initializing demo NAFAgent instance...')
            self.initialize_naf_agent()

            logger.info('Running training for 20 episodes. Do not expect good results, '
                        'this is just a demo of the training configuration process')
            self.run_training(20, 400, verbose=verbose)

            # Reset Environment and NAFAgent
            self.delete_environment()
            self.delete_naf_agent()

        elif demo_type == 'xarm6_training':

            logger.info('Initializing demo Environment instance...')
            xarm_path = os.path.join(pybullet_data.getDataPath(), 'xarm/xarm6_with_gripper.urdf')
            self.initialize_environment(manipulator_file=xarm_path,
                                        endeffector_index=12,
                                        fixed_joints=[0, 7, 8, 9, 10, 11, 12, 13],
                                        involved_joints=[1, 2, 3, 4, 5, 6],
                                        target_position=[0.3, 0.47, 0.61],
                                        obstacle_position=[0.25, 0.27, 0.5],
                                        initial_joint_positions=[0., 1., 0., -2.3, 0., 0., 0.],
                                        initial_positions_variation_range=[0, 0, 0, 0.3, 1, 1, 1],
                                        visualize=True)

            logger.info('Initializing demo NAFAgent instance...')
            self.initialize_naf_agent()

            logger.info('Running training for 20 episodes. Do not expect good results, '
                        'this is just a demo of the confiuration process')
            self.run_training(20, 400, verbose=verbose)

            # Reset Environment and NAFAgent
            self.delete_environment()
            self.delete_naf_agent()

        else:
            logger.error('Incorrect demo type!')

        # Reset log level
        logger.setLevel(old_level)
        logger.warning('Log level has been reset to its original value')

    def run_demo_testing(self, demo_type: str) -> None:
        """
        Run a demo testing from a preconfigured Environment and NAFAgent, which shows how a robotic
        manipulator learns with the framework. The demo loads pretrained weights and executes 50 test episodes.
        Args:
            demo_type: Valid values:\n
            - 'kuka_testing': testing with the KUKA IIWA Robotic Manipulator.\n
            - 'xarm6_testing': testing with the XArm6 Robotic Manipulator.\n
        """
        logger.warning('Both the demo testing and the demo training are executed with the Log level '
                       'set to DEBUG, so that the framework can be understood at a low level.')
        old_level = logger.level
        logger.setLevel(10)

        # CHECK EXISTENT ENVIRONMENT AND NAF AGENT

        # Overwrite previous Environment configuration if present
        if self.env:

            overwrite_env = input('Environment instance found. Overwrite? [Y/n] ').lower() == 'y'
            if not overwrite_env:
                logger.info('Demo could not run due to the presence of a user-configured Environment instance')
                return

            self.delete_environment()

        # Overwrite previous NAFAgent configuration if present
        if self.naf_agent:

            overwrite_naf_agent = input('NAFAgent instance found. Overwrite? [Y/n] ').lower() == 'y'
            if not overwrite_naf_agent:
                logger.info('Demo could not run due to the presence of a user-configured NAFAgent instance')
                return

            self.delete_naf_agent()

        # START DEMO

        if demo_type == 'kuka_testing':

            logger.info('Initializing demo Environment instance...')
            kuka_path = os.path.join(pybullet_data.getDataPath(), 'kuka_iiwa/kuka_with_gripper2.sdf')
            self.initialize_environment(manipulator_file=kuka_path,
                                        endeffector_index=13,
                                        fixed_joints=[6, 7, 8, 9, 10, 11, 12, 13],
                                        involved_joints=[0, 1, 2, 3, 4, 5],
                                        target_position=[0.4, 0.85, 0.71],
                                        obstacle_position=[0.45, 0.55, 0.55],
                                        initial_joint_positions=[0.9, 0.45, 0, 0, 0, 0],
                                        initial_positions_variation_range=[0, 0, .5, .5, .5, .5])

            logger.info('Initializing demo NAFAgent instance...')
            self.initialize_naf_agent()

            logger.info('Loading demo pretrained parameters')
            self.load_pretrained_parameters_from_weights_file(os.path.dirname(
                os.path.realpath(__file__)) + '/naf_components/demo_weights/weights_kuka.p')

            logger.info('Running 50 test episodes...')
            self.test_trained_model(50, 750)

            # Reset Environment and NAFAgent
            self.delete_environment()
            self.delete_naf_agent()

        elif demo_type == 'xarm6_testing':

            logger.info('Initializing demo Environment instance...')
            xarm_path = os.path.join(pybullet_data.getDataPath(), 'xarm/xarm6_with_gripper.urdf')
            self.initialize_environment(manipulator_file=xarm_path,
                                        endeffector_index=12,
                                        fixed_joints=[0, 7, 8, 9, 10, 11, 12, 13],
                                        involved_joints=[1, 2, 3, 4, 5, 6],
                                        target_position=[0.3, 0.47, 0.61],
                                        obstacle_position=[0.25, 0.27, 0.5],
                                        initial_joint_positions=[0., 1., 0., -2.3, 0., 0., 0.],
                                        initial_positions_variation_range=[0, 0, 0, 0.3, 1, 1, 1],
                                        max_force=200,
                                        visualize=True)

            logger.info('Initializing demo NAFAgent instance...')
            self.initialize_naf_agent()

            logger.info('Loading demo pretrained parameters')
            self.load_pretrained_parameters_from_weights_file(
                os.path.dirname(os.path.realpath(__file__)) + '/naf_components/demo_weights/weights_xarm6.p')

            logger.info('Running 50 test episodes...')
            self.test_trained_model(50, 750)

            # Reset Environment and NAFAgent
            self.delete_environment()
            self.delete_naf_agent()

        else:
            logger.error('Incorrect demo type!')

        # Reset log level
        logger.setLevel(old_level)
        logger.warning('Log level has been reset to its original value')
