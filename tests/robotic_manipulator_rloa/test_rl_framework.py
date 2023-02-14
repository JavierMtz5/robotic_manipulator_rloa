from mock import patch, MagicMock, mock_open
from typing import Union
import pytest
import numpy as np
import json
from robotic_manipulator_rloa.rl_framework import HyperParameters, ManipulatorFramework
from robotic_manipulator_rloa.utils.exceptions import (
    InvalidHyperParameter,
    EnvironmentNotInitialized,
    NAFAgentNotInitialized,
    ConfigurationIncomplete,
    InvalidNAFAgentParameter
)


def test_hyperparameters() -> None:
    """Test for Hyperparameters Dataclass"""
    hyperparameters = HyperParameters(buffer_size=1,
                                      batch_size=64,
                                      gamma=0.5,
                                      tau=0.01,
                                      learning_rate=0.001,
                                      update_freq=2,
                                      num_updates=3)
    assert hyperparameters.buffer_size == 1
    assert hyperparameters.batch_size == 64
    assert hyperparameters.gamma == 0.5
    assert hyperparameters.tau == 0.01
    assert hyperparameters.learning_rate == 0.001
    assert hyperparameters.update_freq == 2
    assert hyperparameters.num_updates == 3


@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework(mock_logger: MagicMock) -> None:
    """Test for the ManipulatorFramework class constructor"""
    mf = ManipulatorFramework()
    assert mf.env is None
    assert mf.naf_agent is None
    assert mf._hyperparameters.buffer_size == 100000
    assert mf._hyperparameters.batch_size == 128
    assert mf._hyperparameters.gamma == 0.99
    assert mf._hyperparameters.tau == 0.001
    assert mf._hyperparameters.learning_rate == 0.001
    assert mf._hyperparameters.update_freq == 1
    assert mf._hyperparameters.num_updates == 1


@pytest.mark.parametrize('log_level', [10, 20, 30, 40, 50])
@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__set_log_level(mock_logger: MagicMock,
                                             log_level: int) -> None:
    """Test for the set_log_level() method of the ManipulatorFramework class"""
    mock_logger.level = 10
    ManipulatorFramework.set_log_level(log_level)
    mock_logger.setLevel.assert_any_call(log_level)


@pytest.mark.parametrize('log_level', [11, 21, 31, 'str', []])
@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__set_log_level__invalid_level(mock_logger: MagicMock,
                                                            log_level: int) -> None:
    """Test for the set_log_level() method of the ManipulatorFramework class when an invalid
    log level is passed as parameter"""
    mock_logger.level = 10
    ManipulatorFramework.set_log_level(log_level)
    mock_logger.setLevel.assert_not_called()


@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__get_required_hyperparameters(mock_logger: MagicMock) -> None:
    """Test for the get_required_hyperparameters() method of the ManipulatorFramework class"""
    mock_logger.level = 10

    ManipulatorFramework.get_required_hyperparameters()

    mock_logger.debug.assert_any_call('{:<25} (see {:<10})'.format(
        'Buffer Size', 'https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial?hl=es-419'))


@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__get_required_hyperparameters__invalid_log_level(mock_logger: MagicMock) -> None:
    """Test for the get_required_hyperparameters() method of the ManipulatorFramework
    class when log level is set to INFO"""
    mock_logger.level = 20

    ManipulatorFramework.get_required_hyperparameters()

    mock_logger.error.assert_any_call('get_required_hyperparameters() only shows information for DEBUG log level. '
                                      'Try running this method after setting the log level to DEBUG by calling '
                                      'set_log_level(10) class method')


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('robotic_manipulator_rloa.rl_framework.plt')
@patch('builtins.open', new_callable=mock_open())
def test_manipulatorframework__plot_training_rewards(mock_open_file: MagicMock,
                                                     mock_plt: MagicMock,
                                                     mock_logger: MagicMock) -> None:
    """Test for the plot_training_rewards() method of the ManipulatorFramework class"""
    fake_scores = {1: (100, 0), 2: (200, 0)}
    mock_open_file.return_value.__enter__().read.return_value = json.dumps(fake_scores)

    ManipulatorFramework.plot_training_rewards(1, 1)

    mock_open_file.assert_called_once_with('checkpoints/1/scores.txt', 'r')
    mock_open_file.return_value.__enter__().read.assert_called_once()
    mock_plt.plot.assert_any_call(range(2), [100, 200])


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('builtins.open', new_callable=mock_open())
def test_manipulatorframework__plot_training_rewards__filenotfound(mock_open_file: MagicMock,
                                                                   mock_logger: MagicMock) -> None:
    """Test for the plot_training_rewards() method of the ManipulatorFramework class when
    the file provided does not exist"""
    mock_open_file.return_value.__enter__().read.side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError):
        ManipulatorFramework.plot_training_rewards(1, 1)


@pytest.mark.parametrize('hyperparam, value', [
    ('buffer_size', 200),
    ('buffersize', 200),
    ('BUFFER_SIZE', 200),
    ('BUFFERSIZE', 200),
    ('batch_size', 1),
    ('batchsize', 1),
    ('BATCH_SIZE', 1),
    ('BATCHSIZE', 1),
    ('gamma', 0.5),
    ('GAMMA', 0.5),
    ('tau', 1),
    ('TAU', 1),
    ('learning_rate', 1),
    ('learningrate', 1),
    ('LEARNING_RATE', 1),
    ('LEARNINGRATE', 1),
    ('update_freq', 10),
    ('updatefreq', 10),
    ('UPDATE_FREQ', 10),
    ('UPDATEFREQ', 10),
    ('num_update', 20),
    ('numupdate', 20),
    ('NUM_UPDATE', 20),
    ('NUMUPDATE', 20)
])
@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__set_hyperparameter(mock_logger: MagicMock,
                                                  hyperparam: str, value: Union[float, int]) -> None:
    """Test for the set_hyperparameter() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    mf.set_hyperparameter(hyperparam, value)
    if hyperparam in ['buffer_size', 'buffersize', 'BUFFER_SIZE', 'BUFFERSIZE']:
        assert mf._hyperparameters.buffer_size == 200
    elif hyperparam in ['batch_size', 'batchsize', 'BATCH_SIZE', 'BATCHSIZE']:
        assert mf._hyperparameters.batch_size == 1
    elif hyperparam in ['gamma', 'GAMMA']:
        assert mf._hyperparameters.gamma == 0.5
    elif hyperparam in ['tau', 'TAU']:
        assert mf._hyperparameters.tau == 1
    elif hyperparam in ['learning_rate', 'learningrate', 'LEARNING_RATE', 'LEARNINGRATE']:
        assert mf._hyperparameters.learning_rate == 1
    elif hyperparam in ['update_freq', 'updatefreq', 'UPDATE_FREQ', 'UPDATEFREQ']:
        assert mf._hyperparameters.update_freq == 10
    elif hyperparam in ['num_update', 'numupdate', 'NUM_UPDATE', 'NUMUPDATE']:
        assert mf._hyperparameters.num_updates == 20


@pytest.mark.parametrize('hyperparam, value', [
    ('wrong_key', 200),
    ('buffersize', 'str'),
    ('BUFFER_SIZE', []),
    ('BUFFERSIZE', -1),
    ('BUFFERSIZE', 0.5),
    ('batch_size', 'str'),
    ('batchsize', []),
    ('BATCH_SIZE', -1),
    ('BATCHSIZE', 0.5),
    ('gamma', 'str'),
    ('GAMMA', []),
    ('GAMMA', -0.1),
    ('GAMMA', 1.1),
    ('GAMMA', 0),
    ('GAMMA', 1),
    ('tau', 'str'),
    ('TAU', []),
    ('TAU', -0.5),
    ('TAU', -1),
    ('learning_rate', 'str'),
    ('learningrate', []),
    ('LEARNING_RATE', -0.01),
    ('LEARNINGRATE', -1),
    ('update_freq', 'str'),
    ('updatefreq', []),
    ('UPDATE_FREQ', -0.01),
    ('UPDATEFREQ', -1),
    ('num_update', 'str'),
    ('numupdate', []),
    ('NUM_UPDATE', -0.001),
    ('NUMUPDATE', -1)
])
@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__set_hyperparameter__invalid_hyperparam_or_value(mock_logger: MagicMock,
                                                                               hyperparam: str,
                                                                               value: Union[float, int]) -> None:
    """Test for the set_hyperparameter() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    with pytest.raises(InvalidHyperParameter):
        mf.set_hyperparameter(hyperparam, value)


def test_manipulatorframework__load_pretrained_parameters_from_weights_file() -> None:
    """Test for the load_pretrained_parameters_from_weights_file() method of the ManipulatorFramework class"""
    fake_env, fake_naf_agent = MagicMock(), MagicMock()
    mf = ManipulatorFramework()
    with pytest.raises(EnvironmentNotInitialized):
        mf.load_pretrained_parameters_from_weights_file('params_path')

    mf.env = fake_env
    with pytest.raises(NAFAgentNotInitialized):
        mf.load_pretrained_parameters_from_weights_file('params_path')

    mf.naf_agent = fake_naf_agent
    fake_naf_agent.initialize_pretrained_agent_from_weights_file.return_value = None
    mf.load_pretrained_parameters_from_weights_file('params_path')
    fake_naf_agent.initialize_pretrained_agent_from_weights_file.assert_any_call('params_path')


def test_manipulatorframework__load_pretrained_parameters_from_episode() -> None:
    """Test for the load_pretrained_parameters_from_episode() method of the ManipulatorFramework class"""
    fake_env, fake_naf_agent = MagicMock(), MagicMock()
    mf = ManipulatorFramework()
    with pytest.raises(EnvironmentNotInitialized):
        mf.load_pretrained_parameters_from_episode(1)

    mf.env = fake_env
    with pytest.raises(NAFAgentNotInitialized):
        mf.load_pretrained_parameters_from_episode(1)

    mf.naf_agent = fake_naf_agent
    fake_naf_agent.initialize_pretrained_agent_from_episode.return_value = None
    mf.load_pretrained_parameters_from_episode(1)
    fake_naf_agent.initialize_pretrained_agent_from_episode.assert_any_call(1)


@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__get_environment_configuration(mock_logger: MagicMock) -> None:
    """Test for the get_environment_configuration() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    mf.get_environment_configuration()
    mock_logger.error.assert_any_call("Environment is not initialized yet, can't show configuration")
    mf.env = MagicMock(manipulator_file='manipulator_file',
                       endeffector_index=0,
                       fixed_joints=[0],
                       involved_joints=[1],
                       target_position=[0, 0, 0],
                       obstacle_position=[0, 0, 0],
                       initial_joint_positions=[0],
                       initial_positions_variation_range=[0],
                       max_force=1,
                       visualize=False)
    mf.get_environment_configuration()
    mock_logger.info.assert_any_call('Environment Configuration:')


@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__get_nafagent_configuration(mock_logger: MagicMock) -> None:
    """Test for the get_nafagent_configuration() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    mf.get_nafagent_configuration()
    mock_logger.error.assert_any_call("NAFAgent is not initialized yet, can't show configuration")
    mf.naf_agent = MagicMock(environment=MagicMock(),
                             state_size=10,
                             action_size=5,
                             layer_size=128,
                             batch_size=64,
                             buffer_size=1000,
                             learning_rate=0.001,
                             tau=1,
                             gamma=1,
                             update_freq=1,
                             num_updates=1,
                             checkpoint_frequency=5,
                             device='cpu')
    mf.get_nafagent_configuration()
    mock_logger.info.assert_any_call('NAFAgent Configuration:')


@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__test_trained_model(mock_logger: MagicMock) -> None:
    """Test for the test_trained_model() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    with pytest.raises(ConfigurationIncomplete):
        mf.test_trained_model(1, 1)

    fake_env, fake_naf_agent = MagicMock(), MagicMock()
    mf.env, mf.naf_agent = fake_env, fake_naf_agent
    fake_state, fake_action = MagicMock(), MagicMock()
    fake_env.reset.return_value = fake_state
    fake_naf_agent.act.return_value = fake_action
    fake_env.step.side_effect = [
        (MagicMock(), 250, 1),
        (MagicMock(), 1, 1),
        (MagicMock(), -1, 0)
    ]

    mf.test_trained_model(3, 1)

    fake_env.reset.assert_any_call()
    fake_naf_agent.act.assert_any_call(fake_state)
    fake_env.step.assert_any_call(fake_action)
    mock_logger.info.assert_any_call(f'Results of Iteration 1: COMPLETED: True. FRAMES: 0')
    mock_logger.info.assert_any_call(f'Results of Iteration 2: COMPLETED: False. FRAMES: 0')
    mock_logger.info.assert_any_call(f'Results of Iteration 3: COMPLETED: False. FRAMES: 0')
    mock_logger.info.assert_any_call('Number of episodes terminated because of collisions: 1')


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('robotic_manipulator_rloa.rl_framework.EnvironmentConfiguration')
@patch('robotic_manipulator_rloa.rl_framework.Environment')
def test_manipulatorframework__initialize_environment(mock_env: MagicMock,
                                                      mock_env_config: MagicMock,
                                                      mock_logger: MagicMock) -> None:
    """Test for the initialize_environment() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    fake_env_config = MagicMock()
    mock_env_config.return_value = fake_env_config
    mf.initialize_environment(manipulator_file='manipulator_file.urdf',
                              endeffector_index=0,
                              fixed_joints=[0],
                              involved_joints=[1],
                              target_position=[0, 0, 0],
                              obstacle_position=[0, 0, 0],
                              initial_joint_positions=[0],
                              initial_positions_variation_range=[0],
                              max_force=1,
                              visualize=False)
    mock_env_config.assert_any_call(endeffector_index=0,
                                    fixed_joints=[0],
                                    involved_joints=[1],
                                    target_position=[0, 0, 0],
                                    obstacle_position=[0, 0, 0],
                                    initial_joint_positions=[0],
                                    initial_positions_variation_range=[0],
                                    max_force=1,
                                    visualize=False)
    mock_env.assert_any_call(manipulator_file='manipulator_file.urdf',
                             environment_config=fake_env_config)


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('robotic_manipulator_rloa.rl_framework.p')
def test_manipulatorframework__delete_environment(mock_pybullet: MagicMock,
                                                  mock_logger: MagicMock) -> None:
    """Test for the delete_environment() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    mf.delete_environment()
    mock_logger.error.assert_any_call('No existing instance of Environment found')

    fake_env = MagicMock(physics_client='physics_client')
    mf.env = fake_env
    mf.delete_environment()
    mock_pybullet.disconnect.assert_any_call('physics_client')
    assert mf.env is None


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('robotic_manipulator_rloa.rl_framework.torch.device')
@patch('robotic_manipulator_rloa.rl_framework.torch.cuda.is_available')
@patch('robotic_manipulator_rloa.rl_framework.NAFAgent')
def test_manipulatorframework__initialize_naf_agent(mock_nafagent: MagicMock,
                                                    mock_is_available: MagicMock,
                                                    mock_device: MagicMock,
                                                    mock_logger: MagicMock) -> None:
    """Test for the initialize_naf_agent() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    with pytest.raises(EnvironmentNotInitialized):
        mf.initialize_naf_agent()

    mock_is_available.return_value = False
    mock_device.return_value = 'cpu'
    fake_naf_agent, fake_env = MagicMock(), MagicMock(observation_space=np.zeros((3, 1)),
                                                      action_space=np.zeros((3, 1)))
    mock_nafagent.return_value = fake_naf_agent

    mf.env = fake_env

    with pytest.raises(InvalidNAFAgentParameter):
        mf.initialize_naf_agent(checkpoint_frequency='str')
        mf.initialize_naf_agent(seed='str')

    mf.initialize_naf_agent()

    mock_nafagent.assert_any_call(environment=fake_env,
                                  state_size=3,
                                  action_size=3,
                                  layer_size=256,
                                  batch_size=128,
                                  buffer_size=100000,
                                  learning_rate=0.001,
                                  tau=0.001,
                                  gamma=0.99,
                                  update_freq=1,
                                  num_updates=1,
                                  checkpoint_frequency=500,
                                  device='cpu',
                                  seed=0)


@patch('robotic_manipulator_rloa.rl_framework.logger')
def test_manipulatorframework__delete_naf_agent(mock_logger: MagicMock) -> None:
    """Test for the delete_naf_agent() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    mf.delete_naf_agent()
    mock_logger.error.assert_any_call('No existing instance of NAFAgent found')

    fake_naf_agent = MagicMock()
    mf.naf_agent = fake_naf_agent
    mf.delete_naf_agent()
    assert mf.naf_agent is None


def test_manipulatorframework__run_training() -> None:
    """Test for the run_training() method of the ManipulatorFramework class"""
    mf = ManipulatorFramework()
    with pytest.raises(ConfigurationIncomplete):
        mf.run_training(1, 1)

    mf.env = MagicMock()
    fake_naf_agent = MagicMock()
    mf.naf_agent = fake_naf_agent

    mf.run_training(1, 1)

    fake_naf_agent.run.assert_any_call(1, 1, True)


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('robotic_manipulator_rloa.rl_framework.os')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.delete_environment')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.delete_naf_agent')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.initialize_environment')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.initialize_naf_agent')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.run_training')
@patch('robotic_manipulator_rloa.rl_framework.pybullet_data.getDataPath')
@patch('robotic_manipulator_rloa.rl_framework.input')
@pytest.mark.parametrize('demo_type', ['kuka_training', 'xarm6_training', 'wrong_demo_type'])
def test_manipulatorframework__run_demo_training(mock_input: MagicMock,
                                                 mock_get_datapath: MagicMock,
                                                 mock_run_training: MagicMock,
                                                 mock_initialize_naf_agent: MagicMock,
                                                 mock_initialize_environment: MagicMock,
                                                 mock_delete_naf_agent: MagicMock,
                                                 mock_delete_environment: MagicMock,
                                                 mock_os: MagicMock,
                                                 mock_logger: MagicMock,
                                                 demo_type: str) -> None:
    """Test for the run_demo_training() method of the ManipulatorFramework class"""
    mock_logger.setLevel.return_value = None
    mf = ManipulatorFramework()
    mf.env, mf.naf_agent = MagicMock(), MagicMock()
    mock_input.return_value = 'y'
    mock_delete_environment.return_value = None
    mock_delete_naf_agent.return_value = None
    mf.env, mf.naf_agent = MagicMock(), MagicMock()
    mock_initialize_environment.return_value = None
    mock_initialize_naf_agent.return_value = None
    mock_run_training.return_value = None
    mock_get_datapath.return_value = 'pybullet_data/'
    mock_os.path.join.return_value = 'pybullet_data/xarm/xarm6_with_gripper.urdf'

    mf.run_demo_training(demo_type)

    mock_delete_environment.assert_any_call()
    mock_delete_naf_agent.assert_any_call()
    if demo_type == 'kuka_training':
        mock_initialize_environment.assert_any_call(manipulator_file='kuka_iiwa/kuka_with_gripper2.sdf',
                                                    endeffector_index=13,
                                                    fixed_joints=[6, 7, 8, 9, 10, 11, 12, 13],
                                                    involved_joints=[0, 1, 2, 3, 4, 5],
                                                    target_position=[0.4, 0.85, 0.71],
                                                    obstacle_position=[0.45, 0.55, 0.55],
                                                    initial_joint_positions=[0.9, 0.45, 0, 0, 0, 0],
                                                    initial_positions_variation_range=[0, 0, 0, 0, 0, 0],
                                                    visualize=True)
    if demo_type == 'xarm6_training':
        mock_initialize_environment.assert_any_call(manipulator_file='pybullet_data/xarm/xarm6_with_gripper.urdf',
                                                    endeffector_index=12,
                                                    fixed_joints=[0, 7, 8, 9, 10, 11, 12, 13],
                                                    involved_joints=[1, 2, 3, 4, 5, 6],
                                                    target_position=[0.3, 0.47, 0.61],
                                                    obstacle_position=[0.25, 0.27, 0.5],
                                                    initial_joint_positions=[0., 1., 0., -2.3, 0., 0., 0.],
                                                    initial_positions_variation_range=[0, 0, 0, 0.3, 1, 1, 1],
                                                    visualize=True)

    if demo_type == 'wrong_demo_type':
        mock_logger.error.assert_any_call('Incorrect demo type!')
    else:
        mock_initialize_naf_agent.assert_any_call()
        mock_run_training.assert_any_call(20, 400, verbose=False)


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('robotic_manipulator_rloa.rl_framework.input')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.delete_environment')
@pytest.mark.parametrize('input', [['n', 'y'], ['y', 'n']])
def test_manipulatorframework__run_demo_training__present_env_nafagent(mock_delete_environment: MagicMock,
                                                                       mock_input: MagicMock,
                                                                       mock_logger: MagicMock,
                                                                       input: list) -> None:
    """Test for the run_demo_training() method of the ManipulatorFramework class"""
    mock_logger.setLevel.return_value = None
    mf = ManipulatorFramework()
    mf.env, mf.naf_agent = MagicMock(), MagicMock()
    mock_input.side_effect = input

    mf.run_demo_training('kuka_training')

    if input == ['n', 'y']:
        mock_logger.info.assert_any_call(
            'Demo could not run due to the presence of a user-configured Environment instance')
    else:
        mock_logger.info.assert_any_call(
            'Demo could not run due to the presence of a user-configured NAFAgent instance'
        )


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.delete_environment')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.delete_naf_agent')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.initialize_environment')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.initialize_naf_agent')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.load_pretrained_parameters_from_weights_file')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.test_trained_model')
@patch('robotic_manipulator_rloa.rl_framework.pybullet_data.getDataPath')
@patch('robotic_manipulator_rloa.rl_framework.input')
@pytest.mark.parametrize('demo_type', ['kuka_testing', 'xarm6_testing', 'wrong_demo_type'])
def test_manipulatorframework__run_demo_testing(mock_input: MagicMock,
                                                mock_get_datapath: MagicMock,
                                                mock_test_trained_model: MagicMock,
                                                mock_load_pretrained_parameters_from_weights_file: MagicMock,
                                                mock_initialize_naf_agent: MagicMock,
                                                mock_initialize_environment: MagicMock,
                                                mock_delete_naf_agent: MagicMock,
                                                mock_delete_environment: MagicMock,
                                                mock_logger: MagicMock,
                                                demo_type: str) -> None:
    """Test for the run_demo_training() method of the ManipulatorFramework class"""
    mock_logger.setLevel.return_value = None
    mf = ManipulatorFramework()
    mf.env, mf.naf_agent = MagicMock(), MagicMock()
    mock_input.return_value = 'y'
    mock_delete_environment.return_value = None
    mock_delete_naf_agent.return_value = None
    mf.env, mf.naf_agent = MagicMock(), MagicMock()
    mock_initialize_environment.return_value = None
    mock_initialize_naf_agent.return_value = None
    mock_load_pretrained_parameters_from_weights_file.return_value = None
    mock_test_trained_model.return_value = None
    mock_get_datapath.return_value = 'pybullet_data/'

    mf.run_demo_testing(demo_type)

    mock_delete_environment.assert_any_call()
    mock_delete_naf_agent.assert_any_call()
    if demo_type == 'kuka_testing':
        mock_initialize_environment.assert_any_call(manipulator_file='pybullet_data/kuka_iiwa/kuka_with_gripper2.sdf',
                                                    endeffector_index=13,
                                                    fixed_joints=[6, 7, 8, 9, 10, 11, 12, 13],
                                                    involved_joints=[0, 1, 2, 3, 4, 5],
                                                    target_position=[0.4, 0.85, 0.71],
                                                    obstacle_position=[0.45, 0.55, 0.55],
                                                    initial_joint_positions=[0.9, 0.45, 0, 0, 0, 0],
                                                    initial_positions_variation_range=[0, 0, .5, .5, .5, .5])
        mock_initialize_naf_agent.assert_any_call()
        mock_load_pretrained_parameters_from_weights_file.assert_any_call('demo_weights/weights_kuka.p')
        mock_test_trained_model.assert_any_call(50, 750)

    if demo_type == 'xarm6_testing':
        mock_initialize_environment.assert_any_call(manipulator_file='pybullet_data/xarm/xarm6_with_gripper.urdf',
                                                    endeffector_index=12,
                                                    fixed_joints=[0, 7, 8, 9, 10, 11, 12, 13],
                                                    involved_joints=[1, 2, 3, 4, 5, 6],
                                                    target_position=[0.3, 0.47, 0.61],
                                                    obstacle_position=[0.25, 0.27, 0.5],
                                                    initial_joint_positions=[0., 1., 0., -2.3, 0., 0., 0.],
                                                    initial_positions_variation_range=[0, 0, 0, 0.3, 1, 1, 1],
                                                    max_force=200,
                                                    visualize=True)
        mock_initialize_naf_agent.assert_any_call()
        mock_load_pretrained_parameters_from_weights_file.assert_any_call('demo_weights/weights_xarm6.p')
        mock_test_trained_model.assert_any_call(50, 750)

    if demo_type == 'wrong_demo_type':
        mock_logger.error.assert_any_call('Incorrect demo type!')


@patch('robotic_manipulator_rloa.rl_framework.logger')
@patch('robotic_manipulator_rloa.rl_framework.input')
@patch('robotic_manipulator_rloa.rl_framework.ManipulatorFramework.delete_environment')
@pytest.mark.parametrize('input', [['n', 'y'], ['y', 'n']])
def test_manipulatorframework__run_demo_testing__present_env_nafagent(mock_delete_environment: MagicMock,
                                                                      mock_input: MagicMock,
                                                                      mock_logger: MagicMock,
                                                                      input: list) -> None:
    """Test for the run_demo_training() method of the ManipulatorFramework class"""
    mock_logger.setLevel.return_value = None
    mf = ManipulatorFramework()
    mf.env, mf.naf_agent = MagicMock(), MagicMock()
    mock_input.side_effect = input

    mf.run_demo_testing('kuka_testing')

    if input == ['n', 'y']:
        mock_logger.info.assert_any_call(
            'Demo could not run due to the presence of a user-configured Environment instance')
    else:
        mock_logger.info.assert_any_call(
            'Demo could not run due to the presence of a user-configured NAFAgent instance'
        )
