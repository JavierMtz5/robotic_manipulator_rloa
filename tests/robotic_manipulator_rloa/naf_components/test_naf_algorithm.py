import json

from mock import MagicMock, patch, mock_open
import pytest

from robotic_manipulator_rloa.naf_components.naf_algorithm import NAFAgent
from robotic_manipulator_rloa.utils.exceptions import MissingWeightsFile


@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.time')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.clip_grad_norm_')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.F')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.torch')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.logger')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.Environment')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.os')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.random')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.NAF')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.ReplayBuffer')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.optim.Adam')
def test_naf_agent(mock_optim: MagicMock,
                   mock_replay_buffer: MagicMock,
                   mock_naf: MagicMock,
                   mock_random: MagicMock,
                   mock_os: MagicMock,
                   mock_environment: MagicMock,
                   mock_logger: MagicMock,
                   mock_torch: MagicMock,
                   mock_F: MagicMock,
                   mock_clip_grad_norm_: MagicMock,
                   mock_time: MagicMock) -> None:
    """
    Test for the NAFAgent class constructor
    """
    # ================== Constructor ==========================================
    naf_agent = NAFAgent(environment=mock_environment,
                         state_size=10,
                         action_size=5,
                         layer_size=128,
                         batch_size=64,
                         buffer_size=100000,
                         learning_rate=0.001,
                         tau=0.001,
                         gamma=0.001,
                         update_freq=1,
                         num_updates=1,
                         checkpoint_frequency=1,
                         device='cpu',
                         seed=0)

    mock_os.makedirs.return_value = None
    fake_naf_instance = mock_naf.return_value
    fake_torch_naf = fake_naf_instance.to.return_value
    fake_network_params = fake_torch_naf.parameters.return_value
    fake_optimizer = mock_optim.return_value
    fake_replay_buffer = mock_replay_buffer.return_value

    assert naf_agent.environment == mock_environment
    assert naf_agent.state_size == 10
    assert naf_agent.action_size == 5
    assert naf_agent.layer_size == 128
    assert naf_agent.buffer_size == 100000
    assert naf_agent.learning_rate == 0.001
    mock_random.seed.assert_any_call(0)
    assert naf_agent.device == 'cpu'
    assert naf_agent.tau == 0.001
    assert naf_agent.gamma == 0.001
    assert naf_agent.update_freq == 1
    assert naf_agent.num_updates == 1
    assert naf_agent.batch_size == 64
    assert naf_agent.checkpoint_frequency == 1
    assert naf_agent.qnetwork_main == fake_torch_naf
    assert naf_agent.qnetwork_target == fake_torch_naf
    mock_naf.assert_any_call(10, 5, 128, 0, 'cpu')
    assert naf_agent.optimizer == fake_optimizer
    mock_optim.assert_any_call(fake_network_params, lr=0.001)
    assert naf_agent.memory == fake_replay_buffer
    mock_replay_buffer.assert_any_call(100000, 64, 'cpu', 0)
    assert naf_agent.update_t_step == 0

    # ================== initialize_pretrained_agent_from_episode() ===========
    fake_torch_load = mock_torch.load.return_value
    mock_os.path.isfile.return_value = True
    naf_agent.initialize_pretrained_agent_from_episode(0)
    fake_torch_naf.load_state_dict.assert_any_call(fake_torch_load)
    mock_torch.load.assert_any_call('checkpoints/0/weights.p')

    # ================== initialize_pretrained_agent_from_episode() when file is not present
    mock_os.path.isfile.return_value = False
    with pytest.raises(MissingWeightsFile):
        naf_agent.initialize_pretrained_agent_from_episode(0)

    # ================== initialize_pretrained_agent_from_weights_file() ======
    fake_torch_load = mock_torch.load.return_value
    mock_os.path.isfile.return_value = True
    naf_agent.initialize_pretrained_agent_from_weights_file('weights.p')
    fake_torch_naf.load_state_dict.assert_any_call(fake_torch_load)
    mock_torch.load.assert_any_call('weights.p')

    # ================== initialize_pretrained_agent_from_weights_file() when file is not present
    mock_os.path.isfile.return_value = False
    with pytest.raises(MissingWeightsFile):
        naf_agent.initialize_pretrained_agent_from_weights_file('weights.p')


@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.ReplayBuffer')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.Environment')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.NAFAgent.learn')
def test_naf_agent__step(mock_learn: MagicMock,
                         mock_environment: MagicMock,
                         mock_replay_buffer: MagicMock) -> None:
    """Test for the step() method of the NAFAgent class"""
    # ================== step() ===============================================
    naf_agent = NAFAgent(environment=mock_environment,
                         state_size=10,
                         action_size=5,
                         layer_size=128,
                         batch_size=64,
                         buffer_size=100000,
                         learning_rate=0.001,
                         tau=0.001,
                         gamma=0.001,
                         update_freq=1,
                         num_updates=1,
                         checkpoint_frequency=1,
                         device='cpu',
                         seed=0)
    fake_replay_buffer = mock_replay_buffer.return_value
    fake_replay_buffer.__len__.return_value = 100
    fake_experiences = MagicMock()
    fake_replay_buffer.sample.return_value = fake_experiences

    naf_agent.step('state', 'action', 'reward', 'next_state', 'done')

    fake_replay_buffer.add.assert_any_call('state', 'action', 'reward', 'next_state', 'done')
    assert naf_agent.update_t_step == 0
    fake_replay_buffer.sample.assert_any_call()
    mock_learn.assert_any_call(fake_experiences)


@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.Environment')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.torch')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.NAF')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.optim.Adam')
def test_naf_agent__act(mock_optimizer: MagicMock,
                        mock_naf: MagicMock,
                        mock_torch: MagicMock,
                        mock_environment: MagicMock) -> None:
    """Test for the act() method of the NAFAgent class"""
    # ================== act() ================================================
    naf_agent = NAFAgent(environment=mock_environment,
                         state_size=10,
                         action_size=5,
                         layer_size=128,
                         batch_size=64,
                         buffer_size=100000,
                         learning_rate=0.001,
                         tau=0.001,
                         gamma=0.001,
                         update_freq=1,
                         num_updates=1,
                         checkpoint_frequency=1,
                         device='cpu',
                         seed=0)
    fake_state_from_numpy = mock_torch.from_numpy.return_value
    fake_state_float = fake_state_from_numpy.float.return_value
    fake_state_to = fake_state_float.to.return_value
    fake_state_unsqueezed = fake_state_to.unsqueeze.return_value
    fake_action = MagicMock()
    fake_naf_instance = mock_naf.return_value
    fake_torch_naf = fake_naf_instance.to.return_value
    fake_torch_naf.return_value = (fake_action, None, None)
    fake_action_cpu = fake_action.cpu.return_value
    fake_action_squeeze = fake_action_cpu.squeeze.return_value
    fake_action_numpy = fake_action_squeeze.numpy.return_value
    fake_no_grad = MagicMock(__enter__=MagicMock())
    mock_torch.no_grad.return_value = fake_no_grad

    assert naf_agent.act('state') == fake_action_numpy

    fake_torch_naf.eval.assert_any_call()
    mock_torch.no_grad.assert_any_call()
    fake_torch_naf.assert_any_call(fake_state_unsqueezed)
    fake_torch_naf.train.assert_any_call()


@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.Environment')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.torch')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.NAF')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.optim.Adam')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.F')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.clip_grad_norm_')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.NAFAgent.soft_update')
def test_naf_agent__learn(mock_soft_update: MagicMock,
                          mock_clip_grad_norm_: MagicMock,
                          mock_F: MagicMock,
                          mock_optimizer: MagicMock,
                          mock_naf: MagicMock,
                          mock_torch: MagicMock,
                          mock_environment: MagicMock) -> None:
    """Test for the learn() method in the NAFAgent class"""
    # ================== learn() ==============================================
    naf_agent = NAFAgent(environment=mock_environment,
                         state_size=10,
                         action_size=5,
                         layer_size=128,
                         batch_size=64,
                         buffer_size=100000,
                         learning_rate=0.001,
                         tau=0.001,
                         gamma=0.001,
                         update_freq=1,
                         num_updates=1,
                         checkpoint_frequency=1,
                         device='cpu',
                         seed=0)
    fake_states, fake_actions, fake_rewards, fake_next_states, fake_dones = \
        MagicMock(), MagicMock(), 1, MagicMock(), MagicMock()
    fake_experiences = (fake_states, fake_actions, fake_rewards, fake_next_states, fake_dones)
    fake_v, fake_q_estimate = 1, 1
    fake_naf_instance = mock_naf.return_value
    fake_torch_naf = fake_naf_instance.to.return_value
    fake_network_params = fake_torch_naf.parameters.return_value
    fake_torch_naf.return_value = (None, fake_q_estimate, fake_v)
    fake_optimizer = mock_optimizer.return_value
    fake_loss = mock_F.mse_loss.return_value

    naf_agent.learn(fake_experiences)

    fake_optimizer.zero_grad.assert_any_call()
    fake_torch_naf.assert_any_call(fake_next_states)
    fake_torch_naf.assert_any_call(fake_states, fake_actions)
    mock_F.mse_loss.assert_any_call(fake_q_estimate, 1.001)
    fake_loss.backward.assert_any_call()
    mock_clip_grad_norm_.assert_any_call(fake_network_params, 1)
    fake_optimizer.step.assert_any_call()
    mock_soft_update.assert_any_call(fake_torch_naf, fake_torch_naf)


@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.Environment')
def test_naf_agent__soft_update(mock_environment: MagicMock) -> None:
    """Test for the soft_update() method for the NAFAgent class"""
    # ================== soft_update() ========================================
    naf_agent = NAFAgent(environment=mock_environment,
                         state_size=10,
                         action_size=5,
                         layer_size=128,
                         batch_size=64,
                         buffer_size=100000,
                         learning_rate=0.001,
                         tau=0.001,
                         gamma=0.001,
                         update_freq=1,
                         num_updates=1,
                         checkpoint_frequency=1,
                         device='cpu',
                         seed=0)
    fake_main_nn, fake_target_nn = MagicMock(), MagicMock()
    fake_param_value = MagicMock(value=1)
    fake_main_params, fake_target_params = MagicMock(data=fake_param_value), MagicMock(data=fake_param_value)
    fake_main_nn.parameters.return_value = [fake_main_params]
    fake_target_nn.parameters.return_value = [fake_target_params]

    naf_agent.soft_update(fake_main_nn, fake_target_nn)


@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.logger')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.Environment')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.optim.Adam')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.time')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.os')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.torch')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.NAF')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.NAFAgent.act')
@patch('robotic_manipulator_rloa.naf_components.naf_algorithm.NAFAgent.step')
@patch('builtins.open', new_callable=mock_open())
def test_naf_agent__run(mock_open_file: MagicMock,
                        mock_step: MagicMock,
                        mock_act: MagicMock,
                        mock_naf: MagicMock,
                        mock_torch: MagicMock,
                        mock_os: MagicMock,
                        mock_time: MagicMock,
                        mock_optimizer: MagicMock,
                        mock_environment: MagicMock,
                        mock_logger: MagicMock) -> None:
    """Test for the run() method for the NAFAgent class"""
    # ================== run() ================================================
    naf_agent = NAFAgent(environment=mock_environment,
                         state_size=10,
                         action_size=5,
                         layer_size=128,
                         batch_size=64,
                         buffer_size=100000,
                         learning_rate=0.001,
                         tau=0.001,
                         gamma=0.001,
                         update_freq=1,
                         num_updates=1,
                         checkpoint_frequency=1,
                         device='cpu',
                         seed=0)
    mock_time.time.return_value = 50
    fake_next_state, fake_reward, fake_done = MagicMock(), 1, MagicMock()
    fake_state = mock_environment.reset.return_value
    fake_action = MagicMock()
    mock_act.return_value = fake_action
    mock_environment.step.return_value = (fake_next_state, fake_reward, fake_done)
    mock_step.return_value = None
    mock_os.makedirs.return_value = None
    fake_naf_instance = mock_naf.return_value
    fake_torch_naf = fake_naf_instance.to.return_value
    fake_torch_naf.return_value = (fake_action, None, None)
    fake_torch_naf.state_dict.return_value = 'params_dict'

    assert naf_agent.run(1, 1, True) == {1: (1, 1)}

    mock_time.time.assert_any_call()
    mock_environment.reset.assert_any_call(True)
    mock_act.assert_any_call(fake_state)
    mock_environment.step.assert_any_call(fake_action)
    mock_step.assert_any_call(fake_state, fake_action, fake_reward, fake_next_state, fake_done)
    mock_os.makedirs.assert_any_call('checkpoints/1/', exist_ok=True)
    fake_torch_naf.state_dict.assert_any_call()
    mock_torch.save.assert_any_call('params_dict', 'checkpoints/1/weights.p')
    mock_open_file.assert_called_once_with('checkpoints/1/scores.txt', 'w')
    mock_open_file.return_value.__enter__().write.assert_called_once_with(json.dumps({1: (1, 1)}))
    mock_torch.save.assert_any_call('params_dict', 'model.p')
