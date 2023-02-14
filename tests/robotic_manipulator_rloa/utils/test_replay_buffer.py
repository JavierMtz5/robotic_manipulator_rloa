import pytest
from mock import MagicMock, patch
from collections import deque, namedtuple
import numpy as np

from robotic_manipulator_rloa.utils.replay_buffer import ReplayBuffer


@patch('robotic_manipulator_rloa.utils.replay_buffer.namedtuple')
@patch('robotic_manipulator_rloa.utils.replay_buffer.deque')
@patch('robotic_manipulator_rloa.utils.replay_buffer.random')
def test_replaybuffer(mock_random: MagicMock,
                      mock_deque: MagicMock,
                      mock_namedtuple: MagicMock) -> None:
    """Test for the ReplayBuffer class"""
    replay_buffer = ReplayBuffer(buffer_size=10000,
                                 batch_size=128,
                                 device='cpu',
                                 seed=0)
    assert replay_buffer.device == 'cpu'
    assert replay_buffer.memory == mock_deque.return_value
    assert replay_buffer.batch_size == 128
    assert replay_buffer.experience == mock_namedtuple.return_value

    mock_deque.assert_any_call(maxlen=10000)
    mock_namedtuple.assert_any_call("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    mock_random.seed.assert_any_call(0)


def test_replaybuffer__add() -> None:
    """Test for the add() method of the ReplayBuffer class"""
    replay_buffer = ReplayBuffer(buffer_size=10000,
                                 batch_size=128,
                                 device='cpu',
                                 seed=0)
    named_tuple = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    deque_ = deque(maxlen=10)
    experience = named_tuple('state', 'action', 'reward', 'next_state', 'done')
    deque_.append(experience)

    replay_buffer.add('state', 'action', 'reward', 'next_state', 'done')

    assert replay_buffer.memory == deque_


@patch('robotic_manipulator_rloa.utils.replay_buffer.torch')
@patch('robotic_manipulator_rloa.utils.replay_buffer.random.sample')
@patch('robotic_manipulator_rloa.utils.replay_buffer.np.stack')
@patch('robotic_manipulator_rloa.utils.replay_buffer.np.vstack')
def test_replaybuffer__sample(mock_vstack: MagicMock,
                              mock_stack: MagicMock,
                              mock_random_sample: MagicMock,
                              mock_torch: MagicMock) -> None:
    """Test for the sample() method for the ReplayBuffer class"""
    replay_buffer = ReplayBuffer(buffer_size=10000,
                                 batch_size=1,
                                 device='cpu',
                                 seed=0)
    fake_state = np.array([0, 1])
    fake_action = np.array([2, 3])
    fake_reward = 1.5
    fake_next_state = np.array([4, 5])
    fake_done = 0
    mock_random_sample.return_value = [MagicMock(state=fake_state,
                                                 action=fake_action,
                                                 reward=fake_reward,
                                                 next_state=fake_next_state,
                                                 done=fake_done)]
    fake_stack_state, fake_stack_next_state = MagicMock(), MagicMock()
    fake_vstack_action, fake_vstack_reward, fake_vstack_done = MagicMock(), MagicMock(), MagicMock()

    fake_vstack_done.astype.return_value = 'done_astype'
    mock_stack.side_effect = [fake_stack_state, fake_stack_next_state]
    mock_vstack.side_effect = [fake_vstack_action, fake_vstack_reward, fake_vstack_done]

    fake_from_numpy_state, fake_from_numpy_action, fake_from_numpy_reward, \
        fake_from_numpy_next_state, fake_from_numpy_done = MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    mock_torch.from_numpy.side_effect = [
        fake_from_numpy_state, fake_from_numpy_action, fake_from_numpy_reward,
        fake_from_numpy_next_state, fake_from_numpy_done]

    output_state_float = fake_from_numpy_state.float.return_value
    output_state = output_state_float.to.return_value

    output_action_long = fake_from_numpy_action.long.return_value
    output_action = output_action_long.to.return_value

    output_reward_float = fake_from_numpy_reward.float.return_value
    output_reward = output_reward_float.to.return_value

    output_next_state_float = fake_from_numpy_next_state.float.return_value
    output_next_state = output_next_state_float.to.return_value

    output_done_float = fake_from_numpy_done.float.return_value
    output_done = output_done_float.to.return_value

    assert replay_buffer.sample() == (output_state, output_action, output_reward, output_next_state, output_done)


def test_replaybuffer__len__() -> None:
    """Test for the __len__() method of the ReplayBuffer class"""
    replay_buffer = ReplayBuffer(buffer_size=10000,
                                 batch_size=1,
                                 device='cpu',
                                 seed=0)
    assert len(replay_buffer) == 0
    replay_buffer.add('state', 'action', 'reward', 'next_state', 'done')
    assert len(replay_buffer) == 1
