import numpy as np
import torch
from mock import MagicMock, patch

from robotic_manipulator_rloa.naf_components.naf_neural_network import NAF


@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.manual_seed')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.nn.Module')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.nn.Linear')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.nn.BatchNorm1d')
def test_naf(mock_batchnorm: MagicMock,
             mock_linear: MagicMock,
             mock_nn: MagicMock,
             mock_manual_seed: MagicMock) -> None:
    """Test for the NAF class constructor"""
    mock_manual_seed.return_value = 'manual_seed'
    mock_linear.side_effect = [
        'linear_for_input_layer',
        'linear_for_hidden_layer',
        'linear_for_action_values',
        'linear_for_value',
        'linear_for_matrix_entries'
    ]
    mock_batchnorm.side_effect = [
        'batchnorm_for_bn1',
        'batchnorm_for_bn2'
    ]

    naf = NAF(state_size=10, action_size=5, layer_size=128, seed=0, device='cpu')

    assert naf.seed == 'manual_seed'
    assert naf.state_size == 10
    assert naf.action_size == 5
    assert naf.device == 'cpu'
    assert naf.input_layer == 'linear_for_input_layer'
    assert naf.bn1 == 'batchnorm_for_bn1'
    assert naf.hidden_layer == 'linear_for_hidden_layer'
    assert naf.bn2 == 'batchnorm_for_bn2'
    assert naf.action_values == 'linear_for_action_values'
    assert naf.value == 'linear_for_value'
    assert naf.matrix_entries == 'linear_for_matrix_entries'
    mock_manual_seed.assert_any_call(0)
    mock_linear.assert_any_call(in_features=10, out_features=128)
    mock_linear.assert_any_call(in_features=128, out_features=128)
    mock_linear.assert_any_call(in_features=128, out_features=5)
    mock_linear.assert_any_call(in_features=128, out_features=1)
    mock_linear.assert_any_call(in_features=128, out_features=15)
    mock_batchnorm.assert_any_call(128)
    mock_batchnorm.assert_any_call(128)


def test_naf__forward() -> None:
    """Test for the forward() method of the NAF class"""
    device = torch.device('cpu')
    naf = NAF(state_size=10, action_size=5, layer_size=256, seed=0, device=device)
    states = torch.from_numpy(np.stack(
        [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
    )).float().to(device)
    actions = torch.from_numpy(np.vstack(
        [np.array([0, 1, 2, 3, 4]), np.array([10, 11, 12, 13, 14])]
    )).long().to(device)

    action, q, v = naf(states, actions)

    assert q.tolist() == [[-35.50931930541992], [-638.494873046875]]
    assert v.tolist() == [[0.5665180683135986], [-0.08311141282320023]]
