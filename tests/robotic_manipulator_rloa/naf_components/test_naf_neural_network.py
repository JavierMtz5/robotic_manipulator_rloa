from mock import MagicMock, patch, mock_open
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


@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.relu')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.tanh')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.zeros')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.tril_indices')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.matmul')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.MultivariateNormal')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.inverse')
@patch('robotic_manipulator_rloa.naf_components.naf_neural_network.torch.clamp')
def test_naf__forward(mock_clamp: MagicMock,
                      mock_inverse: MagicMock,
                      mock_multivariate_normal: MagicMock,
                      mock_matmul: MagicMock,
                      mock_tril_indices: MagicMock,
                      mock_zeros: MagicMock,
                      mock_tanh: MagicMock,
                      mock_relu: MagicMock) -> None:
    """Test for the forward() method of the NAF class"""
    # naf = NAF(state_size=10, action_size=5, layer_size=128, seed=0, device='cpu')
    # fake_input, fake_action = MagicMock(), MagicMock()
    # fake_input_layer, fake_hidden_layer, fake_action_values, fake_matrix_entries, fake_value, fake_bn1, fake_bn2 = \
    #     [MagicMock() for _ in range(7)]
    # naf.input_layer, naf.hidden_layer, naf.action_values, naf.matrix_entries, naf.value = \
    #     fake_input_layer, fake_hidden_layer, fake_action_values, fake_matrix_entries, fake_value
    # naf.bn1, naf.bn2 = fake_bn1, fake_bn2
    # fake_linear_output_1, fake_linear_output_2 = MagicMock(), MagicMock()
    # fake_action_value_output, fake_matrix_entries_output, fake_value_output = MagicMock(), MagicMock(), MagicMock()
    # x_0 = fake_input_layer.return_value
    # x_1 = fake_bn1.return_value
    # x_2 = fake_hidden_layer.return_value
    # x_3 = fake_bn2.return_value
    # mock_relu.side_effect = [fake_linear_output_1, fake_linear_output_2]
    # x_5 = fake_action_values.return_value
    # x_6 = fake_matrix_entries.return_value
    # fake_value.return_value = fake_value_output
    # mock_tanh.side_effect = [fake_action_value_output, fake_matrix_entries_output]
    # unsqueezed_action_value = MagicMock()
    # fake_action_value_output.unsqueeze.return_value = unsqueezed_action_value
    # fake_L, fake_zeros = MagicMock(), MagicMock()
    # mock_zeros.return_value = fake_zeros
    # fake_zeros.to.return_value = fake_L
    # fake_lower_tri_indices = MagicMock()
    # mock_tril_indices.return_value = fake_lower_tri_indices
    #
    # fake_L_diagonalized, fake_L_exp = MagicMock(), MagicMock()
    # fake_L.diagonal.retun_value = fake_L_diagonalized
    # fake_L_diagonalized.exp_.return_value = fake_L_exp
    # fake_L_transposed = MagicMock()
    # fake_L_exp.transpose.return_value = fake_L_transposed
    #
    # fake_action_unsqueezed = MagicMock()
    # fake_action.unsqueeze.return_value = fake_action_unsqueezed
    # mock_matmul.side_effect = []
    #
    #
    # naf.forward(fake_input, fake_action)
    pass



