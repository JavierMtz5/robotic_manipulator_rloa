# import pytest
# from mock import MagicMock, patch
# from typing import List
# from robotic_manipulator_rloa.environment.environment import EnvironmentConfiguration, Environment
# from robotic_manipulator_rloa.utils.exceptions import InvalidEnvironmentParameter
#
#
# def test_environment_configuration() -> None:
#     """
#     Test for the EnvironmentConfiguration class' happy path.
#     """
#     env_configuration = EnvironmentConfiguration(
#         endeffector_index=0,
#         fixed_joints=[0],
#         involved_joints=[0],
#         target_position=[0, 0, 0],
#         obstacle_position=[0, 0, 0],
#         initial_joint_positions=[0],
#         initial_positions_variation_range=[0],
#         max_force=1,
#         visualize=False
#     )
#     assert env_configuration.endeffector_index == 0
#     assert env_configuration.fixed_joints == [0]
#     assert env_configuration.involved_joints == [0]
#     assert env_configuration.target_position == [0, 0, 0]
#     assert env_configuration.obstacle_position == [0, 0, 0]
#     assert env_configuration.initial_joint_positions[0]
#     assert env_configuration.initial_positions_variation_range == [0]
#     assert env_configuration.max_force == 1
#     assert env_configuration.visualize is False
#
#
# @pytest.mark.parametrize('endeffector_index, fixed_joints, involved_joints, target_position, obstacle_position, '
#                          'initial_joint_positions', 'initial_positions_variation_range', 'max_force', 'visualize',
#                          [
#                              ('wrong_type', [0], [0], [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
#                              (0, 'wrong_type', [0], [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
#                              (0, ['wrong_type'], [0], [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
#                              (0, [0], 'wrong_type', [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
#                              (0, [0], ['wrong_type'], [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
#                              (0, [0], [0], 'wrong_type', [0, 0, 0], [0], [0], 1, False),
#                              (0, [0], [0], ['wrong_type'], [0, 0, 0], [0], [0], 1, False),
#                              (0, [0], [0], [0, 0, 0], 'wrong_type', [0], [0], 1, False),
#                              (0, [0], [0], [0, 0, 0], ['wrong_type'], [0], [0], 1, False),
#                              (0, [0], [0], [0, 0, 0], [0, 0, 0], 'wrong_type', [0], 1, False),
#                              (0, [0], [0], [0, 0, 0], [0, 0, 0], ['wrong_type'], [0], 1, False),
#                              (0, [0], [0], [0, 0, 0], [0, 0, 0], [0], 'wrong_type', 1, False),
#                              (0, [0], [0], [0, 0, 0], [0, 0, 0], [0], ['wrong_type'], 1, False),
#                              (0, [0], [0], [0, 0, 0], [0, 0, 0], [0], [0], 'wrong_type', False),
#                              (0, [0], [0], [0, 0, 0], [0, 0, 0], [0], [0], 1, 'wrong_type'),
#                          ])
# def test_environment_configuration__invalid_params(endeffector_index: int,
#                                                    fixed_joints: List[int],
#                                                    involved_joints: List[int],
#                                                    target_position: List[float],
#                                                    obstacle_position: List[float],
#                                                    initial_joint_positions: List[float],
#                                                    initial_positions_variation_range: List[float],
#                                                    max_force: float,
#                                                    visualize: bool) -> None:
#     """
#     Test for the EnvironmentConfiguration class if invalid parameters are loaded.
#     Args:
#     endeffector_index:
#     fixed_joints:
#     involved_joints:
#     target_position:
#     obstacle_position:
#     initial_joint_positions:
#     initial_positions_variation_range:
#     max_force:
#     visualize:
#     """
#     with pytest.raises(InvalidEnvironmentParameter):
#         env_configuration = EnvironmentConfiguration(
#             endeffector_index=endeffector_index,
#             fixed_joints=fixed_joints,
#             involved_joints=involved_joints,
#             target_position=target_position,
#             obstacle_position=obstacle_position,
#             initial_joint_positions=initial_joint_positions,
#             initial_positions_variation_range=initial_positions_variation_range,
#             max_force=max_force,
#             visualize=visualize
#         )
#
#
# @patch('robotic_manipulator_rloa.environment.environment.p.loadURDF')
# @patch('robotic_manipulator_rloa.environment.environment.p.connect')
# @patch('robotic_manipulator_rloa.environment.environment.p.setGravity')
# @patch('robotic_manipulator_rloa.environment.environment.p.setRealTimeSimulation')
# @patch('robotic_manipulator_rloa.environment.environment.p.setAdditionalSearchPath')
# @patch('robotic_manipulator_rloa.environment.environment.p.getNumJoints')
# @patch('robotic_manipulator_rloa.environment.environment.p.getJointInfo')
# @patch('robotic_manipulator_rloa.environment.environment.Environment.print_table')
# def test_environment__urdf_file(mock_print_table: MagicMock,
#                                 mock_getJointInfo: MagicMock,
#                                 mock_getNumJoints: MagicMock,
#                                 mock_setAdditionalSearchPath: MagicMock,
#                                 mock_setRealTimeSimulation: MagicMock,
#                                 mock_setGravity: MagicMock,
#                                 mock_connect: MagicMock,
#                                 mock_loadurdf: MagicMock) -> None:
#     env_configuration = EnvironmentConfiguration(
#         endeffector_index=0,
#         fixed_joints=[0],
#         involved_joints=[0],
#         target_position=[0, 0, 0],
#         obstacle_position=[0, 0, 0],
#         initial_joint_positions=[0],
#         initial_positions_variation_range=[0],
#         max_force=1,
#         visualize=False
#     )
#     mock_connect.return_value = 'physics_client'
#     mock_setGravity.return_value = None
#     mock_setRealTimeSimulation.return_value = None
#     mock_setAdditionalSearchPath.return_value = None
#     mock_loadurdf.return_value = 'urdf_env'
#     mock_getNumJoints.return_value = ['1']
#     mock_getJointInfo.return_value = [None, None]
#
#     env = Environment('fake_file.urdf', env_configuration)
#
#     assert env.manipulator_file == 'fake_file.urdf'
#     assert env.visualize == env_configuration.visualize
#     assert env.physics_client == 'physics_client'
#
#
#
#
