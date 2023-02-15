import pytest
from mock import MagicMock, patch
from typing import List
from typing import Union
import pybullet as p
import numpy as np
from robotic_manipulator_rloa.environment.environment import EnvironmentConfiguration, Environment
from robotic_manipulator_rloa.utils.exceptions import InvalidEnvironmentParameter, InvalidManipulatorFile


@pytest.mark.parametrize('initial_pos_and_var_range', [[0], None])
def test_environment_configuration(initial_pos_and_var_range: Union[list, None]) -> None:
    """
    Test for the EnvironmentConfiguration class' happy path.
    """
    env_configuration = EnvironmentConfiguration(
        endeffector_index=0,
        fixed_joints=[0],
        involved_joints=[0],
        target_position=[0, 0, 0],
        obstacle_position=[0, 0, 0],
        initial_joint_positions=initial_pos_and_var_range,
        initial_positions_variation_range=initial_pos_and_var_range,
        max_force=1,
        visualize=False
    )
    assert env_configuration.endeffector_index == 0
    assert env_configuration.fixed_joints == [0]
    assert env_configuration.involved_joints == [0]
    assert env_configuration.target_position == [0, 0, 0]
    assert env_configuration.obstacle_position == [0, 0, 0]
    assert env_configuration.initial_joint_positions == initial_pos_and_var_range
    assert env_configuration.initial_positions_variation_range == initial_pos_and_var_range
    assert env_configuration.max_force == 1
    assert env_configuration.visualize is False


@pytest.mark.parametrize('endeffector_index, fixed_joints, involved_joints, target_position, obstacle_position, '
                         'initial_joint_positions, initial_positions_variation_range, max_force, visualize',
                         [
                             ('wrong_type', [0], [0], [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
                             (0, 'wrong_type', [0], [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
                             (0, ['wrong_type'], [0], [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
                             (0, [0], 'wrong_type', [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
                             (0, [0], ['wrong_type'], [0, 0, 0], [0, 0, 0], [0], [0], 1, False),
                             (0, [0], [0], 'wrong_type', [0, 0, 0], [0], [0], 1, False),
                             (0, [0], [0], ['wrong_type'], [0, 0, 0], [0], [0], 1, False),
                             (0, [0], [0], [0, 0, 0], 'wrong_type', [0], [0], 1, False),
                             (0, [0], [0], [0, 0, 0], ['wrong_type'], [0], [0], 1, False),
                             (0, [0], [0], [0, 0, 0], [0, 0, 0], 'wrong_type', [0], 1, False),
                             (0, [0], [0], [0, 0, 0], [0, 0, 0], ['wrong_type'], [0], 1, False),
                             (0, [0], [0], [0, 0, 0], [0, 0, 0], [0], 'wrong_type', 1, False),
                             (0, [0], [0], [0, 0, 0], [0, 0, 0], [0], ['wrong_type'], 1, False),
                             (0, [0], [0], [0, 0, 0], [0, 0, 0], [0], [0], 'wrong_type', False),
                             (0, [0], [0], [0, 0, 0], [0, 0, 0], [0], [0], 1, 'wrong_type'),
                         ])
def test_environment_configuration__invalid_params(endeffector_index: int,
                                                   fixed_joints: List[int],
                                                   involved_joints: List[int],
                                                   target_position: List[float],
                                                   obstacle_position: List[float],
                                                   initial_joint_positions: List[float],
                                                   initial_positions_variation_range: List[float],
                                                   max_force: float,
                                                   visualize: bool) -> None:
    """
    Test for the EnvironmentConfiguration class if invalid parameters are loaded.
    Args:
    endeffector_index:
    fixed_joints:
    involved_joints:
    target_position:
    obstacle_position:
    initial_joint_positions:
    initial_positions_variation_range:
    max_force:
    visualize:
    """
    with pytest.raises(InvalidEnvironmentParameter):
        env_configuration = EnvironmentConfiguration(
            endeffector_index=endeffector_index,
            fixed_joints=fixed_joints,
            involved_joints=involved_joints,
            target_position=target_position,
            obstacle_position=obstacle_position,
            initial_joint_positions=initial_joint_positions,
            initial_positions_variation_range=initial_positions_variation_range,
            max_force=max_force,
            visualize=visualize
        )


@patch('robotic_manipulator_rloa.environment.environment.p.loadSDF')
@patch('robotic_manipulator_rloa.environment.environment.p.loadURDF')
@patch('robotic_manipulator_rloa.environment.environment.p.connect')
@patch('robotic_manipulator_rloa.environment.environment.p.setGravity')
@patch('robotic_manipulator_rloa.environment.environment.p.setRealTimeSimulation')
@patch('robotic_manipulator_rloa.environment.environment.p.setAdditionalSearchPath')
@patch('robotic_manipulator_rloa.environment.environment.pybullet_data')
@patch('robotic_manipulator_rloa.environment.environment.p.getNumJoints')
@patch('robotic_manipulator_rloa.environment.environment.p.getJointInfo')
@patch('robotic_manipulator_rloa.environment.environment.Environment.print_table')
@patch('robotic_manipulator_rloa.environment.environment.logger')
@patch('robotic_manipulator_rloa.environment.environment.p.resetBasePositionAndOrientation')
@patch('robotic_manipulator_rloa.environment.environment.random.uniform')
@patch('robotic_manipulator_rloa.environment.environment.p.setJointMotorControl2')
@patch('robotic_manipulator_rloa.environment.environment.p.stepSimulation')
@patch('robotic_manipulator_rloa.environment.environment.Environment.get_state')
@patch('robotic_manipulator_rloa.environment.environment.Environment.get_manipulator_obstacle_collisions')
@patch('robotic_manipulator_rloa.environment.environment.Environment.get_endeffector_target_collision')
@patch('robotic_manipulator_rloa.environment.environment.Environment.get_manipulator_collisions_with_itself')
@pytest.mark.parametrize('file, initial_pos, initial_var_range', [
    ('file.urdf', None, [0]),
    ('file.sdf', [0], None),
    ('file.sdf', [0], [1]),
    ('file.sdf', None, None)
])
def test_environment(mock_get_manipulator_collisions_with_itself: MagicMock,
                     mock_get_endeffector_target_collision: MagicMock,
                     mock_get_manipulator_obstacle_collisions: MagicMock,
                     mock_get_state: MagicMock,
                     mock_stepSimulation: MagicMock,
                     mock_setJointMotorControl2: MagicMock,
                     mock_uniform: MagicMock,
                     mock_resetBasePositionAndOrientation: MagicMock,
                     mock_logger: MagicMock,
                     mock_print_table: MagicMock,
                     mock_getJointInfo: MagicMock,
                     mock_getNumJoints: MagicMock,
                     mock_pdata: MagicMock,
                     mock_setAdditionalSearchPath: MagicMock,
                     mock_setRealTimeSimulation: MagicMock,
                     mock_setGravity: MagicMock,
                     mock_connect: MagicMock,
                     mock_loadurdf: MagicMock,
                     mock_loadsdf: MagicMock,
                     file: str,
                     initial_pos: Union[list, None],
                     initial_var_range: Union[list, None]) -> None:
    # ================== TEST FOR Environment constructor =====================

    env_configuration = EnvironmentConfiguration(
        endeffector_index=0,
        fixed_joints=[0],
        involved_joints=[0],
        target_position=[0, 0, 0],
        obstacle_position=[0, 0, 0],
        initial_joint_positions=initial_pos,
        initial_positions_variation_range=initial_var_range,
        max_force=1,
        visualize=False
    )
    mock_connect.return_value = 'physics_client'
    mock_setGravity.return_value = None
    mock_setRealTimeSimulation.return_value = None
    mock_setAdditionalSearchPath.return_value = None
    if file.endswith('.urdf'):
        mock_loadurdf.side_effect = ['urdf_env', 'obstacle', 'target']
    else:
        mock_loadurdf.side_effect = ['obstacle', 'target']
    mock_loadsdf.return_value = ['sdf_env', None]
    mock_getNumJoints.return_value = 1
    mock_getJointInfo.return_value = [None, 'name'.encode('utf-8'), None, None, None, None, None, None, 'limit_min',
                                      'limit_max', None, None, None, 'axis']
    mock_print_table.return_value = None

    env = Environment(file, env_configuration)

    assert env.manipulator_file == file
    assert env.visualize == env_configuration.visualize
    assert env.physics_client == 'physics_client'
    mock_connect.assert_any_call(p.DIRECT)
    mock_setGravity.assert_any_call(0, 0, -9.81)
    mock_setRealTimeSimulation.assert_any_call(0)
    mock_setAdditionalSearchPath.assert_any_call(mock_pdata.getDataPath.return_value)
    assert env.target_pos == [0, 0, 0]
    assert env.obstacle_pos == [0, 0, 0]
    assert env.max_force == 1
    assert env.initial_joint_positions == initial_pos
    assert env.initial_positions_variation_range == initial_var_range
    assert env.endeffector_index == 0
    assert env.fixed_joints == [0]
    assert env.involved_joints == [0]
    if file.endswith('.urdf'):
        mock_loadurdf.assert_any_call(file)
    else:
        mock_loadsdf.assert_any_call(file)
    mock_getNumJoints.assert_any_call('urdf_env' if file.endswith('urdf') else 'sdf_env')
    mock_getJointInfo.assert_any_call('urdf_env' if file.endswith('urdf') else 'sdf_env', 0)
    mock_print_table.assert_any_call([(0, 'name', 'limit_max', 'limit_min', 'axis')])
    assert env.obstacle == 'obstacle'
    assert env.target == 'target'
    mock_loadurdf.assert_any_call('sphere_small.urdf', basePosition=[0, 0, 0],
                                  useFixedBase=1, globalScaling=2.5)
    mock_loadurdf.assert_any_call('cube_small.urdf', basePosition=[0, 0, 0],
                                  useFixedBase=1, globalScaling=1)
    assert np.array_equal(env.observation_space, np.zeros((11,)))
    assert np.array_equal(env.action_space, np.zeros((1,)))

    # ================== TEST FOR reset() method ==============================

    mock_resetBasePositionAndOrientation.return_value = None
    mock_uniform.return_value = 1
    mock_setJointMotorControl2.return_value = None
    mock_stepSimulation.return_value = None
    mock_get_state.return_value = 'new_state'

    assert env.reset(True) == 'new_state'

    mock_resetBasePositionAndOrientation.assert_any_call('urdf_env' if file.endswith('urdf') else 'sdf_env',
                                                         [0.000000, 0.000000, 0.000000],
                                                         [0.000000, 0.000000, 0.000000, 1.000000])
    if not env.initial_positions_variation_range and not env.initial_joint_positions:
        mock_setJointMotorControl2.assert_any_call('urdf_env' if file.endswith('urdf') else 'sdf_env',
                                                   0, controlMode=p.POSITION_CONTROL, targetPosition=0)
    elif env.initial_joint_positions:
        if env.initial_positions_variation_range:
            mock_setJointMotorControl2.assert_any_call('urdf_env' if file.endswith('urdf') else 'sdf_env',
                                                       0, controlMode=p.POSITION_CONTROL, targetPosition=1)
        else:
            mock_setJointMotorControl2.assert_any_call('urdf_env' if file.endswith('urdf') else 'sdf_env',
                                                       0, controlMode=p.POSITION_CONTROL, targetPosition=0)
    else:
        mock_setJointMotorControl2.assert_any_call('urdf_env' if file.endswith('urdf') else 'sdf_env',
                                                   0, controlMode=p.POSITION_CONTROL, targetPosition=1)
    mock_stepSimulation.assert_any_call(env.physics_client)
    mock_get_state.assert_any_call()
    mock_logger.info.assert_any_call('Environment Reset')

    # ================== TEST FOR is_terminal_state() method ==================

    mock_get_manipulator_obstacle_collisions.return_value = True
    assert env.is_terminal_state() == 1
    mock_get_manipulator_obstacle_collisions.assert_any_call(threshold=0.)

    mock_get_manipulator_obstacle_collisions.return_value = False
    mock_get_endeffector_target_collision.return_value = [True, None]
    assert env.is_terminal_state() == 1
    mock_get_manipulator_obstacle_collisions.assert_any_call(threshold=0.)
    mock_get_endeffector_target_collision.assert_any_call(threshold=0.05)

    mock_get_manipulator_obstacle_collisions.return_value = False
    mock_get_endeffector_target_collision.return_value = [False, None]
    mock_get_manipulator_collisions_with_itself.return_value = {'joint_1': np.array([-0.5])}
    assert env.is_terminal_state(consider_autocollision=True) == 1
    mock_get_manipulator_obstacle_collisions.assert_any_call(threshold=0.)
    mock_get_endeffector_target_collision.assert_any_call(threshold=0.05)
    mock_get_manipulator_collisions_with_itself.assert_any_call()

    mock_get_manipulator_obstacle_collisions.return_value = False
    mock_get_endeffector_target_collision.return_value = [False, None]
    assert env.is_terminal_state() == 0
    mock_get_manipulator_obstacle_collisions.assert_any_call(threshold=0.)
    mock_get_endeffector_target_collision.assert_any_call(threshold=0.05)

    # ================== TEST FOR get_reward() method =========================

    mock_get_manipulator_collisions_with_itself.return_value = {'joint_1': np.array([2])}
    mock_get_endeffector_target_collision.return_value = True, 0.01
    assert env.get_reward(consider_autocollision=True) == 250
    mock_get_manipulator_collisions_with_itself.assert_any_call()
    mock_get_endeffector_target_collision.assert_any_call(threshold=0.05)

    mock_get_manipulator_collisions_with_itself.return_value = {'joint_1': np.array([-0.5])}
    mock_get_endeffector_target_collision.return_value = False, 0.5
    mock_get_manipulator_obstacle_collisions.return_value = False
    assert env.get_reward(consider_autocollision=True) == -1000
    mock_get_manipulator_collisions_with_itself.assert_any_call()
    mock_get_endeffector_target_collision.assert_any_call(threshold=0.05)
    mock_get_manipulator_obstacle_collisions.assert_any_call(threshold=0)

    mock_get_manipulator_collisions_with_itself.return_value = {'joint_1': np.array([2])}
    mock_get_endeffector_target_collision.return_value = False, 0.5
    mock_get_manipulator_obstacle_collisions.return_value = False
    assert env.get_reward(consider_autocollision=True) == -0.5
    mock_get_manipulator_collisions_with_itself.assert_any_call()
    mock_get_endeffector_target_collision.assert_any_call(threshold=0.05)
    mock_get_manipulator_obstacle_collisions.assert_any_call(threshold=0)


@patch('robotic_manipulator_rloa.environment.environment.p.loadURDF')
@patch('robotic_manipulator_rloa.environment.environment.p.connect')
@patch('robotic_manipulator_rloa.environment.environment.p.setGravity')
@patch('robotic_manipulator_rloa.environment.environment.p.setRealTimeSimulation')
@patch('robotic_manipulator_rloa.environment.environment.p.setAdditionalSearchPath')
@patch('robotic_manipulator_rloa.environment.environment.pybullet_data')
@pytest.mark.parametrize('file', ['file.urdf', 'invalid_file', None])
def test_environment__invalid_manipulator_file(mock_pdata: MagicMock,
                                               mock_setAdditionalSearchPath: MagicMock,
                                               mock_setRealTimeSimulation: MagicMock,
                                               mock_setGravity: MagicMock,
                                               mock_connect: MagicMock,
                                               mock_loadurdf: MagicMock,
                                               file: str) -> None:
    """Test for the Environment constructor when InvalidManipulatorFile exception is raised"""
    env_configuration = EnvironmentConfiguration(
        endeffector_index=0,
        fixed_joints=[0],
        involved_joints=[0],
        target_position=[0, 0, 0],
        obstacle_position=[0, 0, 0],
        initial_joint_positions=[0],
        initial_positions_variation_range=[0],
        max_force=1,
        visualize=False
    )
    mock_connect.return_value = 'physics_client'
    mock_setGravity.return_value = None
    mock_setRealTimeSimulation.return_value = None
    mock_setAdditionalSearchPath.return_value = None
    mock_loadurdf.side_effect = p.error

    with pytest.raises(InvalidManipulatorFile):
        env = Environment(file, env_configuration)


@patch('robotic_manipulator_rloa.environment.environment.p.loadURDF')
@patch('robotic_manipulator_rloa.environment.environment.p.connect')
@patch('robotic_manipulator_rloa.environment.environment.p.setGravity')
@patch('robotic_manipulator_rloa.environment.environment.p.setRealTimeSimulation')
@patch('robotic_manipulator_rloa.environment.environment.p.setAdditionalSearchPath')
@patch('robotic_manipulator_rloa.environment.environment.pybullet_data')
@patch('robotic_manipulator_rloa.environment.environment.p.getNumJoints')
@patch('robotic_manipulator_rloa.environment.environment.p.getJointInfo')
@patch('robotic_manipulator_rloa.environment.environment.Environment.print_table')
@patch('robotic_manipulator_rloa.environment.environment.CollisionObject')
@patch('robotic_manipulator_rloa.environment.environment.CollisionDetector')
def test_environment__get_manipulator_obstacle_collisions(mock_collision_detector: MagicMock,
                                                          mock_collision_object: MagicMock,
                                                          mock_print_table: MagicMock,
                                                          mock_getJointInfo: MagicMock,
                                                          mock_getNumJoints: MagicMock,
                                                          mock_pdata: MagicMock,
                                                          mock_setAdditionalSearchPath: MagicMock,
                                                          mock_setRealTimeSimulation: MagicMock,
                                                          mock_setGravity: MagicMock,
                                                          mock_connect: MagicMock,
                                                          mock_loadurdf: MagicMock) -> None:
    """Test for the get_manipulator_obstacle_collisions() method of the Environment class"""
    # Initialize Fake Environment
    env_configuration = EnvironmentConfiguration(
        endeffector_index=0,
        fixed_joints=[0],
        involved_joints=[0],
        target_position=[0, 0, 0],
        obstacle_position=[0, 0, 0],
        initial_joint_positions=[0],
        initial_positions_variation_range=[0],
        max_force=1,
        visualize=False
    )
    mock_connect.return_value = 'physics_server'
    mock_loadurdf.side_effect = ['urdf_env', 'obstacle', 'target']
    mock_getNumJoints.return_value = 0
    env = Environment('file.urdf', env_configuration)

    # Test for get_manipulator_obstacle_collisions() method
    env.num_joints = 1
    fake_collision_object, fake_collision_detector = MagicMock(), MagicMock()
    mock_collision_object.return_value = fake_collision_object
    mock_collision_detector.return_value = fake_collision_detector
    fake_collision_detector.compute_distances.return_value = (1, None)

    assert not env.get_manipulator_obstacle_collisions(threshold=0.5)
    assert env.get_manipulator_obstacle_collisions(threshold=1.5)

    mock_collision_object.assert_any_call(body='urdf_env', link=0)
    mock_collision_detector.assert_any_call(collision_object=fake_collision_object,
                                            obstacle_ids=['obstacle'])
    fake_collision_detector.compute_distances.assert_any_call()


@patch('robotic_manipulator_rloa.environment.environment.p.loadURDF')
@patch('robotic_manipulator_rloa.environment.environment.p.connect')
@patch('robotic_manipulator_rloa.environment.environment.p.setGravity')
@patch('robotic_manipulator_rloa.environment.environment.p.setRealTimeSimulation')
@patch('robotic_manipulator_rloa.environment.environment.p.setAdditionalSearchPath')
@patch('robotic_manipulator_rloa.environment.environment.pybullet_data')
@patch('robotic_manipulator_rloa.environment.environment.p.getNumJoints')
@patch('robotic_manipulator_rloa.environment.environment.p.getJointInfo')
@patch('robotic_manipulator_rloa.environment.environment.Environment.print_table')
@patch('robotic_manipulator_rloa.environment.environment.CollisionObject')
@patch('robotic_manipulator_rloa.environment.environment.CollisionDetector')
def test_environment__get_manipulator_collisions_with_itself(mock_collision_detector: MagicMock,
                                                             mock_collision_object: MagicMock,
                                                             mock_print_table: MagicMock,
                                                             mock_getJointInfo: MagicMock,
                                                             mock_getNumJoints: MagicMock,
                                                             mock_pdata: MagicMock,
                                                             mock_setAdditionalSearchPath: MagicMock,
                                                             mock_setRealTimeSimulation: MagicMock,
                                                             mock_setGravity: MagicMock,
                                                             mock_connect: MagicMock,
                                                             mock_loadurdf: MagicMock) -> None:
    """Test for the get_manipulator_collisions_with_itself() method of the Environment class"""
    # Initialize Fake Environment
    env_configuration = EnvironmentConfiguration(
        endeffector_index=0,
        fixed_joints=[0],
        involved_joints=[0],
        target_position=[0, 0, 0],
        obstacle_position=[0, 0, 0],
        initial_joint_positions=[0],
        initial_positions_variation_range=[0],
        max_force=1,
        visualize=False
    )
    mock_connect.return_value = 'physics_server'
    mock_loadurdf.side_effect = ['urdf_env', 'obstacle', 'target']
    mock_getNumJoints.return_value = 0
    env = Environment('file.urdf', env_configuration)

    # Test for get_manipulator_collisions_with_itself() method
    env.num_joints = 1
    fake_collision_object, fake_collision_detector = MagicMock(), MagicMock()
    mock_collision_object.return_value = fake_collision_object
    mock_collision_detector.return_value = fake_collision_detector
    fake_collision_detector.compute_collisions_in_manipulator.return_value = 1.5

    assert env.get_manipulator_collisions_with_itself() == {'joint_0': 1.5}

    mock_collision_object.assert_any_call(body='urdf_env', link=0)
    mock_collision_detector.assert_any_call(collision_object=fake_collision_object,
                                            obstacle_ids=[])
    fake_collision_detector.compute_collisions_in_manipulator.assert_any_call(
        affected_joints=[0], max_distance=10
    )


@patch('robotic_manipulator_rloa.environment.environment.p.loadURDF')
@patch('robotic_manipulator_rloa.environment.environment.p.connect')
@patch('robotic_manipulator_rloa.environment.environment.p.setGravity')
@patch('robotic_manipulator_rloa.environment.environment.p.setRealTimeSimulation')
@patch('robotic_manipulator_rloa.environment.environment.p.setAdditionalSearchPath')
@patch('robotic_manipulator_rloa.environment.environment.pybullet_data')
@patch('robotic_manipulator_rloa.environment.environment.p.getNumJoints')
@patch('robotic_manipulator_rloa.environment.environment.p.getJointInfo')
@patch('robotic_manipulator_rloa.environment.environment.Environment.print_table')
@patch('robotic_manipulator_rloa.environment.environment.CollisionObject')
@patch('robotic_manipulator_rloa.environment.environment.CollisionDetector')
def test_environment__get_endeffector_target_collision(mock_collision_detector: MagicMock,
                                                       mock_collision_object: MagicMock,
                                                       mock_print_table: MagicMock,
                                                       mock_getJointInfo: MagicMock,
                                                       mock_getNumJoints: MagicMock,
                                                       mock_pdata: MagicMock,
                                                       mock_setAdditionalSearchPath: MagicMock,
                                                       mock_setRealTimeSimulation: MagicMock,
                                                       mock_setGravity: MagicMock,
                                                       mock_connect: MagicMock,
                                                       mock_loadurdf: MagicMock) -> None:
    """Test for the get_endeffector_target_collision() method of the Environment class"""
    # Initialize Fake Environment
    env_configuration = EnvironmentConfiguration(
        endeffector_index=2,
        fixed_joints=[0],
        involved_joints=[0],
        target_position=[0, 0, 0],
        obstacle_position=[0, 0, 0],
        initial_joint_positions=[0],
        initial_positions_variation_range=[0],
        max_force=1,
        visualize=False
    )
    mock_connect.return_value = 'physics_server'
    mock_loadurdf.side_effect = ['urdf_env', 'obstacle', 'target']
    mock_getNumJoints.return_value = 0
    env = Environment('file.urdf', env_configuration)

    # Test for get_endeffector_target_collision() method
    env.num_joints = 1
    fake_collision_object, fake_collision_detector = MagicMock(), MagicMock()
    mock_collision_object.return_value = fake_collision_object
    mock_collision_detector.return_value = fake_collision_detector
    fake_collision_detector.compute_distances.return_value = np.array([.75])

    assert env.get_endeffector_target_collision(threshold=0.5) == (False, 0.25)
    assert env.get_endeffector_target_collision(threshold=1) == (True, -0.25)

    mock_collision_object.assert_any_call(body='urdf_env', link=2)
    mock_collision_detector.assert_any_call(collision_object=fake_collision_object,
                                            obstacle_ids=['target'])
    fake_collision_detector.compute_distances.assert_any_call()


@patch('robotic_manipulator_rloa.environment.environment.p.loadURDF')
@patch('robotic_manipulator_rloa.environment.environment.p.connect')
@patch('robotic_manipulator_rloa.environment.environment.p.setGravity')
@patch('robotic_manipulator_rloa.environment.environment.p.setRealTimeSimulation')
@patch('robotic_manipulator_rloa.environment.environment.p.setAdditionalSearchPath')
@patch('robotic_manipulator_rloa.environment.environment.pybullet_data')
@patch('robotic_manipulator_rloa.environment.environment.p.getNumJoints')
@patch('robotic_manipulator_rloa.environment.environment.p.getJointInfo')
@patch('robotic_manipulator_rloa.environment.environment.Environment.print_table')
@patch('robotic_manipulator_rloa.environment.environment.p.getJointState')
@patch('robotic_manipulator_rloa.environment.environment.p.getLinkState')
def test_environment__get_state(mock_getLinkState: MagicMock,
                                mock_getJointState: MagicMock,
                                mock_print_table: MagicMock,
                                mock_getJointInfo: MagicMock,
                                mock_getNumJoints: MagicMock,
                                mock_pdata: MagicMock,
                                mock_setAdditionalSearchPath: MagicMock,
                                mock_setRealTimeSimulation: MagicMock,
                                mock_setGravity: MagicMock,
                                mock_connect: MagicMock,
                                mock_loadurdf: MagicMock) -> None:
    """Test for the get_state() method of the Environment class"""
    # Initialize Fake Environment
    env_configuration = EnvironmentConfiguration(
        endeffector_index=2,
        fixed_joints=[0],
        involved_joints=[0],
        target_position=[0, 0, 0],
        obstacle_position=[0, 0, 0],
        initial_joint_positions=[0],
        initial_positions_variation_range=[0],
        max_force=1,
        visualize=False
    )
    mock_connect.return_value = 'physics_server'
    mock_loadurdf.side_effect = ['urdf_env', 'obstacle', 'target']
    mock_getNumJoints.return_value = 0
    env = Environment('file.urdf', env_configuration)

    # Test for get_state() method
    mock_getJointState.return_value = [0, 3]
    mock_getLinkState.return_value = [(6, 7, 8), None]

    assert np.array_equal(env.get_state(), np.array([0, 3, 6, 7, 8, 0, 0, 0, 0, 0, 0]))

    mock_getJointState.assert_any_call('urdf_env', 0)
    mock_getLinkState.assert_any_call('urdf_env', 2)


@patch('robotic_manipulator_rloa.environment.environment.p.loadURDF')
@patch('robotic_manipulator_rloa.environment.environment.p.connect')
@patch('robotic_manipulator_rloa.environment.environment.p.setGravity')
@patch('robotic_manipulator_rloa.environment.environment.p.setRealTimeSimulation')
@patch('robotic_manipulator_rloa.environment.environment.p.setAdditionalSearchPath')
@patch('robotic_manipulator_rloa.environment.environment.pybullet_data')
@patch('robotic_manipulator_rloa.environment.environment.p.getNumJoints')
@patch('robotic_manipulator_rloa.environment.environment.p.getJointInfo')
@patch('robotic_manipulator_rloa.environment.environment.Environment.print_table')
@patch('robotic_manipulator_rloa.environment.environment.p.setJointMotorControl2')
@patch('robotic_manipulator_rloa.environment.environment.p.stepSimulation')
@patch('robotic_manipulator_rloa.environment.environment.Environment.get_reward')
@patch('robotic_manipulator_rloa.environment.environment.Environment.get_state')
@patch('robotic_manipulator_rloa.environment.environment.Environment.is_terminal_state')
def test_environment__step(mock_is_terminal_state: MagicMock,
                           mock_get_state: MagicMock,
                           mock_get_reward: MagicMock,
                           mock_stepSimulation: MagicMock,
                           mock_setJointMotorControl2: MagicMock,
                           mock_print_table: MagicMock,
                           mock_getJointInfo: MagicMock,
                           mock_getNumJoints: MagicMock,
                           mock_pdata: MagicMock,
                           mock_setAdditionalSearchPath: MagicMock,
                           mock_setRealTimeSimulation: MagicMock,
                           mock_setGravity: MagicMock,
                           mock_connect: MagicMock,
                           mock_loadurdf: MagicMock) -> None:
    """Test for the step() method of the Environment class"""
    # Initialize Fake Environment
    env_configuration = EnvironmentConfiguration(
        endeffector_index=2,
        fixed_joints=[1],
        involved_joints=[0],
        target_position=[0, 0, 0],
        obstacle_position=[0, 0, 0],
        initial_joint_positions=[0],
        initial_positions_variation_range=[0],
        max_force=1,
        visualize=False
    )
    mock_connect.return_value = 'physics_client'
    mock_loadurdf.side_effect = ['urdf_env', 'obstacle', 'target']
    mock_getNumJoints.return_value = 0
    env = Environment('file.urdf', env_configuration)

    # Test for step() method
    mock_setJointMotorControl2.return_value = None
    mock_stepSimulation.return_value = None
    mock_get_reward.return_value = 'reward'
    mock_get_state.return_value = 'state'
    mock_is_terminal_state.return_value = 'done'

    assert env.step(np.array([7])) == ('state', 'reward', 'done')

    mock_setJointMotorControl2.assert_any_call('urdf_env', 0, p.VELOCITY_CONTROL, targetVelocity=7, force=env.max_force)
    mock_setJointMotorControl2.assert_any_call('urdf_env', 1, p.POSITION_CONTROL, targetPosition=0)
    mock_stepSimulation.assert_any_call(physicsClientId='physics_client')
    mock_get_reward.assert_any_call()
    mock_get_state.assert_any_call()
    mock_is_terminal_state.assert_any_call()


@patch('robotic_manipulator_rloa.environment.environment.logger')
def test_print_table(mock_logger: MagicMock) -> None:
    """Test for the print_table() method of the Environment class"""
    data = [(0, 'joint_name', 1.5, -1.5, (0, 0, 1))]
    Environment.print_table(data)
    mock_logger.debug.assert_any_call('{:<6} {:<35} {:<15} {:<15} {:<15}'.format(
        'Index', 'Name', 'Upper Limit', 'Lower Limit', 'Axis'))
    mock_logger.debug.assert_any_call('{:<6} {:<35} {:<15} {:<15} {:<15}'.format(
        0, 'joint_name', 1.5, -1.5, str((0, 0, 1))))
