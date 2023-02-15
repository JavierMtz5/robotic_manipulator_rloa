import random
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from numpy.typing import NDArray

from robotic_manipulator_rloa.utils.logger import get_global_logger
from robotic_manipulator_rloa.utils.exceptions import (
    InvalidManipulatorFile,
    InvalidEnvironmentParameter
)
from robotic_manipulator_rloa.utils.collision_detector import CollisionObject, CollisionDetector

logger = get_global_logger()


class EnvironmentConfiguration:

    def __init__(self,
                 endeffector_index: int,
                 fixed_joints: List[int],
                 involved_joints: List[int],
                 target_position: List[float],
                 obstacle_position: List[float],
                 initial_joint_positions: List[float] = None,
                 initial_positions_variation_range: List[float] = None,
                 max_force: float = 200.,
                 visualize: bool = True):
        """
        Validates each of the parameters required for the Environment class initialization.
        Args:
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
        """
        self._validate_endeffector_index(endeffector_index)
        self._validate_fixed_joints(fixed_joints)
        self._validate_involved_joints(involved_joints)
        self._validate_target_position(target_position)
        self._validate_obstacle_position(obstacle_position)
        self._validate_initial_joint_positions(initial_joint_positions)
        self._validate_initial_positions_variation_range(initial_positions_variation_range)
        self._validate_max_force(max_force)
        self._validate_visualize(visualize)

    def _validate_endeffector_index(self, endeffector_index: int) -> None:
        """
        Validates the "endeffector_index" parameter.
        Args:
            endeffector_index: int
        Raises:
            InvalidEnvironmentParameter
        """
        if not isinstance(endeffector_index, int):
            raise InvalidEnvironmentParameter('End Effector index received is not an integer')
        self.endeffector_index = endeffector_index

    def _validate_fixed_joints(self, fixed_joints: List[int]) -> None:
        """
        Validates the "fixed_joints" parameter
        Args:
            fixed_joints: list of integers
        Raises:
            InvalidEnvironmentParameter
        """
        if not isinstance(fixed_joints, list):
            raise InvalidEnvironmentParameter('Fixed Joints received is not a list')
        for val in fixed_joints:
            if not isinstance(val, int):
                raise InvalidEnvironmentParameter('An item inside the Fixed Joints list is not an integer')
        self.fixed_joints = fixed_joints

    def _validate_involved_joints(self, involved_joints: List[int]) -> None:
        """
        Validates the "involved_joints" parameter
        Args:
            involved_joints: list of integers
        Raises:
            InvalidEnvironmentParameter
        """
        if not isinstance(involved_joints, list):
            raise InvalidEnvironmentParameter('Involved Joints received is not a list')
        for val in involved_joints:
            if not isinstance(val, int):
                raise InvalidEnvironmentParameter('An item inside the Involved Joints list is not an integer')
        self.involved_joints = involved_joints

    def _validate_target_position(self, target_position: List[float]) -> None:
        """
        Validates the "target_position" parameter
        Args:
            target_position: list of floats
        Raises:
            InvalidEnvironmentParameter
        """
        if not isinstance(target_position, list):
            raise InvalidEnvironmentParameter('Target Position received is not a list')
        for val in target_position:
            if not isinstance(val, (int, float)):
                raise InvalidEnvironmentParameter('An item inside the Target Position list is not a float')
        self.target_position = target_position

    def _validate_obstacle_position(self, obstacle_position: List[float]) -> None:
        """
        Validates the "obstacle_position" parameter
        Args:
            obstacle_position: list of floats
        Raises:
            InvalidEnvironmentParameter
        """
        if not isinstance(obstacle_position, list):
            raise InvalidEnvironmentParameter('Obstacle Position received is not a list')
        for val in obstacle_position:
            if not isinstance(val, (int, float)):
                raise InvalidEnvironmentParameter('An item inside the Obstacle Position list is not a float')
        self.obstacle_position = obstacle_position

    def _validate_initial_joint_positions(self, initial_joint_positions: List[float]) -> None:
        """
        Validates the "initial_joint_positions" parameter
        Args:
            initial_joint_positions: list of floats
        Raises:
            InvalidEnvironmentParameter
        """
        if initial_joint_positions is None:
            self.initial_joint_positions = None
            return
        if not isinstance(initial_joint_positions, list):
            raise InvalidEnvironmentParameter('Initial Joint Positions received is not a list')
        for val in initial_joint_positions:
            if not isinstance(val, (int, float)):
                raise InvalidEnvironmentParameter('An item inside the Initial Joint Positions list is not a float')
        self.initial_joint_positions = initial_joint_positions

    def _validate_initial_positions_variation_range(self, initial_positions_variation_range: List[float]) -> None:
        """
        Validates the "initial_positions_variation_range" parameter
        Args:
            initial_positions_variation_range: list of floats
        Raises:
            InvalidEnvironmentParameter
        """
        if initial_positions_variation_range is None:
            self.initial_positions_variation_range = None
            return
        if not isinstance(initial_positions_variation_range, list):
            raise InvalidEnvironmentParameter('Initial Positions Variation Range received is not a list')
        for val in initial_positions_variation_range:
            if not isinstance(val, (float, int)):
                raise InvalidEnvironmentParameter('An item inside the Initial Positions Variation Range '
                                                  'list is not a float')
        self.initial_positions_variation_range = initial_positions_variation_range

    def _validate_max_force(self, max_force: float) -> None:
        """
        Validates the "max_force" parameter
        Args:
            max_force: float
        Raises:
            InvalidEnvironmentParameter
        """
        if not isinstance(max_force, (int, float)):
            raise InvalidEnvironmentParameter('Maximum Force value received is not a float')
        self.max_force = max_force

    def _validate_visualize(self, visualize: bool) -> None:
        """
        Validates the "visualize" parameter
        Args:
            visualize: bool
        Raises:
            InvalidEnvironmentParameter
        """
        if not isinstance(visualize, bool):
            raise InvalidEnvironmentParameter('Visualize value received is not a boolean')
        self.visualize = visualize


class Environment:

    def __init__(self,
                 manipulator_file: str,
                 environment_config: EnvironmentConfiguration):
        """
        Creates the Pybullet environment used along the training.
        Args:
            manipulator_file: Path to the URDF or SDF file from which to load the Robotic Manipulator.
            environment_config: Instance of the EnvironmentConfiguration class with all its attributes set.
        Raises:
            InvalidManipulatorFile: The URDF/SDF file doesn't exist, is invalid or has an invalid extension.
        """
        self.manipulator_file = manipulator_file
        self.visualize = environment_config.visualize

        # Initialize pybullet
        self.physics_client = p.connect(p.GUI if environment_config.visualize else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.target_pos = environment_config.target_position
        self.obstacle_pos = environment_config.obstacle_position
        self.max_force = environment_config.max_force  # Maximum force to be applied (DEFAULT=200)
        self.initial_joint_positions = environment_config.initial_joint_positions
        self.initial_positions_variation_range = environment_config.initial_positions_variation_range

        self.endeffector_index = environment_config.endeffector_index  # Index of the Manipulator's End-Effector
        self.fixed_joints = environment_config.fixed_joints  # List of indexes for the joints to be fixed
        self.involved_joints = environment_config.involved_joints  # List of indexes of joints involved in the training

        # Load Manipulator from URDF/SDF file
        logger.debug(f'Loading URDF/SDF file {manipulator_file} for Robot Manipulator...')
        if not isinstance(manipulator_file, str):
            raise InvalidManipulatorFile('The filename provided is not a string')

        try:
            if manipulator_file.endswith('.urdf'):
                self.manipulator_uid = p.loadURDF(manipulator_file)
            elif manipulator_file.endswith('.sdf'):
                self.manipulator_uid = p.loadSDF(manipulator_file)[0]
            else:
                raise InvalidManipulatorFile('The file extension is neither .sdf nor .urdf')
        except p.error as err:
            logger.critical(err)
            raise InvalidManipulatorFile

        self.num_joints = p.getNumJoints(self.manipulator_uid)

        logger.debug(f'Robot Manipulator URDF/SDF file {manipulator_file} has been successfully loaded. '
                     f'The Robot Manipulator has {self.num_joints} joints, and its joints, '
                     f'together with the information of each, are:')
        data = list()
        for joint_ind in range(self.num_joints):
            joint_info = p.getJointInfo(self.manipulator_uid, joint_ind)
            data.append((joint_ind, joint_info[1].decode("utf-8"), joint_info[9], joint_info[8], joint_info[13]))

        # Print Joints info
        self.print_table(data)

        # Create obstacle with the shape of a sphere, and the target object with square shape
        self.obstacle = p.loadURDF('sphere_small.urdf', basePosition=self.obstacle_pos,
                                   useFixedBase=1, globalScaling=2.5)
        self.target = p.loadURDF('cube_small.urdf', basePosition=self.target_pos,
                                 useFixedBase=1, globalScaling=1)
        logger.debug(f'Both the obstacle and the target object have been generated in positions {self.obstacle_pos} '
                     f'and {self.target_pos} respectively')

        # 9 elements correspond to the 3 vector indicating the position of the target, end effector and obstacle
        # The other elements are the two arrays of the involved joint's position and velocities
        self._observation_space = np.zeros((9 + 2 * len(self.involved_joints),))
        self._action_space = np.zeros((len(self.involved_joints),))

    def reset(self, verbose: bool = True) -> NDArray:
        """
        Resets the environment to a initial state.\n
        - If "initial_joint_positions" and "initial_positions_variation_range" are not set, all joints will be reset to
        the 0 position.\n
        - If only "initial_joint_positions" is set, the joints will be reset to those positions.\n
        - If only "initial_positions_variation_range" is set, the joints will be reset to 0 plus the variation noise.\n
        - If both "initial_joint_positions" and "initial_positions_variation_range" are set, the joints will be reset
        to the positions specified plus the variation noise.
        Args:
            verbose: Boolean indicating whether to print context information or not.
        Returns:
            New state reached after reset.
        """
        if verbose: logger.info('Resetting Environment...')

        # Reset the robot's base position and orientation
        p.resetBasePositionAndOrientation(self.manipulator_uid, [0.000000, 0.000000, 0.000000],
                                          [0.000000, 0.000000, 0.000000, 1.000000])

        if not self.initial_joint_positions and not self.initial_positions_variation_range:
            initial_state = [0 for _ in range(self.num_joints)]
        elif self.initial_joint_positions:
            if self.initial_positions_variation_range:
                initial_state = [random.uniform(pos - var, pos + var) for pos, var
                                 in zip(self.initial_joint_positions, self.initial_positions_variation_range)]
            else:
                initial_state = self.initial_joint_positions
        else:
            initial_state = [random.uniform(0 - var, 0 + var) for var in self.initial_positions_variation_range]

        for joint_index, pos in enumerate(initial_state):
            p.setJointMotorControl2(self.manipulator_uid, joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=pos)

        for _ in range(50):
            p.stepSimulation(self.physics_client)

        # Generate first state, and return it
        # The states are defined as {joint_pos, joint_vel, end-effector_pos, target_pos, obstacle_pos}, where
        # both joint_pos and joint_vel are arrays with the pos and vel of each joint
        new_state = self.get_state()
        if verbose: logger.info('Environment Reset')

        return new_state

    def is_terminal_state(self, target_threshold: float = 0.05, obstacle_threshold: float = 0.,
                          consider_autocollision: bool = False) -> int:
        """
        Calculates if a terminal state is reached.
        Args:
            target_threshold: Threshold which delimits the terminal state. If the end-effector is closer
                to the target position than the threshold value, then a terminal state is reached.
            obstacle_threshold: Threshold which delimits the terminal state. If the end-effector is closer
                to the obstacle position than the threshold value, then a terminal state is reached.
            consider_autocollision: If set to True, the collision of any of the joints and parts of the manipulator
                with any other joint or part will be considered a terminal state.
        Returns:
            Integer (0 or 1) indicating whether the new state reached is a terminal state or not.
        """
        # If the manipulator has a collision with the obstacle, the episode terminates
        if self.get_manipulator_obstacle_collisions(threshold=obstacle_threshold):
            logger.info('Collision detected, terminating episode...')
            return 1

        # If the position of the end-effector is the same as the one of the target position, episode terminates
        if self.get_endeffector_target_collision(threshold=target_threshold)[0]:
            logger.info('The goal state has been reached, terminating episode...')
            return 1

        # If the manipulator collides with itself, a terminal state is reached
        if consider_autocollision:
            self_distances = self.get_manipulator_collisions_with_itself()
            for distances in self_distances.values():
                if (distances < 0).any():
                    logger.info('Auto-Collision detected, terminating episode...')
                    return 1

        return 0

    def get_reward(self, consider_autocollision: bool = False) -> float:
        """
        Computes the reward from the given state.
        Returns:
            Rewards:\n
            - If the end effector reaches the target position, a reward of +250 is returned.\n
            - If the end effector collides with the obstacle or with itself*, a reward of -1000 is returned.\n
            - Otherwise, the negative value of the distance from end effector to the target is returned.\n
            * The manipulator's collisions with itself are only considered if "consider_autocollision" parameter is set
            to True.
        """
        # Auto-Collision is only calculated if requested
        self_collision = False
        if consider_autocollision:
            self_distances = self.get_manipulator_collisions_with_itself()
            for distances in self_distances.values():
                if (distances < 0).any():
                    self_collision = True

        endeffector_target_collision, endeffector_target_dist = self.get_endeffector_target_collision(threshold=0.05)

        if endeffector_target_collision:
            return 250
        elif self.get_manipulator_obstacle_collisions(threshold=0) or self_collision:
            return -1000
        else:
            return -1 * float(endeffector_target_dist)

    def get_manipulator_obstacle_collisions(self, threshold: float) -> bool:
        """
        Calculates if there is a collision between the manipulator and the obstacle.
        Args:
            threshold: If the distance between the end effector and the obstacle is below the "threshold", then
                it is considered a collision.
        Returns:
            Boolean indicating whether a collision occurred.
        """
        joint_distances = list()
        for joint_ind in range(self.num_joints):
            end_effector_collision_obj = CollisionObject(body=self.manipulator_uid, link=joint_ind)
            collision_detector = CollisionDetector(collision_object=end_effector_collision_obj,
                                                   obstacle_ids=[self.obstacle])

            dist = collision_detector.compute_distances()
            joint_distances.append(dist[0])

        joint_distances = np.array(joint_distances)
        return (joint_distances < threshold).any()

    def get_manipulator_collisions_with_itself(self) -> dict:
        """
        Calculates the distances between each of the manipulator's joints and the other joints.
        Returns:
            Dictionary where each key is the index of a joint, and where each value is an array with the
            distances from that joint to any other joint in the manipulator.
        """
        joint_distances = dict()
        for joint_ind in range(self.num_joints):
            joint_collision_obj = CollisionObject(body=self.manipulator_uid, link=joint_ind)
            collision_detector = CollisionDetector(collision_object=joint_collision_obj,
                                                   obstacle_ids=[])
            distances = collision_detector.compute_collisions_in_manipulator(
                affected_joints=[_ for _ in range(self.num_joints)],  # all joints are taken into account
                max_distance=10
            )
            joint_distances[f'joint_{joint_ind}'] = distances

        return joint_distances

    def get_endeffector_target_collision(self, threshold: float) -> Tuple[bool, float]:
        """
        Calculates if there are any collisions between the end effector and the target.
        Args:
            threshold: If the distance between the end effector and the target is below {threshold}, then
                it is considered a collision.
        Returns:
            Tuple where the first element is a boolean indicating whether a collision occurred, adn where
            the second is the distance from end effector to target minus the threshold.
        """
        kuka_end_effector = CollisionObject(body=self.manipulator_uid, link=self.endeffector_index)
        collision_detector = CollisionDetector(collision_object=kuka_end_effector, obstacle_ids=[self.target])

        dist = collision_detector.compute_distances()

        return (dist < threshold).any(), dist - threshold

    def get_state(self) -> NDArray:
        """
        Retrieves information from the environment's current state.
        Returns:
            State as (joint_pos, joint_vel, end-effector_pos, target_pos, obstacle_pos):\n
            - The positions of the target, obstacle and end effector are given as 3D cartesian coordinates.\n
            - The joint positions and joint velocities are given as arrays of length equal to the number of
            joint involved in the training.
        """
        joint_pos, joint_vel = list(), list()

        for joint_index in range(len(self.involved_joints)):
            joint_pos.append(p.getJointState(self.manipulator_uid, joint_index)[0])
            joint_vel.append(p.getJointState(self.manipulator_uid, joint_index)[1])

        end_effector_pos = p.getLinkState(self.manipulator_uid, self.endeffector_index)[0]
        end_effector_pos = list(end_effector_pos)

        state = np.hstack([np.array(joint_pos), np.array(joint_vel), np.array(end_effector_pos),
                           np.array(self.target_pos), np.array(self.obstacle_pos)])
        return state.astype(float)

    def step(self, action: NDArray) -> Tuple[NDArray, float, int]:
        """
        Applies the action on the Robot's joints, so that each joint reaches the desired velocity for
        each involved joint.
        Args:
            action: Array where each element corresponds to the velocity to be applied on the joint
                with that same index.
        Returns:
            (new_state, reward, done)
        """
        # Apply velocities on the involved joints according to action
        for joint_index, vel in zip(self.involved_joints, action):
            p.setJointMotorControl2(self.manipulator_uid,
                                    joint_index,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=vel,
                                    force=self.max_force)

        # Create constraint for fixed joints (maintain joint on fixed position)
        for joint_ind in self.fixed_joints:
            p.setJointMotorControl2(self.manipulator_uid,
                                    joint_ind,
                                    p.POSITION_CONTROL,
                                    targetPosition=0)

        # Perform actions on simulation
        p.stepSimulation(physicsClientId=self.physics_client)

        reward = self.get_reward()
        new_state = self.get_state()
        done = self.is_terminal_state()

        return new_state, reward, done

    @staticmethod
    def print_table(data: List[Tuple[int, str, float, float, tuple]]) -> None:
        """
        Prints a table such that the elements received in the "data" parameter are displayed under
        "Index", "Name", "Upper Limit", "Lower Limit" and "Axis" columns. It is used to print the Manipulator's
        joint's information in an ordered manner.
        Args:
            data: List where each element contains all the information about a given joint.
                Each element on the list will be a tuple containing (index, name, upper_limit, lower_limit, axis).
        """
        logger.debug('{:<6} {:<35} {:<15} {:<15} {:<15}'.format('Index', 'Name', 'Upper Limit', 'Lower Limit', 'Axis'))
        for index, name, up_limit, lo_limit, axis in data:
            logger.debug('{:<6} {:<35} {:<15} {:<15} {:<15}'.format(index, name, up_limit, lo_limit, str(axis)))

    @property
    def observation_space(self) -> np.ndarray:
        """
        Getter for the observation space of the environment.
        Returns:
            Numpy array of zeros with same shape as the environment's states.
        """
        return self._observation_space

    @property
    def action_space(self) -> np.ndarray:
        """
        Getter for the action space of the environment.
        Returns:
            Numpy array of zeros with same shape as the environment's actions.
        """
        return self._action_space
