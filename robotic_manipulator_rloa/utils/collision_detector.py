from dataclasses import dataclass
from typing import List

import numpy as np
import pybullet as p
from numpy.typing import NDArray


@dataclass
class CollisionObject:
    """
    Dataclass which contains the UID of the manipulator/body and the link number of the joint from which to
    calculate distances to other bodies.
    """
    body: str
    link: int


class CollisionDetector:

    def __init__(self, collision_object: CollisionObject, obstacle_ids: List[str]):
        """
        Calculates distances between bodies' joints.
        Args:
            collision_object: CollisionObject instance, which indicates the body/joint from
                which to calculate distances/collisions.
            obstacle_ids: Obstacle body UID. Distances are calculated from the joint/body given in the
                "collision_object" parameter to the "obstacle_ids" bodies.
        """
        self.obstacles = obstacle_ids
        self.collision_object = collision_object

    def compute_distances(self, max_distance: float = 10.0) -> NDArray:
        """
        Compute the closest distances from the joint given by the CollisionObject instance in self.collision_object
        to the bodies defined in self.obstacles.
        Args:
            max_distance: Bodies farther apart than this distance are not queried by PyBullet, the return value
                for the distance between such bodies will be max_distance.
        Returns:
            A numpy array of distances, one per pair of collision objects.
        """
        distances = list()
        for obstacle in self.obstacles:

            # Compute the shortest distances between the collision-object and the given obstacle
            closest_points = p.getClosestPoints(
                self.collision_object.body,
                obstacle,
                distance=max_distance,
                linkIndexA=self.collision_object.link
            )

            # If bodies are above max_distance apart, nothing is returned, so
            # we just saturate at max_distance. Otherwise, take the minimum
            if len(closest_points) == 0:
                distances.append(max_distance)
            else:
                distances.append(np.min([point[8] for point in closest_points]))

        return np.array(distances)

    def compute_collisions_in_manipulator(self, affected_joints: List[int], max_distance: float = 10.) -> NDArray:
        """
        Compute collisions between manipulator's parts.
        Args:
            affected_joints: Joints to consider when calculating distances.
            max_distance: Maximum distance to be considered. Distances further than this will be ignored, and
                the "max_distance" value will be returned.
        Returns:
            Array where each element corresponds to the distances from a given joint to the other joints.
        """
        distances = list()
        for joint_ind in affected_joints:

            # Collisions with the previous and next joints are omitted, as they will be always in contact
            if (self.collision_object.link == joint_ind) or \
                    (joint_ind == self.collision_object.link - 1) or \
                    (joint_ind == self.collision_object.link + 1):
                continue    # pragma: no cover

            # Compute the shortest distances between all object pairs
            closest_points = p.getClosestPoints(
                self.collision_object.body,
                self.collision_object.body,
                distance=max_distance,
                linkIndexA=self.collision_object.link,
                linkIndexB=joint_ind
            )

            # If bodies are above max_distance apart, nothing is returned, so
            # we just saturate at max_distance. Otherwise, take the minimum
            if len(closest_points) == 0:
                distances.append(max_distance)
            else:
                distances.append(np.min([point[8] for point in closest_points]))

        return np.array(distances)
