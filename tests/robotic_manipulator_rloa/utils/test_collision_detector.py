from mock import MagicMock, patch
import pytest
import numpy as np

from robotic_manipulator_rloa.utils.collision_detector import CollisionDetector, CollisionObject


def test_collisionobject() -> None:
    """Test for the CollisionObject dataclass"""
    collision_object = CollisionObject(body='manipulator_body', link=0)
    assert collision_object.body == 'manipulator_body'
    assert collision_object.link == 0


@pytest.mark.parametrize('closest_points', [
    [(None, None, None, None, None, None, None, None, 1),
     (None, None, None, None, None, None, None, None, 3),
     (None, None, None, None, None, None, None, None, 5)],
    []
])
@patch('robotic_manipulator_rloa.utils.collision_detector.p')
def test_collision_detector(mock_pybullet: MagicMock,
                            closest_points: list) -> None:
    """Test for the CollisionDetector class"""
    collision_object = CollisionObject(body='manipulator_body', link=0)
    collision_detector = CollisionDetector(collision_object=collision_object,
                                           obstacle_ids=['obstacle'])

    assert collision_detector.obstacles == ['obstacle']
    assert collision_detector.collision_object == collision_object

    # ================== TEST FOR compute_distances() method ==================

    mock_pybullet.getClosestPoints.return_value = closest_points
    output = np.array([10.0]) if len(closest_points) == 0 else np.array([1])

    assert collision_detector.compute_distances() == output

    mock_pybullet.getClosestPoints.assert_any_call(collision_object.body,
                                                   'obstacle',
                                                   distance=10.0,
                                                   linkIndexA=collision_object.link)

    # ================== TEST FOR compute_collisions_in_manipulator() method ==================

    mock_pybullet.getClosestPoints.return_value = closest_points
    output = np.array([10.0]) if len(closest_points) == 0 else np.array([1])

    assert collision_detector.compute_collisions_in_manipulator([0, 3]) == output

    mock_pybullet.getClosestPoints.assert_any_call(collision_object.body,
                                                   collision_object.body,
                                                   distance=10.0,
                                                   linkIndexA=collision_object.link,
                                                   linkIndexB=3)
