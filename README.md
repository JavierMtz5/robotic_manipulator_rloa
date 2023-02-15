# robotic_manipulator_rloa

**robotic_manipulator_rloa** is a framework for training Robotic Manipulators on the Obstacle Avoidance task through Reinforcement Learning.

## Installation

Install the package with [pip](https://pip.pypa.io/en/stable/).

```bash
$ pip install robotic_manipulator_rloa
```

## Usage

### Execution of a demo training and testing process for the KUKA IIWA Robotic Manipulator

```python
from robotic_manipulator_rloa import ManipulatorFramework

# Initialize the framework
mf = ManipulatorFramework()

# Run a demo of the training process for the KUKA IIWA Robotic Manipulator
mf.run_demo_training('kuka_training', verbose=False)

# Run a demo of the testing process for the KUKA IIWA Robotic Manipulator
mf.run_demo_testing('kuka_testing')
```

### Execution of a training for the KUKA IIWA Robotic Manipulator

```python
from robotic_manipulator_rloa import ManipulatorFramework

# Initialize the framework
mf = ManipulatorFramework()

# Initialize KUKA IIWA Robotic Manipulator environment
mf.initialize_environment(manipulator_file='kuka_iiwa/kuka_with_gripper2.sdf',
                          endeffector_index=13,
                          fixed_joints=[6, 7, 8, 9, 10, 11, 12, 13],
                          involved_joints=[0, 1, 2, 3, 4, 5],
                          target_position=[0.4, 0.85, 0.71],
                          obstacle_position=[0.45, 0.55, 0.55],
                          initial_joint_positions=[0.9, 0.45, 0, 0, 0, 0],
                          initial_positions_variation_range=[0, 0, 0.5, 0.5, 0.5, 0.5],
                          visualize=False)

# Initialize NAF Agent (checkpoint files will be generated every 100 episodes)
mf.initialize_naf_agent(checkpoint_frequency=100)

# Run training for 3000 episodes, 400 timesteps per episode
mf.run_training(3000, 400, verbose=False)
```

### Execution of a testing process for the KUKA IIWA Robotic Manipulator (must execute a training for 3000 episodes before)

```python
import os
import pybullet_data
from robotic_manipulator_rloa import ManipulatorFramework

# Initialize the framework
mf = ManipulatorFramework()

# Initialize KUKA IIWA Robotic Manipulator environment
kuka_path = os.path.join(pybullet_data.getDataPath(), 'kuka_iiwa/kuka_with_gripper2.sdf')
mf.initialize_environment(manipulator_file=kuka_path,
                          endeffector_index=13,
                          fixed_joints=[6, 7, 8, 9, 10, 11, 12, 13],
                          involved_joints=[0, 1, 2, 3, 4, 5],
                          target_position=[0.4, 0.85, 0.71],
                          obstacle_position=[0.45, 0.55, 0.55],
                          initial_joint_positions=[0.9, 0.45, 0, 0, 0, 0],
                          initial_positions_variation_range=[0, 0, .5, .5, .5, .5],
                          visualize=False)

# Initialize NAF Agent
mf.initialize_naf_agent()

# Load pretrained weights from .p file
mf.load_pretrained_parameters_from_episode(3000)

# Test the pretrained model for 50 test episodes, 750 timesteps each
mf.test_trained_model(50, 750)

```

## Contributing

Pull requests are welcome! For major changes, please open an issue first
to discuss what you would like to change. Please make sure to update and execute the tests!

```bash
robotic_manipulator_rloa$ pytest --cov-report term-missing --cov=robotic_manipulator_rloa/ tests/robotic_manipulator_rloa/
```

## License

[MIT License](https://choosealicense.com/licenses/mit/)