## Framework and Code for the application of Normalized Advantage Function algorithm for Obstacle Avoidance on Robotic Manipulators

### Installation of requirements (on virtual environment)

1. Create a virtual environment
   
    ```bash
    $ python3 -m venv venv
    ```
   
2. Activate the virtual environment
   
    ```bash
    $ source venv/bin/activate
    ```
   
3. Install required packages

    ```bash
    $ python -m pip install -r requirements.txt
    ```

### Execution of preloaded trainings

There are two preloaded trainings in the framework, which can be used to test its functionality
or the work done in the project. The trainings correspond to the robotic manipulators KUKA LBR iiwa and 
xArm6, as stated in the report. The code neccessary to launch these training is located in the bottom of the 
**framework.py** file, commented. 

There are 4 sections commented, with the following titles: 
1. CODE FOR KUKA IIWA ROBOT
2. TESTING CODE FOR KUKA IIWA ROBOT
3. CODE FOR XARM6 ROBOT
4. TESTING CODE FOR XARM6 ROBOT

Sections **1** and **3** can be used to launch a training with the manipulator robot mentioned in the title
Sections **2** and **4** can be used to test the previous trainings, for the manipulators mentioned in the title.

It is important to note that **ONLY** the section that is required to be executed can be uncommented to ensure 
a controlled execution of the framework. It is also worth noting that the code related to the execution of the training
and testing expects to be running for 3000 episodes, with 400 timesteps per episode. If either the number of episodes or the number
of timesteps change, so must change the testing code related to that training, as the testing code will load the 
trained model from episode 3000. This means that, if the training is executed for less than 3000 episodes, the test 
will not find the required model, and that if more than 3000 episodes are executed, the test will only show the 
behaviour of the model trained until 3000 episodes. 


### Execution of customized training

1. **Initialize the framework**

    ```bash
    $ framework = ManipulatorFramework()
    ```

2. **Initialize environment**

    ```bash
    $ framework.initialize_environment(manipulator_file=<path_to_sdf_urdf_file>>,
                                       endeffector_index=<endeffector_index>,
                                       fixed_joints=<list_of_indices_of_joints_to_fix>,
                                       involved_joints=<list_of_indices_of_joints_to_involve>,
                                       target_position=<target_pos_as_3d_array>,
                                       obstacle_position=<obstacle_pos_as_3d_array>,
                                       initial_joint_positions=<list_of_initial_pos_for_each_joint>,
                                       initial_positions_variation_range=<list_of_variation_range_for_each_joint>,
                                       visualize=<bool>)
    ```

3. **Initialize NAF Agent**

    ```bash
    $ framework.initialize_naf_agent()
    ```

4. **Run Training**

    ```bash
    $ framework.run_training(n_episodes, n_timesteps)
    ```