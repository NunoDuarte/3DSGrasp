# 3DSGrasp
![Language grade: Python](https://img.shields.io/badge/python-3.7|3.8%20-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

IEEE ICRA 2023 - [<b>3DSGrasp: 3D Shape-Completion for Robotic Grasp </b>](https://arxiv.org/abs/2301.00866) [[<b> Youtube Video </b>](https://youtu.be/i_v4EX_Nkls)]

We present a grasping strategy, named 3DSGrasp, that predicts the missing geometry from the partial Point-Cloud data (PCD) to produce reliable grasp poses. Our proposed PCD completion network is a Transformer-based encoder-decoder network with an Offset-Attention layer. Our network is inherently invariant to the object pose and point's permutation, which generates PCDs that are geometrically consistent and completed properly. Experiments on a wide range of partial PCD show that 3DSGrasp outperforms the best state-of-the-art method on PCD completion tasks and largely improves the grasping success rate in real-world scenarios.

<img src="media/first_gif.gif" width="400" height="225" /> <img src="media/second_gif.gif" width="400" height="225" />

# :computer:  Quick Start
:arrow_heading_down: <b>After installing follow the appropriate instructions if you want to:</b>
- run the full pipeline :arrow_right: [Full Pipeline](#page_facing_up-step-by-step-of-3dsgrasp)
- run the completion network only :arrow_right: [Completion Network](#completion-network)
- run only GPD for point cloud data either partial.pc or complete.pc :arrow_right: [Test GPD](#gpd-for-point-cloud)

To begin, clone this repository locally
```bash
git clone git@github.com:NunoDuarte/3DSGrasp.git
$ export 3DSG_ROOT=$(pwd)/3DSGrasp
```
This repo was tested on Ubuntu 20.04 and with ROS Noetic

## Install requirements for Completion Network:
```bash
$ cd $3DSG_ROOT
$ conda create -n 3dsg_venv python=3.8  # or use virtualenv
$ conda activate 3dsg_venv
$ sh install.sh
```

## Install ROS, ROS_kortex + ROS_kortex_vision (Kinova gen3), Moveit, GPD for full Pipeline
- Check oficial documentation for [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu)
- Check oficial documentation for [ROS_kortex](https://github.com/Kinovarobotics/ros_kortex) and [ROS_kortex_vision](https://github.com/Kinovarobotics/ros_kortex_vision)
- Check oficial documentation for [Moveit](https://moveit.ros.org/install/) or just do this 
```bash
sudo apt install ros-noetic-moveit
```
- Check oficial documentation for [GPD](https://github.com/atenpas/gpd) (:warning: gpd repo was tested on Ubuntu 16.04; if you trouble installing on Ubuntu 20.04 send an issue to us and we'll help)

## Install GPD to test see grasps generated of your partial.pc or complete.pc 
- Check oficial documentation for [GPD](https://github.com/atenpas/gpd) (:warning: gpd repo was tested on Ubuntu 16.04; if you trouble installing on Ubuntu 20.04 send an issue to us and we'll help)

# :page_facing_up: Step by step of 3DSGrasp
Open terminals:
1. ROS KINOVA
```bash
source catkin/devel/setup.bash
roslaunch kortex_driver kortex_driver.launch
```
(optional) import rviz environment and/or table for collision detection
```
open config file 3DSG_ROOT/ROS/rviz/grasp_kinova.rviz
go to scene objects -> import -> 3DSG_ROOT/ROS/rviz/my_table -> Publish
```
2. ROS KINOVA VISION
```bash
source catkin/devel/setup.bash
roslaunch kinova_vision kinova_vision_rgbd.launch device:=$IP_KINOVA
```
3. Configure kinova and gpd files
set an initial pose for kinova manually or (optional) set it as a .npy file and load it in reach_approach_grasp_pose.py
```bash
cd 3DSG_ROOT/ROS/src/
```
(optional) open reach_approach_grasp_pose.py and set location of initial_state.npy of kinova 
```python
        # Load initial state of robot (joint angles)
        initial_state = np.load('location_of_initial_state.npy')
```
set location of final_pose.npy and final_approach.npy (These are the best grasp and approach from GPD)
```python
        print('Load grasp pose')
        final_pose = np.load('location_of_final_pose.npy')
        final_approach = np.load('location_of_final_approach.npy')
```
4. RUN PIPELINE
```
source catkin_ws/devel/setup.bash
cd 3DSG_ROOT/
python main_gpd.py
```
if segmentation fails (partial point cloud includes artifacts then "quit every plot; imediately ctrl c (multiple times), wait to close, run again"

5. RUN ON KINOVA
```bash
source catkin/devel/setup.bash
roslaunch kortex_examples reach_approach_grasp_pose.launch
```

# :information_source: Information:
- When grasping the closure of the gripper is predefine, if you want to change it is
```
in reach_approach_grasp_pose.py
change the variable approach.example_send_gripper_command(0.3)
```

- For acquiring the point cloud and segmenting it run:
```bash
python main_agile.py
```
it saves the acquired_point cloud as original_pc and the segmented as partial_pc in tmp_data


### GPD for Point Cloud
To run GPD on .pcd file 
```
cd $GPD_ROOT/build
./detect_grasps ../cfg/eigen_params.cfg $LOCATION_OF_FILE.PCD 
```

# :soon: The Dataset and pre-trained model will get released soon 
# Completion Network

## Citation 
If you find this code useful in your research, please consider citing our [paper](https://arxiv.org/abs/2301.00866):
```bibtex
@article{mohammadi20233dsgrasp,
  title={3DSGrasp: 3D Shape-Completion for Robotic Grasp},
  author={Mohammadi, Seyed S and Duarte, Nuno F and Dimou, Dimitris and Wang, Yiming and Taiana, Matteo and Morerio, Pietro and Dehban, Atabak and Moreno, Plinio and Bernardino, Alexandre and Del Bue, Alessio and others},
  journal={arXiv preprint arXiv:2301.00866},
  year={2023}
}
```
