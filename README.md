# 3DSGrasp
![Language grade: Python](https://img.shields.io/badge/python-3.7|3.8%20-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

IEEE ICRA 2023 - [<b>3DSGrasp: 3D Shape-Completion for Robotic Grasp </b>](https://arxiv.org/abs/2301.00866) [[<b> Youtube Video </b>](https://youtu.be/i_v4EX_Nkls)]

We present a grasping strategy, named 3DSGrasp, that predicts the missing geometry from the partial Point-Cloud data (PCD) to produce reliable grasp poses. Our proposed PCD completion network is a Transformer-based encoder-decoder network with an Offset-Attention layer. Our network is inherently invariant to the object pose and point's permutation, which generates PCDs that are geometrically consistent and completed properly. Experiments on a wide range of partial PCD show that 3DSGrasp outperforms the best state-of-the-art method on PCD completion tasks and largely improves the grasping success rate in real-world scenarios.

<img src="media/first_gif.gif" width="400" height="225" /> <img src="media/second_gif.gif" width="400" height="225" />

# :computer:  Quick Start
To begin, clone this repository locally
```bash
git clone git@github.com:NunoDuarte/3DSGrasp.git
$ export 3DSG_ROOT=$(pwd)/3DSGrasp
```
This repo was tested on Ubuntu 20.04 and with ROS Noetic

Install requirements:
```bash
$ cd $3DSG_ROOT
$ conda create -n 3dsg_venv python=3.8  # or use virtualenv
$ conda activate 3dsg_venv
$ sh install.sh
```

If you want to:
- run the full pipeline goto [Pipeline](#step-by-step-of-3dsgrasp)
- run the completion network only goto [Completion Network](#completion-network)
- run only GPD for point cloud data either partial.pc or complete.pc goto [GPD](#gpd-for-point-cloud)

# :page_facing_up: Dependencies
- OpenCV 
- pylsl
- numpy
- os
- math
- msgpack
- zmq
- Tensorflow with CUDA
	- CUDA 11.2; Tensorflow 2.7; Cudnn v8.1; nvidia driver 460.32 (for RTX 3090)

# Step by step of 3DSGrasp
Open terminals:
1. ROS KINOVA
```bash
source catkin/devel/setup.bash
roslaunch kortex_driver kortex_driver.launch
```
(optional) import rviz environment and/or table for collision detection
```
open config file ~/Documents/kinova_grasping/grasp_kinova.rviz
go to scene objects -> import -> ~/Documents/kinova_grasping/table -> Publish
```
2. ROS KINOVA CAMERA
```bash
source catkin/devel/setup.bash
roslaunch kinova_vision kinova_vision_rgbd.launch device:=10.0.3.26
```
If robot is not in initial position do this, otherwise skip 
```bash
source catkin/devel/setup.bash
roslaunch kortex_examples reach_approach_grasp_pose.launch
```
In the terminal answer to the prompted questions
```
initial pose? y
CTR+C
```
3. COMPUTE GRASPS FROM POINT CLOUD
```
source catkin_ws/devel/setup.bash
cd ~/Documents/kinova_grasping
python main_agile.py
```
if segmentation fails:
"quit every plot; imediately ctrl c (multiple times), wait to close, run again until it works"

# :information_source: Information:
1. to change closure of gripper:
```
in /home/nuno/catkin_ws_kortex/src/ros_kortex/kortex_examples/src/move_it/reach_approach_grasp_pose.py
change the variable approach.example_send_gripper_command(0.3)
```
to change gripper size:
```
in/home/nuno/catkin_ws/src/agile_grasp/src/nodes/find_grasps.cpp
change these variable: const double HAND_OUTER_DIAMETER = 0.20;
```
then compile
```bash
cd /home/nuno/catkin_ws
catkin_make
```
2. For acquiring the point cloud and segmenting it run:
```bash
python main_agile.py
```
(saves the acquired_point cloud as original_pc and the segmented as partial_pc in tmp_data)

After getting the grasps and copying them in test_grasps_agile.py run:
```bash
cd ~/Documents/kinova_grasping
python test_grasps_agile.py
```
(computes the grasp pose in the base frame, publishes it in the goal_frame topic, and saves it in final_pose.npy in tmp_data)
Finally execute grasp pose in real robot using:
```bash
roslaunch kortex_examples reach_grasp_pose.launch 
```

### GPD for Point Cloud
```
catkin_make -DGPD_LIB=/home/nuno/Documents/third_party/gpd/build/INSTALL/lib/libgpd.so -DGPD_INCLUDE_DIRS=/home/nuno/Documents/third_party/gpd/build/INSTALL/include/
```

# Add documentation on how to run:
- [ ] model
- [ ] kinova
- [ ] gpd
- [ ] moveit
- [ ] instruction to run complete pipeline


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
