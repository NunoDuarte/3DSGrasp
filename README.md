# 3DSGrasp
![Language grade: Python](https://img.shields.io/badge/python-3.7|3.8%20-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

IEEE ICRA 2023 - [<b>3DSGrasp: 3D Shape-Completion for Robotic Grasp </b>](https://ieeexplore.ieee.org/document/10160350) [[<b> Youtube Video </b>](https://youtu.be/i_v4EX_Nkls)]

We present a grasping strategy, named 3DSGrasp, that predicts the missing geometry from the partial Point-Cloud data (PCD) to produce reliable grasp poses. Our proposed PCD completion network is a Transformer-based encoder-decoder network with an Offset-Attention layer. Our network is inherently invariant to the object pose and point's permutation, which generates PCDs that are geometrically consistent and completed properly. Experiments on a wide range of partial PCD show that 3DSGrasp outperforms the best state-of-the-art method on PCD completion tasks and largely improves the grasping success rate in real-world scenarios.

<img src="media/first_gif.gif" width="400" height="225" /> <img src="media/second_gif.gif" width="400" height="225" />

# :computer:  Quick Start
:arrow_heading_down: <b>After installing follow the appropriate instructions if you want to:</b>
- run the full pipeline (from camera depth input to kinova grasping the object) :arrow_right: [Full Pipeline](#page_facing_up-step-by-step-of-3dsgrasp-pipeline)
- run only the completion network to generate shape completion on a partial.pc :arrow_right: [Completion Network](#completion-network)
- run only GPD to generate grasp candidates for point cloud data of either partial.pc or complete.pc :arrow_right: [Test GPD](#page_facing_up-gpd-for-point-cloud)
- :train2: use our model? :arrow_right: [Completion Network](#completion-network)
- :vertical_traffic_light: use the same train-test split of the YCB dataset? :arrow_right: [Completion Network](#completion-network)

# :key: Installations
To begin, clone this repository locally
```bash
git clone git@github.com:NunoDuarte/3DSGrasp.git
$ export 3DSG_ROOT=$(pwd)/3DSGrasp
```
This repo was tested on Ubuntu 20.04 and with ROS Noetic

## :key: Install requirements for Completion Network:
```bash
$ cd $3DSG_ROOT
$ conda create -n 3dsg_venv python=3.8  # or use virtualenv
$ conda activate 3dsg_venv
$ sh install.sh
```

## :key: Install requirements for full Pipeline
- Check oficial documentation for [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu)
- Check oficial documentation for [ROS_kortex](https://github.com/Kinovarobotics/ros_kortex) and [ROS_kortex_vision](https://github.com/Kinovarobotics/ros_kortex_vision)
- Check oficial documentation for [Moveit](https://moveit.ros.org/install/) or just do this 
```bash
sudo apt install ros-noetic-moveit
```
- Check oficial documentation for [GPD](https://github.com/atenpas/gpd) (:warning: gpd repo was tested on Ubuntu 16.04; if you have trouble installing on Ubuntu 20.04 send an issue to us and we'll help)

## :key: Install requirements to test GPD (see grasps generated of your partial.pc or complete.pc)
- Check oficial documentation for [GPD](https://github.com/atenpas/gpd) (:warning: gpd repo was tested on Ubuntu 16.04; if you have trouble installing on Ubuntu 20.04 send an issue to us and we'll help)

# :page_facing_up: Step by step to run 3DSGrasp Pipeline
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
- When grasping the closure of the gripper is predefine, if you want to change open ```reach_approach_grasp_pose.py``` and change variable
```  python
approach.example_send_gripper_command(0.3)
```

- For acquiring the point cloud and segmenting it run:
```bash
python main_agile.py
```
it saves the acquired_point cloud as original_pc and the segmented as partial_pc in tmp_data


# :page_facing_up: GPD for Point Cloud
To run GPD on .pcd file 
```
cd $GPD_ROOT/build
./detect_grasps ../cfg/eigen_params.cfg $LOCATION_OF_FILE.PCD 
```


## Completion Network

To train point cloud completion model  
```
cd Completion
python3 main.py --config  ./cfgs/YCB_models/SGrasp.yaml
```

To test point cloud completion model  
```
python3 main.py  --test --ckpts /PATH_TO_pre_trained_MODEL/MODEL.pth --config  ./cfgs/YCB_models/SGrasp.yaml
```
# Note that the input of the network for the real-world grasping experiment should be a single sample

## :tada: The pre-trained model is [here](https://drive.google.com/file/d/11vTsY0MQw9pzsqz3MyvCKjQT2rQ9VxVi/view?usp=share_link) (around 500 MB)!!!

## Citation 
If you find this code useful in your research, please consider citing our paper. Available on [IEEE Xplore](https://ieeexplore.ieee.org/document/10160350) and [ArXiv](https://arxiv.org/abs/2301.00866):
```bibtex
@INPROCEEDINGS{10160350,
  author={Mohammadi, Seyed S. and Duarte, Nuno F. and Dimou, Dimitrios and Wang, Yiming and Taiana, Matteo and Morerio, Pietro and Dehban, Atabak and Moreno, Plinio and Bernardino, Alexandre and Del Bue, Alessio and Santos-Victor, José},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={3DSGrasp: 3D Shape-Completion for Robotic Grasp}, 
  year={2023},
  volume={},
  number={},
  pages={3815-3822},
  doi={10.1109/ICRA48891.2023.10160350}
}
```
