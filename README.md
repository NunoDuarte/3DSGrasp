# 3DSGrasp
![Language grade: Python](https://img.shields.io/badge/python-3.7|3.8%20-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

IEEE ICRA 2023 - [<b>3DSGrasp: 3D Shape-Completion for Robotic Grasp </b>](https://arxiv.org/abs/2301.00866) [[<b> Youtube Video </b>](https://youtu.be/i_v4EX_Nkls)]

We present a grasping strategy, named 3DSGrasp, that predicts the missing geometry from the partial Point-Cloud data (PCD) to produce reliable grasp poses. Our proposed PCD completion network is a Transformer-based encoder-decoder network with an Offset-Attention layer. Our network is inherently invariant to the object pose and point's permutation, which generates PCDs that are geometrically consistent and completed properly. Experiments on a wide range of partial PCD show that 3DSGrasp outperforms the best state-of-the-art method on PCD completion tasks and largely improves the grasping success rate in real-world scenarios.

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
# Add documentation on how to run:
- [ ] model
- [ ] kinova
- [ ] gpd
- [ ] moveit
- [ ] instruction to run complete pipeline

### Find best grasp candidate from GPD 
```
catkin_make -DGPD_LIB=/home/nuno/Documents/third_party/gpd/build/INSTALL/lib/libgpd.so -DGPD_INCLUDE_DIRS=/home/nuno/Documents/third_party/gpd/build/INSTALL/include/
```
## The Dataset and pre-trained model will get released soon 

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
