# 3DSGrasp
IEEE ICRA 2023 - [<b>3DSGrasp: 3D Shape-Completion for Robotic Grasp </b>](https://arxiv.org/abs/2301.00866)

## Tested on 
Ubuntu 20.04 and with ROS Noetic

This documentation needs work!
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
