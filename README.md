# 3DSGrasp: 3D Shape-Completion for Robotic Grasp 
(paper accepted for presentation at ICRA 2023)

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

## Citation 
If you find this code useful in your research, please consider citing our [paper](https://arxiv.org/abs/2301.00866):

	Mohammadi, S. S., Duarte, N. F., Dimou, D., Wang, Y., Taiana, M., Morerio, P., Dehban, A., Moreno, P., Bernardino, A., Del Bue, A., Santos-Victor, J. (2023). 3DSGrasp: 3D Shape-Completion for Robotic Grasp. arXiv preprint arXiv:2301.00866

