#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

# Inspired from http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
# Modified by Alexandre Vannobel to test the FollowJointTrajectory Action Server for the Kinova Gen3 robot

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_moveit_trajectories.py __ns:=my_gen3

import sys
import time
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_srvs.srv import Empty
import rospy
from std_msgs.msg import String
import tf2_ros
import numpy as np

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/nuno/catkin_ws_kortex/src/ros_kortex/kortex_examples/src/full_arm/')
from kinova_lfd import Kinova_LfD


class Kinova_LfD_data_record(object):
    """Kinova_LfD_data_record"""

    def __init__(self):
        # Initialize the node
        super(Kinova_LfD_data_record, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('Kinova_LfD_data_record')

        try:
            self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            # self.arm_group.set_planner_id('KPIECE')
            self.arm_group.set_planner_id('RRTstar')
            self.display_trajectory_publisher = rospy.Publisher(
                rospy.get_namespace() + 'move_group/display_planned_path',
                moveit_msgs.msg.DisplayTrajectory,
                queue_size=20)

            if self.is_gripper_present:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        except Exception as e:
            print(e)
            self.is_init_success = False
        else:
            self.is_init_success = True

    def reach_named_position(self, target):
        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        self.arm_group.set_named_target(target)
        # Plan the trajectory
        planned_path = self.arm_group.plan()
        # Execute the trajectory and block while it's not finished
        return self.arm_group.execute(planned_path, wait=True)

    def reach_joint_angles(self, joint_positions, tolerance):
        arm_group = self.arm_group
        success = True

        # Get the current joint positions
        # joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions before movement :")
        for p in joint_positions: rospy.loginfo(p)

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the joint target configuration
        # if self.degrees_of_freedom == 7:
        #  joint_positions[0] = pi/2
        #  joint_positions[1] = 0
        #  joint_positions[2] = pi/4
        #  joint_positions[3] = -pi/4
        #  joint_positions[4] = 0
        #  joint_positions[5] = pi/2
        #  joint_positions[6] = 0.2
        # elif self.degrees_of_freedom == 6:
        #  joint_positions[0] = 0
        #  joint_positions[1] = 0
        #  joint_positions[2] = pi/2
        #  joint_positions[3] = pi/4
        #  joint_positions[4] = 0
        #  joint_positions[5] = pi/2
        arm_group.set_joint_value_target(joint_positions)

        # Plan and execute in one command
        success &= arm_group.go(wait=True)

        # Show joint positions after movement
        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions after movement :")
        for p in new_joint_positions: rospy.loginfo(p)
        return success

    def get_cartesian_pose(self, display=False):
        # Get the current pose and display it
        pose = self.arm_group.get_current_pose()
        if display:
            rospy.loginfo("Actual cartesian pose is : ")
            rospy.loginfo(pose.pose)
        return pose.pose

    def get_joint_angles(self):
        joint_angles = self.arm_group.get_current_joint_values()
        # rospy.loginfo("Current joint angles is : ")
        # rospy.loginfo(joint_angles)

        return joint_angles

    def reach_cartesian_pose(self, pose, eef, tolerance, constraints):
        arm_group = self.arm_group

        # Set the tolerance
        arm_group.clear_pose_targets()
        arm_group.set_goal_position_tolerance(tolerance)
        arm_group.set_max_velocity_scaling_factor(0.4)

        # Set the trajectory constraint if one is specified
        if constraints is not None:
            arm_group.set_path_constraints(constraints)

        # Get the current Cartesian Position
        # arm_group.set_end_effector_link('end_effector_link')
        arm_group.set_pose_target(pose, eef)

        # Plan and execute
        rospy.loginfo("Planning and going to the Cartesian Pose")
        return arm_group.go(wait=True)

    def plan_traj(self, pose, tolerance, constraints):
        arm_group = self.arm_group

        arm_group.set_max_velocity_scaling_factor(0.5)
        arm_group.set_max_acceleration_scaling_factor(0.3)
        # Set the tolerance
        arm_group.set_goal_position_tolerance(tolerance)

        # Set the trajectory constraint if one is specified
        if constraints is not None:
            arm_group.set_path_constraints(constraints)

        # Get the current Cartesian Position
        arm_group.set_pose_target(pose)

        # Plan and execute
        rospy.loginfo("Planning and going to the Cartesian Pose")
        return arm_group.plan()

    def reach_gripper_position(self, relative_position):
        gripper_group = self.gripper_group

        # We only have to move this joint because all others are mimic!
        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        try:
            val = gripper_joint.move(
                relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos,
                True)
            return val
        except:
            return False


def listener():
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            # print ('Trying')
            trans = tfBuffer.lookup_transform('base_link', 'goal_frame', rospy.Time())
            print(trans)
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue


def main():
    example = Kinova_LfD_data_record()

    # For testing purposes
    success = example.is_init_success
    try:
        rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
    except:
        pass

    if success:

        # rospy.wait_for_service('/my_gen3/clear_octomap')
        # clear_octo = rospy.ServiceProxy('/my_gen3/clear_octomap', Empty)
        # clear_octo()

        actual_pose = example.get_cartesian_pose()
        lfd = Kinova_LfD()

        time.sleep(1)

        print('Move gripper to cube.')
        execute = input('Is the gripper in the right location? ')
        if execute == 'y':
            print('Saving current state')
            actual_pose = lfd.check_gripper_pose(actual_pose)

        elif execute == 'n':
            print('Closing program')
            return -1

        time.sleep(1)
        print('Move gripper to initial pose.')
        execute = input('Is the gripper in the right location? ')
        if execute == 'y':
            lfd.main(actual_pose)


if __name__ == '__main__':
    main()
