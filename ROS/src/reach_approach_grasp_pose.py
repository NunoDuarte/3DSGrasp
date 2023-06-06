#!/usr/bin/env python

# insert the location of the second file or put it in the same local folder
# sys.path.insert(1, 'location_of_approach_object_movement.py')
from approach_object_movement import Approach_Object

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


class ExampleMoveItTrajectories(object):
    """ExampleMoveItTrajectories"""

    def __init__(self):
        # Initialize the node
        super(ExampleMoveItTrajectories, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('example_move_it_trajectories')

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

        arm_group.set_joint_value_target(joint_positions)

        # Plan and execute in one command
        success &= arm_group.go(wait=True)

        # Show joint positions after movement
        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions after movement :")
        for p in new_joint_positions: rospy.loginfo(p)
        return success

    def get_cartesian_pose(self):
        # Get the current pose and display it
        pose = self.arm_group.get_current_pose()
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
    example = ExampleMoveItTrajectories()

    # For testing purposes
    success = example.is_init_success
    try:
        rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
    except:
        pass

    if success:
        # Load initial state of robot (joint angles)
        initial_state = np.load('/home/nuno/Documents/kinova_grasping/tmp_data/initial_state_4.npy') 	 # gripper
        initial_state = list(initial_state)
        print('Loaded initial state: ', initial_state)

        execute = input('Go to initial pose? : ')
        if execute == 'y':
            print('Reaching initial state')
            success &= example.reach_joint_angles(initial_state, tolerance=0.01)

            if success:
                print('Reached successfully!')
            else:
                print('Failed to reach goal.')
                return -1

        actual_pose = example.get_cartesian_pose()

        print('Load grasp pose')
        final_pose = np.load('/home/nuno/Documents/kinova_grasping/tmp_data/final_pose.npy')
        final_approach = np.load('/home/nuno/Documents/kinova_grasping/tmp_data/final_approach.npy')

        actual_pose.position.x = final_pose[0]
        actual_pose.position.y = final_pose[1]
        actual_pose.position.z = final_pose[2]
        actual_pose.orientation.x = final_pose[3]
        actual_pose.orientation.y = final_pose[4]
        actual_pose.orientation.z = final_pose[5]
        actual_pose.orientation.w = final_pose[6]

        execute = input('Plan movement? (tool frame): ')
        if execute == 'y':
            print(actual_pose)
            success &= example.reach_cartesian_pose(pose=actual_pose, eef='tool_frame', tolerance=0.01,
                                                    constraints=None)
            if success:
                actual_pose = example.get_cartesian_pose()
                constraints = moveit_msgs.msg.Constraints()
                orientation_constraint = moveit_msgs.msg.OrientationConstraint()
                orientation_constraint.orientation = actual_pose.orientation
                constraints.orientation_constraints.append(orientation_constraint)

                # Approach the object
                execute = input('Plan movement? (end effector link frame): ')
                actual_approach = example.get_cartesian_pose()
                actual_approach.position.x = final_approach[0]
                actual_approach.position.y = final_approach[1]
                actual_approach.position.z = final_approach[2]
                actual_approach.orientation.x = final_approach[3]
                actual_approach.orientation.y = final_approach[4]
                actual_approach.orientation.z = final_approach[5]
                actual_approach.orientation.w = final_approach[6]
                if execute == 'y':
                    approach = Approach_Object()
                    approach.main(actual_approach)

                    execute = input('Close gripper? : ')
                    if execute == 'y':
                        # Let's close the gripper at 50%
                        approach.example_send_gripper_command(0.70)

                    execute = input('Lift object? : ')
                    if execute == 'y':
                        actual_approach.position.z = final_approach[2] + 0.15
                        approach.example_send_cartesian_pose(actual_approach)

                    execute = input('Open gripper? : ')
                    if execute == 'y':
                        approach.example_send_gripper_command(0)
            else:
                print('Failed to reach goal.')
                print('Stopping')
                return -1

        else:
            print('Stopping')

        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/moveit_general_python", success)

        if not success:
            rospy.logerr("The example encountered an error.")


if __name__ == '__main__':
    main()
