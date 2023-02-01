import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
import tf2_geometry_msgs  # **Do not use geometry_msgs. Use this instead for PoseStamped
import rospy
import numpy as np


def transform_pose(input_pose, from_frame, to_frame):
    # **Assuming /tf2 topic is being broadcasted
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = input_pose
    pose_stamped.header.frame_id = from_frame

    rospy.sleep(2)
    try:
        # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
        output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(1))
        return output_pose_stamped.pose

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        raise


def publish_tf(px, py, pz, rx, ry, rz, rw, par_frame="base_link", child_frame="goal_frame"):
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = par_frame
    static_transformStamped.child_frame_id = child_frame

    static_transformStamped.transform.translation.x = float(px)
    static_transformStamped.transform.translation.y = float(py)
    static_transformStamped.transform.translation.z = float(pz)

    static_transformStamped.transform.rotation.x = float(rx)
    static_transformStamped.transform.rotation.y = float(ry)
    static_transformStamped.transform.rotation.z = float(rz)
    static_transformStamped.transform.rotation.w = float(rw)

    broadcaster.sendTransform(static_transformStamped)
    rospy.sleep(5)

    return True


def str_to_np (grasp_string):
    grasp_pose = []
    for line in g.splitlines():
        d = line.split(':')
        nums = d[1].split()
        if d[0] == "grasp width" or d[0] == "grasp surface" : continue
        #print ()
        #print ([float(n) for n in nums])
        grasp_pose.append([float(n) for n in nums])

    grasp_pose = np.asarray(grasp_pose)
    return grasp_pose


def read_grasps():
    grasp_poses = []
    with open(tmp_dir + 'gpd_grasp_poses.txt') as f:
        lines = f.readlines()
        for idx in range(0, len(lines), 3):
            grasp_pose = []
            for i in range(3):
                nums = lines[idx + i].split()
                grasp_pose.append([float(n) for n in nums])
            grasp_poses.append(grasp_pose)

    return np.array(grasp_poses, dtype="object")


def coord_to_transform(grasp_pose):
    #grasp_pose = np.load(tmp_dir + 'grasp_pose.npy') 
    quat = grasp_pose[2]
    
    # grasp_bottom = grasp_pose[1]
    # grasp_bottom[0] = grasp_bottom[0] + 0.025
    # grasp_bottom[2] = grasp_bottom[2] + 0.02

    # bottom = grasp_bottom
    getApproach = np.array(grasp_pose[1])
    grasp_bottom = grasp_pose[0]
    bottom = grasp_bottom - 0.03 * getApproach
                       
    #r = R.from_matrix([[binormal[0],  axis[0], approach[0]],
    #                   [binormal[1],  axis[1], approach[1]],
    #                   [binormal[2],  axis[2], approach[2]]])
                       
    #quat = r.as_quat()
    quatR = R.from_quat(quat)
    print(quat)
    quat_to_mat = quatR.as_matrix()
    print(quat_to_mat)

    transf1 = R.from_euler('y', -90, degrees=True)
    transf = R.from_euler('zyx', [[90, 0, 0], [0, -90, 0], [0, 0, 0]], degrees=True)
    transf2 = R.from_euler('z', -90, degrees=True)
    matF = transf1.apply(quat_to_mat)
    matF = transf2.apply(matF)
    matF_to_quat = R.from_matrix(matF).as_quat()

    return np.concatenate((bottom, matF_to_quat))

def approach_coord_to_transform(grasp_pose):
    quat = grasp_pose[2]

    # grasp_bottom = grasp_pose[1]
    # grasp_bottom[0] = grasp_bottom[0] + 0.025
    # grasp_bottom[2] = grasp_bottom[2] + 0.02

    # bottom = grasp_bottom
    getApproach = np.array(grasp_pose[1])
    grasp_bottom = grasp_pose[0]
    bottom = grasp_bottom + 0.12 * getApproach

    quatR = R.from_quat(quat)
    quat_to_mat = quatR.as_matrix()

    transf1 = R.from_euler('y', -90, degrees=True)
    transf2 = R.from_euler('z', -90, degrees=True)
    matF = transf1.apply(quat_to_mat)
    matF = transf2.apply(matF)
    matF_to_quat = R.from_matrix(matF).as_quat()

    return np.concatenate((bottom, matF_to_quat))

tmp_dir = 'tmp_data/'

grasp_poses = read_grasps()
print('Number of grasp poses: ', grasp_poses.shape[0])
#######  ---------------------------// ---------------------------------- #####
# change grasp [k]
# pose = coord_to_transform(grasp_poses[0])
k = grasp_poses[4]
pose = coord_to_transform(k)
approach = approach_coord_to_transform(k)

#######  ---------------------------// ---------------------------------- #####

print (grasp_poses.shape)


print ('Pose')
print (pose.shape)

#rospy.init_node('my_static_tf2_broadcaster')

my_pose = Pose()
my_pose.position.x = pose[0]
my_pose.position.y = pose[1]
my_pose.position.z = pose[2]
my_pose.orientation.x = pose[3] 
my_pose.orientation.y = pose[4] 
my_pose.orientation.z = pose[5] 
my_pose.orientation.w = pose[6] 

rospy.init_node('listener', anonymous=True)
### Transform the pose from the camera frame to the base frame (world)
transformed_pose = transform_pose(my_pose, "camera_depth_frame", "base_link")

# apply matrix rotations (2 transformations)
# quat = [transformed_pose.orientation.x, transformed_pose.orientation.y, transformed_pose.orientation.z, transformed_pose.orientation.w]
# quatR = R.from_quat(quat)
# print(quat)
# quat_to_mat = quatR.as_matrix()
# print(quat_to_mat)
#
# transf1 = R.from_euler('y', 90, degrees=True)
# transf2 = R.from_euler('z', 90, degrees=True)
# transf = R.from_euler('zyx', [[90, 90, 90], [90, 90, 90], [0, 0, 0]], degrees=True)
# matF = transf.apply(quat_to_mat)
#
# matF_to_quat = R.from_matrix(matF).as_quat()

#transformed_pose.orientation.x = matF_to_quat[0]
#transformed_pose.orientation.y = matF_to_quat[1]
#transformed_pose.orientation.z = matF_to_quat[2]
#transformed_pose.orientation.w = matF_to_quat[3]
print(transformed_pose)

final_pose = np.array([transformed_pose.position.x,
                       transformed_pose.position.y,
                       transformed_pose.position.z,
                       transformed_pose.orientation.x,
                       transformed_pose.orientation.y,
                       transformed_pose.orientation.z,
                       transformed_pose.orientation.w])

### Publish the transform as goal_frame topic for visualization in RViz
publish_tf(final_pose[0], final_pose[1], final_pose[2], final_pose[3], final_pose[4], final_pose[5], final_pose[6], child_frame='complete_goal_frame')

np.save(tmp_dir + 'final_pose.npy', final_pose)

### Final approach
my_pose = Pose()
my_pose.position.x = approach[0]
my_pose.position.y = approach[1]
my_pose.position.z = approach[2]
my_pose.orientation.x = approach[3]
my_pose.orientation.y = approach[4]
my_pose.orientation.z = approach[5]
my_pose.orientation.w = approach[6]

rospy.init_node('listener', anonymous=True)
### Transform the pose from the camera frame to the base frame (world)
transformed_pose = transform_pose(my_pose, "camera_depth_frame", "base_link")

print(transformed_pose)

final_pose = np.array([transformed_pose.position.x,
                       transformed_pose.position.y,
                       transformed_pose.position.z,
                       transformed_pose.orientation.x,
                       transformed_pose.orientation.y,
                       transformed_pose.orientation.z,
                       transformed_pose.orientation.w])

np.save(tmp_dir + 'final_approach.npy', final_pose)