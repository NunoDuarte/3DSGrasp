from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import subprocess
import rospy
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs  # **Do not use geometry_msgs. Use this instead for PoseStamped
import signal
import ctypes
import struct
import ros_numpy


def handler(signum, frame):
    res = input("Ctrl-c was pressed. Do you really want to exit? [Y]/n ")
    if res == 'y':
        exit(1)


def transform_pose(input_pose, from_frame, to_frame):
    # **Assuming /tf2 topic is being broadcasted
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = input_pose
    pose_stamped.header.frame_id = from_frame
    # pose_stamped.header.stamp = rospy.Time.now()

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


def str_to_np(grasp_string):
    grasp_pose = []
    for line in g.splitlines():
        d = line.split(':')
        nums = d[1].split()
        if d[0] == "grasp width" or d[0] == "grasp surface": continue
        # print ()
        # print ([float(n) for n in nums])
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

    return np.array(grasp_poses)


def coord_to_transform(grasp_pose):
    quat = grasp_pose[2]

    getApproach = np.array(grasp_pose[1])
    grasp_bottom = grasp_pose[0]
    bottom = grasp_bottom - 0.03 * getApproach

    quatR = R.from_quat(quat)
    quat_to_mat = quatR.as_matrix()

    transf1 = R.from_euler('y', -90, degrees=True)
    transf2 = R.from_euler('z', -90, degrees=True)
    matF = transf1.apply(quat_to_mat)
    matF = transf2.apply(matF)
    matF_to_quat = R.from_matrix(matF).as_quat()

    return np.concatenate((bottom, matF_to_quat))


def approach_coord_to_transform(grasp_pose):
    quat = grasp_pose[2]

    getApproach = np.array(grasp_pose[1])
    grasp_bottom = grasp_pose[0]
    bottom = grasp_bottom + 0.08 * getApproach

    quatR = R.from_quat(quat)
    quat_to_mat = quatR.as_matrix()

    transf1 = R.from_euler('y', -90, degrees=True)
    transf2 = R.from_euler('z', -90, degrees=True)
    matF = transf1.apply(quat_to_mat)
    matF = transf2.apply(matF)
    matF_to_quat = R.from_matrix(matF).as_quat()

    return np.concatenate((bottom, matF_to_quat))


print('Starting')
rospy.init_node('listener', anonymous=True)
signal.signal(signal.SIGINT, handler)  # catch ctrl+c
point_cloud = rospy.wait_for_message("/camera/depth_registered/points", PointCloud2)

tmp_dir = 'tmp_data/'

# pc = []
# for p in pc2.read_points(point_cloud, field_names=("x", "y", "z"), skip_nans=True):
#     # print(" x : %f  y: %f  z: %f" % (p[0], p[1], p[2]))
#     pc.append([p[0], p[1], p[2]])
#

pc = ros_numpy.numpify(point_cloud)
pc = ros_numpy.point_cloud2.split_rgb_field(pc)
pc = np.array(pc)

points = []
rgb = []
for x in pc:
    for y in x:
        points.append([y[0], y[1], y[2]])
        rgb.append([y[3], y[4], y[5]])

pc = points

# gen = pc2.read_points(point_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=False)
# int_data = list(gen)
# pc = []
# rgb = []
# R = []
# G = []
# B = []
# for x in int_data:
#     pc.append([x[0], x[1], x[2]])
#     test = x[3]
#     # cast float32 to int so that bitwise operations are possible
#     s = struct.pack('>f', test)
#     i = struct.unpack('>l', s)[0]
#     # you can get back the float value by the inverse operations
#     pack = ctypes.c_uint32(i).value
#     # r = (pack & 0x00FF0000) >> 16
#     # g = (pack & 0x0000FF00) >> 8
#     # b = (pack & 0x000000FF)
#
#     b = pack & 0xff
#     g = (pack >> 8) & 0xff
#     r = (pack >> 16) & 0xff
#
#     # print (r, g, b)  # prints r,g,b values in the 0-255 range
#     rgb.append([r, g, b])
#     R.append(r)
#     G.append(g)
#     B.append(b)
#
# # x,y,z can be retrieved from the x[0],x[1],x[2]
print(np.shape(pc))
print(np.shape(rgb))

# Segmentation of Point Cloud
xyz = np.asarray(pc)
rgb = np.asarray(rgb)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.colors = o3d.utility.Vector3dVector(rgb)
# o3d.visualization.draw_geometries([pcd])

# Segmentation of Point Cloud
xyz = np.asarray(pc)
idx = np.where(xyz[:, 2] < 0.6)     # Prune point cloud to 0.6 meters from camera in z direction
xyz = xyz[idx]
rgb = rgb[idx]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.uint8) / 255.0)
o3d.visualization.draw_geometries([pcd])

plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                         ransac_n=5,
                                         num_iterations=1000)
[a, b, c, d] = plane_model

# Partial Point Cloud
inlier_cloud = pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)

outlier_cloud, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=60,
                                                              std_ratio=0.6)
# outlier_cloud.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([outlier_cloud])

# # Save point clouds in pcd format
# o3d.io.write_point_cloud(tmp_dir + 'partial_pc.pcd', outlier_cloud)
# o3d.io.write_point_cloud(tmp_dir + 'original_pc.pcd', pcd)
#
# # Save point clouds in xyz format
# o3d.io.write_point_cloud(tmp_dir + 'partial_pc.xyz', outlier_cloud)
# o3d.io.write_point_cloud(tmp_dir + 'original_pc.xyz', pcd)

input()

print()
print('Object completion')
# subprocess.run(
#     ["/home/nuno/virtual_envs/poinTr/bin/python3.8", "/home/nuno/Documents/third_party/completion/main.py", "--test",
#      "--ckpts", "/home/nuno/Documents/third_party/completion/ycb_all_prebest_loss_pietro.pth", "--config",
#      "/home/nuno/Documents/third_party/completion/cfgs/ShapeNet55_models/PoinTr.yaml"])

subprocess.run(
    ["/home/nuno/virtual_envs/poinTr/bin/python3.8", "/home/nuno/Documents/third_party/completion/main.py", "--test",
     "--ckpts", "/home/nuno/Documents/third_party/completion/ycb_all_prebest_loss_model_chanheset.pth", "--config",
     "/home/nuno/Documents/third_party/completion/cfgs/ShapeNet55_models/PoinTr.yaml"])

compl_pc = o3d.io.read_point_cloud('/home/nuno/Documents/kinova_grasping/tmp_data/complete_pc_x.pcd')
compl_pc.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([compl_pc, outlier_cloud])

merged_xyz = np.concatenate((np.asarray(compl_pc.points), np.asarray(pcd.points)), axis=0)
merged_pc = o3d.geometry.PointCloud()
merged_pc.points = o3d.utility.Vector3dVector(merged_xyz)
merged_pc.paint_uniform_color([0.5, 0.706, 0.3])
o3d.visualization.draw_geometries([merged_pc])

pub = rospy.Publisher("/camera/depth_registered/points", PointCloud2, queue_size=2)

points = []
for i in range(merged_xyz.shape[0]):
    points.append([merged_xyz[i, 0] + 0.02, merged_xyz[i, 1], merged_xyz[i, 2]])

fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          # PointField('rgba', 12, PointField.UINT32, 1),
          ]

header = Header()
header.frame_id = "camera_depth_frame"
pc2 = point_cloud2.create_cloud(header, fields, points)

for i in range(100):
    pc2.header.stamp = rospy.Time.now()
    pub.publish(pc2)
    rospy.sleep(0.1)

# exit(0)
print()
print('Partial grasps')

subprocess.call(
    './detect_grasps ../cfg/eigen_params_complete.cfg /home/nuno/Documents/kinova_grasping/tmp_data/partial_pc.pcd',
    shell=True, cwd='/home/nuno/Documents/third_party/gpd/build')

grasp_poses = read_grasps()
print('Number of grasp poses: ', grasp_poses.shape[0])
#######  ---------------------------// ---------------------------------- #####
# change grasp [k]
# pose = coord_to_transform(grasp_poses[0])
k = grasp_poses[0]
pose = coord_to_transform(k)
approach = approach_coord_to_transform(k)

#######  ---------------------------// ---------------------------------- #####

print('Pose')
print(pose)

my_pose = Pose()
my_pose.position.x = pose[0]
my_pose.position.y = pose[1]
my_pose.position.z = pose[2]
my_pose.orientation.x = pose[3]
my_pose.orientation.y = pose[4]
my_pose.orientation.z = pose[5]
my_pose.orientation.w = pose[6]

# Transform the pose from the camera frame to the base frame (world)
transformed_pose = transform_pose(my_pose, "camera_depth_frame", "base_link")

print(transformed_pose)

final_pose = np.array([transformed_pose.position.x,
                       transformed_pose.position.y,
                       transformed_pose.position.z,
                       transformed_pose.orientation.x,
                       transformed_pose.orientation.y,
                       transformed_pose.orientation.z,
                       transformed_pose.orientation.w])

# Publish the transform as goal_frame topic for visualization in RViz
publish_tf(final_pose[0], final_pose[1], final_pose[2], final_pose[3], final_pose[4], final_pose[5], final_pose[6],
           child_frame='partial_goal_frame')

np.save(tmp_dir + 'final_pose.npy', final_pose)

print()
print('Complete grasps')
subprocess.call(
    './detect_grasps ../cfg/eigen_params_complete.cfg /home/nuno/Documents/kinova_grasping/tmp_data/complete_pc.pcd',
    shell=True, cwd='/home/nuno/Documents/third_party/gpd/build')

### Take the first grasp proposal from GPD
grasp_poses = read_grasps()
print('Number of grasp poses: ', grasp_poses.shape[0])
#######  ---------------------------// ---------------------------------- #####
# change grasp [k]
# pose = coord_to_transform(grasp_poses[0])
k = grasp_poses[0]
pose = coord_to_transform(k)
approach = approach_coord_to_transform(k)

#######  ---------------------------// ---------------------------------- #####

print('Pose')
print(pose)

# rospy.init_node('my_static_tf2_broadcaster')

my_pose = Pose()
my_pose.position.x = pose[0]
my_pose.position.y = pose[1]
my_pose.position.z = pose[2]
my_pose.orientation.x = pose[3]
my_pose.orientation.y = pose[4]
my_pose.orientation.z = pose[5]
my_pose.orientation.w = pose[6]

# Transform the pose from the camera frame to the base frame (world)
transformed_pose = transform_pose(my_pose, "camera_depth_frame", "base_link")

print(transformed_pose)

final_pose = np.array([transformed_pose.position.x,
                       transformed_pose.position.y,
                       transformed_pose.position.z,
                       transformed_pose.orientation.x,
                       transformed_pose.orientation.y,
                       transformed_pose.orientation.z,
                       transformed_pose.orientation.w])

# Publish the transform as goal_frame topic for visualization in RViz
publish_tf(final_pose[0], final_pose[1], final_pose[2], final_pose[3], final_pose[4], final_pose[5], final_pose[6],
           child_frame='complete_goal_frame')

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
