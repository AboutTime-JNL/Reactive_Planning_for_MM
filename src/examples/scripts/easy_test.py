import threading

import torch
import copy
import numpy as np
import open3d as o3d

import utils.open3d_ros_helper as orh
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2


def update_current_pose(msg: JointState):
    global q
    q[0] = msg.position[0]
    q[1] = msg.position[1]
    q[2] = msg.position[2]
    q[3] = msg.position[3]
    q[4] = msg.position[4]
    q[5] = msg.position[5]
    q[6] = msg.position[6]
    q[7] = msg.position[7]
    q[8] = msg.position[8]


def subscriber_thread():
    rospy.Subscriber("/sim_joint_states", JointState, update_current_pose, queue_size=1)

    rospy.spin()


def init_aim_joint_pub():
    aim_joint_pub = rospy.Publisher("/aim_joint_states", JointState, queue_size=1)
    aim_joint_msg = JointState()
    aim_joint_msg.header.frame_id = "world"
    aim_joint_msg.name = [
        "x_base_joint",
        "y_base_joint",
        "w_base_joint",
        "jaka_shoulder_pan_joint",
        "jaka_shoulder_lift_joint",
        "jaka_elbow_joint",
        "jaka_wrist_1_joint",
        "jaka_wrist_2_joint",
        "jaka_wrist_3_joint",
        "x_base_joint",
        "y_base_joint",
        "w_base_joint",
        "jaka_shoulder_pan_joint",
        "jaka_shoulder_lift_joint",
        "jaka_elbow_joint",
        "jaka_wrist_1_joint",
        "jaka_wrist_2_joint",
        "jaka_wrist_3_joint"
    ]

    return aim_joint_pub, aim_joint_msg


def init_sensor_pub(pcd_source):
    local_points_pub = rospy.Publisher(
        "/camera/depth_registered/points", PointCloud2, queue_size=1
    )
    pcd = o3d.io.read_point_cloud(pcd_source)

    # 裁剪pcd
    points = np.asarray(pcd.points)
    points = points[(points[:, 0] > -10) & (points[:, 0] < 10)]
    points = points[(points[:, 1] > -10) & (points[:, 1] < 10)]
    points = points[(points[:, 2] > 0) & (points[:, 2] < 2)]
    pcd.points = o3d.utility.Vector3dVector(points)

    local_trans_pub = rospy.Publisher(
        "/kinect/vrpn_client/estimated_transform", TransformStamped, queue_size=1
    )
    trans = TransformStamped()
    trans.header.frame_id = "world"
    trans.child_frame_id = "robot"
    trans.transform.translation.x = 0.0
    trans.transform.translation.y = 0.0
    trans.transform.translation.z = 0.0
    trans.transform.rotation.x = 0.0
    trans.transform.rotation.y = 0.0
    trans.transform.rotation.z = 0.0
    trans.transform.rotation.w = 1.0

    global_vis_pub = rospy.Publisher(
        "/global_map", PointCloud2, queue_size=1
    )
    points = np.asarray(pcd.points)
    global_map_pcd = o3d.geometry.PointCloud()
    global_map_pcd.points = o3d.utility.Vector3dVector(points)

    return local_points_pub, pcd, local_trans_pub, trans, global_vis_pub, global_map_pcd


if __name__ == "__main__":
    rospy.init_node("success_test", anonymous=True)

    perception_radius = 3.0

    pcd_source = "/app/src/examples/assets/pcd/shelf2_desk.pcd"
    q_start = np.array([-3, -1, 0.0, 0, 1.3094, -1.1732, -4.9579, -3.0013, -3.2900])
    q_aim = np.array([3, 4.0, 0.0, 0, 1.3094, -1.1732, -4.9579, -3.0013, -3.2900])

    aim_joint_pub, aim_joint_msg = init_aim_joint_pub()
    local_points_pub, pcd, local_trans_pub, trans, global_vis_pub, global_map_pcd = init_sensor_pub(pcd_source)

    kdtree_map = o3d.geometry.KDTreeFlann(pcd)

    rospy.sleep(1)

    global_map_pcd = orh.o3dpc_to_rospc(global_map_pcd, "world", rospy.Time.now())
    global_vis_pub.publish(global_map_pcd)

    aim_joint_msg.header.stamp = rospy.Time.now()
    aim_joint_msg.position = np.concatenate((q_aim, q_start))
    aim_joint_pub.publish(aim_joint_msg)

    q = q_start
    subscribe_thread = threading.Thread(target=subscriber_thread)
    subscribe_thread.start()

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        # Sensor
        search_point = np.array([q[0], q[1], 0.5])

        trans.transform.translation.x = search_point[0]
        trans.transform.translation.y = search_point[1]
        trans.transform.translation.z = 0.8

        [_, idx, _] = kdtree_map.search_radius_vector_3d(search_point, perception_radius)
        if len(idx) > 0:
            local_map = pcd.select_by_index(idx)
            points = np.asarray(local_map.points)

            local_map_pcd = o3d.geometry.PointCloud()
            local_map_pcd.points = o3d.utility.Vector3dVector(points)
            local_map_pcd = orh.o3dpc_to_rospc(local_map_pcd, "world", rospy.Time.now())
            local_points_pub.publish(local_map_pcd)
        else:
            pass

        trans.header.stamp = rospy.Time.now()
        local_trans_pub.publish(trans)

        rate.sleep()
