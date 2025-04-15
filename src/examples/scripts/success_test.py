import threading

import time as ti
import torch
import numpy as np
import open3d as o3d

import utils.open3d_ros_helper as orh
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2

from utils.kinematics import MMKinematics


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


def save_to_txt(filename, data, append=False):
    mode = 'a' if append else 'w'  # 选择追加模式或写模式
    with open(filename, mode) as file:
        # 将6维numpy数组展平成一维，并转为字符串写入文件
        data_str = ' '.join(map(str, data.flatten()))
        file.write(data_str + '\n')


if __name__ == "__main__":
    rospy.init_node("success_test", anonymous=True)

    perception_radius = 3.0
    time_limit = 300.0
    num_limit = 100

    time = 0.0
    success = 0
    total = 0

    # pcd_source = "/app/src/examples/assets/pcd/6_classroom.pcd"
    # q_start_file = '/app/data/scene/start6.txt'
    # q_aim_file = "/app/data/scene/aim6.txt"
    # ee_file = "ee6.txt"
    # q_file = 'data6.txt'
    # time_file = 'time6.txt'

    pcd_source = "/app/src/examples/assets/pcd/7_gym.pcd"
    q_start_file = '/app/data/scene/start7.txt'
    q_aim_file = "/app/data/scene/aim7.txt"
    ee_file = "ee7.txt"
    q_file = 'data7.txt'
    time_file = 'time7.txt'

    # pcd_source = "/app/src/examples/assets/pcd/1_trees.pcd"
    # q_start_file = '/app/data/scene/start1.txt'
    # q_aim_file = "/app/data/scene/aim1.txt"
    # ee_file = "ee1.txt"
    # q_file = 'data1.txt'
    # time_file = 'time1.txt'

    # pcd_source = "/app/src/examples/assets/pcd/5_studio.pcd"
    # q_start_file = '/app/data/scene/start5.txt'
    # q_aim_file = "/app/data/scene/aim5.txt"
    # ee_file = "ee5.txt"
    # q_file = 'data5.txt'
    # time_file = 'time5.txt'

    # pcd_source = "/app/src/examples/assets/pcd/8_villiage_mc.pcd"
    # q_start_file = '/app/data/scene/start8.txt'
    # q_aim_file = "/app/data/scene/aim8.txt"
    # ee_file = "ee8.txt"
    # q_file = 'data8.txt'
    # time_file = 'time8.txt'

    # pcd_source = "/app/src/examples/assets/pcd/2_block.pcd"
    # q_start_file = '/app/data/scene/start2.txt'
    # q_aim_file = "/app/data/scene/aim2.txt"
    # ee_file = "ee2.txt"
    # q_file = 'data2.txt'
    # time_file = 'time2.txt'

    # pcd_source = "/app/src/examples/assets/pcd/3_corridor_mc.pcd"
    # q_start_file = '/app/data/scene/start3.txt'
    # q_aim_file = "/app/data/scene/aim3.txt"
    # ee_file = "ee3.txt"
    # q_file = 'data3.txt'
    # time_file = 'time3.txt'

    # pcd_source = "/app/src/examples/assets/pcd/4_warehouse.pcd"
    # q_start_file = '/app/data/scene/start4.txt'
    # q_aim_file = "/app/data/scene/aim4.txt"
    # ee_file = "ee4.txt"
    # q_file = 'data4.txt'
    # time_file = 'time4.txt'

    q_start_all = np.loadtxt(q_start_file)
    q_aim_all = np.loadtxt(q_aim_file)

    kin = MMKinematics()

    aim_joint_pub, aim_joint_msg = init_aim_joint_pub()
    local_points_pub, pcd, local_trans_pub, trans, global_vis_pub, global_map_pcd = init_sensor_pub(pcd_source)
    kdtree_map = o3d.geometry.KDTreeFlann(pcd)

    rospy.sleep(1)

    global_map_pcd = orh.o3dpc_to_rospc(global_map_pcd, "world", rospy.Time.now())
    global_vis_pub.publish(global_map_pcd)

    q_start = q_start_all[total]
    q_aim = q_aim_all[total]
    aim_joint_msg.header.stamp = rospy.Time.now()
    aim_joint_msg.position = np.concatenate((q_aim, q_start))
    aim_joint_pub.publish(aim_joint_msg)
    kin.forward_kinematics_batch(torch.tensor(q_aim).unsqueeze(0))
    ee_aim = kin.trans_batch[:, -1, :3, 3].squeeze()

    q = torch.zeros(9)
    subscribe_thread = threading.Thread(target=subscriber_thread)
    subscribe_thread.start()

    rate = rospy.Rate(100)

    while not rospy.is_shutdown() and total < num_limit:
        # change aim joint
        kin.forward_kinematics_batch(q.unsqueeze(0))
        ee = kin.trans_batch[:, -1, :3, 3].squeeze()
        if torch.norm(ee_aim - ee) < 0.11 or time > time_limit:
            save_to_txt(q_file, q.numpy(), append=True)
            save_to_txt(time_file, np.array([time]), append=True)
            save_to_txt(ee_file, ee.numpy(), append=True)
            save_to_txt(ee_file, ee_aim.numpy(), append=True)

            total += 1
            if time <= time_limit:
                success += 1
                print("Reached aim joint: ", q_aim)
            else:
                print("Failed to reach aim joint: ", q_aim)
            print("Success rate: ", success / total)
            time = 0.0

            if total <= num_limit - 1:
                q_start = q_start_all[total]
                q_aim = q_aim_all[total]
                aim_joint_msg.header.stamp = rospy.Time.now()
                aim_joint_msg.position = np.concatenate((q_aim, q_start))
                aim_joint_pub.publish(aim_joint_msg)
                kin.forward_kinematics_batch(torch.tensor(q_aim).unsqueeze(0))
                ee_aim = kin.trans_batch[:, -1, :3, 3].squeeze()

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

        time += 1. / 100.

        rate.sleep()

    print("Success rate: ", success / total)
