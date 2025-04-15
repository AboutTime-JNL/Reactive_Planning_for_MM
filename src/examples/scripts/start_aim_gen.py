import torch
import rospy
import numpy as np
import open3d as o3d


from utils.kinematics import MMKinematics
from utils.collision_detection import SphereNNModel


def collision(test_q, kdtree_map, perception_radius, pcd, kinematics, collision_detection):
    search_point = np.array([test_q[0, 0], test_q[0, 1], 0.8])
    [_, idx, _] = kdtree_map.search_radius_vector_3d(search_point, perception_radius + 1)
    local_map = pcd.select_by_index(idx)
    points = torch.as_tensor(np.asarray(local_map.points))
    obs = torch.cat([points, torch.ones(points.shape[0], 1) * 0.2], dim=1)

    if obs.size(0) == 0:
        return False
    kinematics.forward_kinematics_batch(test_q)
    collision_detection.forward_spheres_batch(test_q)
    dist, _, _ = collision_detection.get_distances_batch(test_q, obs)
    self_dist, _ = collision_detection.get_self_distances_batch(test_q)
    dist = torch.cat([dist, self_dist], dim=1)
    dist = torch.min(dist)

    return dist < 0.1


def save_to_txt(filename, data, append=False):
    mode = 'a' if append else 'w'  # 选择追加模式或写模式
    with open(filename, mode) as file:
        # 将6维numpy数组展平成一维，并转为字符串写入文件
        data_str = ' '.join(map(str, data.flatten()))
        file.write(data_str + '\n')


if __name__ == "__main__":
    rospy.init_node("start_aim_gen", anonymous=True)

    perception_radius = 1.0
    num_limit = 100
    total = 0

    kinematics = MMKinematics()
    collision_detection = SphereNNModel(kinematics)

    # pcd_source = "/app/src/examples/assets/pcd/1_trees.pcd"
    # q_start_file = '/app/data/scene/start1.txt'
    # q_aim_file = "/app/data/scene/aim1.txt"
    # q_min = np.asarray([-10, -10, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
    # q_max = np.asarray([10, 10, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    # pcd_source = "/app/src/examples/assets/pcd/2_block.pcd"
    # q_start_file = '/app/data/scene/start2.txt'
    # q_aim_file = "/app/data/scene/aim2.txt"
    # q_min = np.asarray([-7, -7, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
    # q_max = np.asarray([7, 7, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    # pcd_source = "/app/src/examples/assets/pcd/3_corridor_mc.pcd"
    # q_start_file = '/app/data/scene/start3.txt'
    # q_aim_file = "/app/data/scene/aim3.txt"
    # q_min = np.asarray([-10, -8, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
    # q_max = np.asarray([10, 8, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    pcd_source = "/app/src/examples/assets/pcd/4_warehouse.pcd"
    q_start_file = '/app/data/scene/start4.txt'
    q_aim_file = "/app/data/scene/aim4.txt"
    q_min = np.asarray([-4, -9, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
    q_max = np.asarray([4, 9, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    # pcd_source = "/app/src/examples/assets/pcd/5_studio.pcd"
    # q_start_file = '/app/data/scene/start5.txt'
    # q_aim_file = "/app/data/scene/aim5.txt"
    # q_min = np.asarray([-7, -4, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
    # q_max = np.asarray([7, 4, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    # pcd_source = "/app/src/examples/assets/pcd/6_classroom.pcd"
    # q_start_file = '/app/data/scene/start6.txt'
    # q_aim_file = "/app/data/scene/aim6.txt"
    # q_min = np.asarray([-4, -4, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
    # q_max = np.asarray([4, 4, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    # pcd_source = "/app/src/examples/assets/pcd/7_gym.pcd"
    # q_start_file = '/app/data/scene/start7.txt'
    # q_aim_file = "/app/data/scene/aim7.txt"
    # q_min = np.asarray([-1, -4, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
    # q_max = np.asarray([1, 5.5, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    # pcd_source = "/app/src/examples/assets/pcd/8_villiage_mc.pcd"
    # q_start_file = '/app/data/scene/start8.txt'
    # q_aim_file = "/app/data/scene/aim8.txt"
    # q_min = np.asarray([-6, -10, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
    # q_max = np.asarray([6, 10, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    # 裁剪pcd
    pcd = o3d.io.read_point_cloud(pcd_source)
    points = np.asarray(pcd.points)
    points = points[(points[:, 0] > -10) & (points[:, 0] < 10)]
    points = points[(points[:, 1] > -10) & (points[:, 1] < 10)]
    points = points[(points[:, 2] > 0) & (points[:, 2] < 2)]
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree_map = o3d.geometry.KDTreeFlann(pcd)

    while not rospy.is_shutdown() and total < num_limit:
        q_aim = np.random.uniform(q_min, q_max)
        test_q = torch.as_tensor(q_aim).unsqueeze(0)

        if not collision(test_q, kdtree_map, perception_radius, pcd, kinematics, collision_detection):
            q_start = np.random.uniform(q_min, q_max)
            test_q = torch.as_tensor(q_start).unsqueeze(0)
            while not rospy.is_shutdown() and (np.linalg.norm(q_start[:2] - q_aim[:2]) < 2.0 or
                                               collision(test_q, kdtree_map, perception_radius, pcd, kinematics, collision_detection)):
                q_start = np.random.uniform(q_min, q_max)
                test_q = torch.as_tensor(q_start).unsqueeze(0)

            save_to_txt(q_start_file, q_start, append=True)
            save_to_txt(q_aim_file, q_aim, append=True)
            total += 1
