import threading

import numpy as np
import rospy
import torch
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


from utils.kinematics import MobileBaseKinematics
from utils.collision import SphereNNModel
from utils.rollout import Rollout
from utils.mppi import MPPI
from utils.prm import PRM
from utils.open3d_ros_helper import rospc_to_numpy


class MPPI_DS:
    def __init__(self, q, q_f, params):
        self.params = params

        self.dt_H = 30  # horizon length in exploration sampling

        self.path_pub = rospy.Publisher("/base_path", Float64MultiArray, queue_size=1)
        self.post_path_pub = rospy.Publisher('/post_path', MarkerArray, queue_size=10)
        self.post_path = Marker()
        self.post_path.header.frame_id = "world"  # 坐标系
        self.post_path.header.stamp = rospy.Time.now()
        self.post_path.ns = "post_path"
        self.post_path.id = 0
        self.post_path.type = Marker.LINE_STRIP  # 标记类型为线段
        self.post_path.action = Marker.ADD
        self.post_path.pose.orientation.w = 1.0
        self.post_path.scale.x = 0.08  # 线的宽度
        self.post_path.color.r = 1.0  # 线的颜色为红色
        self.post_path.color.a = 1.0  # 线的透明度

        self.origin_path_pub = rospy.Publisher('/origin_path', MarkerArray, queue_size=10)
        self.origin_path = Marker()
        self.origin_path.header.frame_id = "world"  # 坐标系
        self.origin_path.header.stamp = rospy.Time.now()
        self.origin_path.ns = "origin_path"
        self.origin_path.id = 1
        self.origin_path.type = Marker.LINE_STRIP  # 标记类型为线段
        self.origin_path.action = Marker.ADD
        self.origin_path.pose.orientation.w = 1.0
        self.origin_path.scale.x = 0.05  # 线的宽度
        self.origin_path.color.b = 1.0  # 线的颜色为蓝色
        self.origin_path.color.a = 1.0  # 线的透明度

        self.kin = MobileBaseKinematics(params["device"])

        self.nn_model = SphereNNModel(self.kin)

        self.rollout = Rollout(q_f, self.nn_model, params, dt=0.5)

        self.mppi_gym = MPPI(
            dynamics=self.rollout.dynamics,
            running_cost=self.rollout.running_cost,
            nx=self.kin.dof,
            noise_sigma=torch.tensor([[10]], **params),
            u_min=torch.tensor([-2.0], **params),
            num_samples=100,
            horizon=self.dt_H,
            device=params["device"],
            terminal_state_cost=self.rollout.terminal_state_cost,
            sample_null_action=True,
        )
        self.prm = PRM(self.dt_H + 2, self.kin, self.nn_model, device=params["device"], dtype=params["dtype"])

        self.q_curr = q.clone()
        self.path_optim = None
        self.path_cout = None
        self.cost_cout = None

    def update_state(self, q, q_f, global_obs):
        if (self.rollout.qf - q_f).norm() > 0.1 or (self.q_curr - q).norm() > 1.0:
            self.path_optim = None
            self.path_cout = None

            print("update aim", q_f)

        # update the current state
        self.q_curr = q.clone()

        # update the goal state
        self.rollout.qf = q_f.clone()

        # select the obstacle point near the base
        squared_distances = (global_obs[:, 0] - self.q_curr[0]) ** 2 + \
            (global_obs[:, 1] - self.q_curr[1]) ** 2
        condition = squared_distances < 25
        obs = global_obs[condition]
        if obs.size(0) > 0:
            self.nn_model.obs = obs

    def mppi_optimize(self):
        _ = self.mppi_gym.command(self.q_curr)
        self.path_optim = self.mppi_gym.states[-1, :, :]

    def send_path(self):
        self.path_optim = torch.cat(
            (self.path_optim, self.rollout.qf.unsqueeze(0)), dim=0
        )
        path_optim, cost = self.prm.prm_planning(self.path_optim)

        if self.path_cout is not None:
            path_cout, cost_cout = self.prm.prm_planning(self.path_cout)
            if cost_cout < cost:
                if (cost_cout - self.cost_cout) < 0.1:
                    return
                self.path_optim = self.path_cout.clone()
                path_optim = path_cout
                cost = cost_cout

        # publish optimal path
        self.path_cout = self.path_optim.clone()
        self.cost_cout = cost
        path = Float64MultiArray()
        path.data = path_optim.flatten().cpu().numpy().tolist()
        path.layout.data_offset = path_optim.size(0)
        self.path_pub.publish(path)
        print("update path")

        self.visualize_path(path_optim)

    def visualize_path(self, path_optim):
        # 可视化
        origin_path = MarkerArray()
        self.origin_path.points.clear()
        for i in range(self.path_cout.shape[0]):
            point = Point()
            point.x = self.path_cout[i][0].item()
            point.y = self.path_cout[i][1].item()
            point.z = 0.0  # 平面路径，z坐标设为0
            self.origin_path.points.append(point)
        origin_path.markers.append(self.origin_path)

        self.origin_path_pub.publish(origin_path)

        post_path = MarkerArray()
        self.post_path.points.clear()
        for i in range(path_optim.shape[0]):
            point = Point()
            point.x = path_optim[i][0].item()
            point.y = path_optim[i][1].item()
            point.z = 0.0  # 平面路径，z坐标设为0
            self.post_path.points.append(point)
        post_path.markers.append(self.post_path)

        self.post_path_pub.publish(post_path)


def update_current_pose(msg: JointState):
    global q
    q[0] = msg.position[0]
    q[1] = msg.position[1]


def update_desired_pose(msg: JointState):
    global q_f
    q_f[0] = msg.position[0]
    q_f[1] = msg.position[1]


def update_obs(msg: PointCloud2):
    global global_obs
    obs_pos, _ = rospc_to_numpy(msg)
    obs_list = np.ones((obs_pos.shape[0], 4)) * 0.1
    obs_list[:, :3] = obs_pos

    global_obs = torch.tensor(obs_list, **params)
    condition = global_obs[:, 2] < 1.0
    global_obs = global_obs[condition]


def subscriber_thread():
    rospy.Subscriber("/aim_joint_states", JointState, update_desired_pose, queue_size=1)
    rospy.Subscriber("/sim_joint_states", JointState, update_current_pose, queue_size=1)
    rospy.Subscriber(
        "/panda_mobile_control/ESDFMap/occ_pc", PointCloud2, update_obs, queue_size=1
    )

    rospy.spin()


if __name__ == "__main__":
    rospy.init_node("mppi_ds")

    params = {"device": "cuda", "dtype": torch.float32}

    q = torch.zeros(2, **params)
    q_f = torch.zeros(2, **params)
    global_obs = torch.zeros((0, 4), **params)

    subscrib_thread = threading.Thread(target=subscriber_thread)
    subscrib_thread.start()

    while global_obs.size(0) < 2 and rospy.is_shutdown() is False:
        pass
    print("finish init, obstacle size is ", global_obs.size(0))

    mppi_ds = MPPI_DS(q, q_f, params)

    rate = rospy.Rate(1)

    while rospy.is_shutdown() is False:
        mppi_ds.update_state(q, q_f, global_obs)
        if mppi_ds.nn_model.obs is not None:
            mppi_ds.mppi_optimize()
            mppi_ds.send_path()

        rate.sleep()
