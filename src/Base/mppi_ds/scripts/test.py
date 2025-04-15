from utils.prm import PRM
from utils.mppi import MPPI
from utils.rollout import Rollout
from utils.collision import SphereNNModel
from utils.kinematics import MobileBaseKinematics


from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib
import threading
import time
matplotlib.use('TkAgg')


class MPPI_DS:
    def __init__(self, q_0, q_f, params):
        self.params = params

        # Integration parameters
        self.dt_H = 5  # horizon length in exploration sampling

        self.kin = MobileBaseKinematics(params["device"])
        self.kin.sphere_info = {
            "mobile_base": np.asarray([[0.0, 0, 0, 0.001]]),
        }

        self.nn_model = SphereNNModel(self.kin)

        self.rollout = Rollout(q_f, self.nn_model, params, dt=0.8)

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

        self.curr_state = q_0.clone()

        self.path_optim = None

    def update_state(self, global_obs):
        self.nn_model.obs = global_obs

    def mppi_optimize(self):
        self.u = self.mppi_gym.command(self.curr_state)
        self.path_optim = self.mppi_gym.states[-1, :, :]


params = {"device": "cpu", "dtype": torch.float32}

q_0 = torch.tensor([0, 3], **params)
q_f = torch.tensor([0, 7], **params)
obs = torch.tensor([[0, 5, 0, 0.5]], **params)
radius = 5
for i in range(15):
    obs = torch.cat((obs, torch.tensor([[radius * torch.sin(torch.tensor(i / 180 * torch.pi)),
                    radius * torch.cos(torch.tensor(i / 180 * torch.pi)), 0, 0.5]], **params)), dim=0)
    obs = torch.cat((obs, torch.tensor([[-radius * torch.sin(torch.tensor(i / 180 * torch.pi)),
                    radius * torch.cos(torch.tensor(i / 180 * torch.pi)), 0, 0.5]], **params)), dim=0)

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-3, 3)
ax.set_ylim(2.5, 7.5)
ax.set_aspect('equal')
ax.scatter(q_0[0], q_0[1], color='green', s=100)
ax.scatter(q_f[0], q_f[1], color='red', s=100)

# 障碍物
circles = []
for ob in obs:
    circle = plt.Circle(ob[:2], ob[3], color='r')
    ax.add_artist(circle)
    circles.append(circle)
trajectory, = ax.plot([], [], color='red', linewidth=2, alpha=0.5)  # 实时轨迹

direction = 1
path = None
time_record = []
obs_lock = threading.Lock()
path_lock = threading.Lock()

mppi_ds = MPPI_DS(q_0, q_f, params)


def path_planning_thread():
    global path, obs, mppi_ds, time_record
    while True:
        with obs_lock:
            mppi_ds.update_state(obs)
        start = time.time()
        mppi_ds.mppi_optimize()
        time_record.append(time.time() - start)
        print("Time: ", time.time() - start)
        with path_lock:
            path = mppi_ds.path_optim
        time.sleep(0.1)


def update(frame):
    global direction, obs, path

    if obs[0, 0].abs() > 1:
        direction = -direction
    with obs_lock:
        obs[:, 0] += direction * 0.05
        for ob, circle in zip(obs, circles):
            circle.center = ob[:2]

    with path_lock:
        if path is not None:
            trajectory.set_data(path[:, 0], path[:, 1])

    return circles + [trajectory]


path_thread = threading.Thread(target=path_planning_thread)
path_thread.daemon = True
path_thread.start()

# 创建动画
ani = FuncAnimation(fig, update, frames=100, interval=20, blit=True)

plt.show()

with open('time_mppi.txt', 'a') as file:
    # 将6维numpy数组展平成一维，并转为字符串写入文件
    data_str = ' '.join(map(str, time_record))
    file.write(data_str + '\n')
