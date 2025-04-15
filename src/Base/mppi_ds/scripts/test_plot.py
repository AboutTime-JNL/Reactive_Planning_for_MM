from utils.mppi import MPPI
from utils.rollout import Rollout
from utils.collision import SphereNNModel
from utils.kinematics import MobileBaseKinematics
from utils.A_star import A_star


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

        self.rollout = Rollout(q_f, self.nn_model, params, dt=1.0)

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

obs[:, 0] += 1 * 0.01

fig, ax = plt.subplots()
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
trajectory_astar, = ax.plot([], [], color='blue', linewidth=4, alpha=1.0)
trajectory_mppi, = ax.plot([], [], color='red', linewidth=4, alpha=1.0)

path_astar = None
path_mppi = None
path_lock = threading.Lock()

mppi_ds = MPPI_DS(q_0, q_f, params)
mppi_ds.update_state(obs)
astar = A_star(q_0, q_f, obs)


def path_planning_thread():
    global path_astar, path_mppi, mppi_ds, astar, path_lock
    while True:
        mppi_ds.mppi_optimize()
        new_path = astar.plan()
        with path_lock:
            path_astar = new_path
            path_mppi = mppi_ds.path_optim
        time.sleep(0.1)


def update(frame):
    global path_astar, path_mppi, path_lock

    with path_lock:
        if path_astar is not None:
            path_ = torch.stack(path_astar)
            trajectory_astar.set_data(path_[:, 0], path_[:, 1])
        if path_mppi is not None:
            trajectory_mppi.set_data(path_mppi[:, 0], path_mppi[:, 1])
    return [trajectory_astar, trajectory_mppi]


path_thread = threading.Thread(target=path_planning_thread)
path_thread.daemon = True
path_thread.start()

# 创建动画
ani = FuncAnimation(fig, update, frames=100, interval=20, blit=True)

plt.show()
