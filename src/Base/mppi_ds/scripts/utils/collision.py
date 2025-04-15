import numpy as np
import torch


class SphereNNModel:
    def __init__(self, kin):
        """
        Args:
            kin: torch_version, PandaKinematics(device=self.tensor_args['device'])
        """
        self.device = kin.device
        self.tensor_args = {"device": self.device, "dtype": torch.float32}

        self.kin = kin
        self.n_dof = self.kin.dof
        self.obs = None

        link_names = self.kin.sphere_info.keys()
        self.robot_sphere_info = {}
        sphere_list = []
        for link_name in link_names:
            link_spheres = self.kin.sphere_info[link_name]  # (m, 4)
            self.robot_sphere_info[link_name] = torch.as_tensor(
                link_spheres, **self.tensor_args
            )
            sphere_list.append(self.robot_sphere_info[link_name])
        self.num_link = len(link_names)
        self.link_names = np.array(list(self.robot_sphere_info.keys()))
        self.sphere = torch.stack(sphere_list, dim=0)  # (num_link, m, 4)

        self.link_spheres_trans = None
        self.link_spheres_jacob = None

    def get_spheres_batch(self, q: torch.Tensor):
        """
        Args:
            q: torch (B, Dof)
        Returns:
            self.link_spheres_trans: torch (B, num_link, m_sphere, 4), :3 pos, 1 radius,
              B组关节构型下的机械臂球的拟合，k为连杆数，max_m为所有连杆中最大的球数，前3个维度表示球心，最后一个维度表示球的半径。
        """
        B = q.shape[0]
        trans_batch = self.kin.get_joint_pose(self.link_names[0]).unsqueeze(1)  # (B, 1, 4, 4)

        rotations = trans_batch[:, :, :3, :3]  # (B, num_link, 3, 3)
        translations = trans_batch[:, :, :3, 3]  # (B, num_link, 3)

        points = self.sphere[:, :, :3]  # (num_link, m, 3)
        radii = self.sphere[:, :, 3:]  # (num_link, m, 1)

        transformed_points = torch.einsum(
            "bknx,kmx->bkmn", rotations, points
        ) + translations.unsqueeze(2)  # (B, num_link, m, 3)

        # 合并球心和半径
        self.link_spheres_trans = torch.cat([transformed_points, radii.unsqueeze(0).
                                             expand(B, -1, -1, -1)], dim=-1)  # (B, num_link, m, 4)

        return

    def get_distances_batch(self, q, obs):
        """
        Calculate the distance between the robot links and multiple spheres in task space.

        Notes:
            self.kin.forward_kinematics_batch(q) should be called first
        Args:
            q: torch(B, DOF)
            obs: torch(n, 4), device == self.device,环境中的障碍物，前3个维度表示球心，最后一个维度表示球的半径。
        Returns:
            min_distances: torch.Tensor(B, k), the minimal distance between links and obs, 机械臂和障碍物之间的最短距离。
            closest_link_sphere_indices: torch.Tensor(B, k), Indices of the ball closest to the obstacle on links,
              机械臂连杆上与障碍物最近的球的索引。
            closest_obs_indices: torch.Tensor(B, k), Indices of the ball closest to the links on obstacle,
              障碍物与机械臂连杆最近的障碍物的索引。
        """
        # TODO: every link must have same spheres
        self.get_spheres_batch(q)
        self.obs = obs.clone().to(self.device)
        obs_pos = self.obs[:, :3]  # 障碍物的球心坐标 (n, 3)
        obs_rad = self.obs[:, 3]  # 障碍物的半径 (n)

        links = self.link_spheres_trans[..., :3]
        links_rad = self.link_spheres_trans[..., 3]  # (B, k, max_m)
        # 计算每个球到每个障碍物球心的距离
        pos_dif = links.unsqueeze(3) - obs_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        dst = (
                torch.norm(pos_dif, dim=4)
                - obs_rad.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                - links_rad.unsqueeze(3)
        )

        # 找到每个连杆上每个球体到所有障碍物的最小距离及其索引
        min_distances_per_sphere, min_indices_per_sphere = torch.min(dst, dim=3)

        # 找到每个连杆上与障碍物最近的球的最小距离及其索引
        min_distances, closest_link_sphere_indices = torch.min(
            min_distances_per_sphere, dim=2
        )

        # 找到每个连杆上与障碍物最近的障碍物的索引
        closest_obs_indices = min_indices_per_sphere[
            torch.arange(min_indices_per_sphere.shape[0]).unsqueeze(1),
            torch.arange(min_indices_per_sphere.shape[1]),
            closest_link_sphere_indices,
        ]
        return min_distances, closest_link_sphere_indices, closest_obs_indices

    def get_gradient_batch(
            self, closest_link_sphere_indices, closest_obs_indices
    ):
        """
        Notes:
            The input is the output of get_distances_batch
        Args:
            closest_link_sphere_indices: torch.Tensor (B, k), Indices of the ball closest to the obstacle on links
            closest_obs_indices: torch.Tensor (B, k), Indices of the ball closest to the links on obstacle
        Returns:
            gradient_dq_f: torch (B, num_dof, num_link)
        """
        B, k = closest_link_sphere_indices.shape

        num_dof = self.kin.dof
        num_link = self.num_link
        num_sphere = self.link_spheres_trans.shape[2]

        # 求所有位置的雅可比矩阵
        link_spheres = self.link_spheres_trans.view(B, num_link * num_sphere, 4)  # (B, num_link*num_sphere, 4)
        # (B, num_link*num_sphere, 6, num_dof)
        jacobian = self.kin.jacobian_batch(self.link_names[0], link_spheres[:, :, :3].squeeze(1)).unsqueeze(1)
        self.link_spheres_jacob = jacobian.view(B, num_link, num_sphere, 6, num_dof)

        # 计算雅可比矩阵
        # 求所有位置的雅可比矩阵
        link_indices = closest_link_sphere_indices[:, :, None, None, None].\
            repeat(1, 1, 1, 6, num_dof)  # (B, num_link, 1, 6, num_dof)
        jacobian = torch.gather(self.link_spheres_jacob, dim=2, index=link_indices).\
            squeeze(2)  # (B, num_link, 6, num_dof)
        lin_jac = jacobian[:, :, :3, :]  # (B, num_link, 3, num_dof)

        link_indices = closest_link_sphere_indices[:, :, None, None].repeat(1, 1, 1, 4)  # (B, num_link, 1, 4)
        link_spheres = torch.gather(self.link_spheres_trans, dim=2, index=link_indices).squeeze(2)  # (B, num_link, 4)
        link_spheres = link_spheres[:, :, :3]  # (B, num_link, 3)

        expand_obs = self.obs[None, None, :, :].repeat(B, num_link, 1, 1)  # (B, num_link, n, 4)
        obs_indices = closest_obs_indices[:, :, None, None].repeat(1, 1, 1, 4)  # (B, num_link, 1, 4)
        obs_spheres = torch.gather(expand_obs, dim=2, index=obs_indices).squeeze(2)  # (B, num_link, 4)
        obs_spheres = obs_spheres[:, :, :3]  # (B, num_link, 3)

        normal_pos = link_spheres - obs_spheres  # (B, num_link, 3)
        normal_pos_norm = torch.norm(normal_pos, dim=-1)  # (B, num_link)

        jacobian_dq_f = torch.matmul(normal_pos[:, :, None, :], lin_jac).squeeze(
            2) / normal_pos_norm[:, :, None]  # (B, num_link, num_dof)

        return jacobian_dq_f.transpose(dim0=-1, dim1=-2)  # (B, num_dof, num_link)
