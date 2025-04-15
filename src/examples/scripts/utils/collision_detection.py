import torch
import numpy as np


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

        link_indices_pairs = [(i, j) for i in range(self.num_link - 2) for j in range(i + 2, self.num_link)]
        self.self_collision_link_pairs = torch.as_tensor(
            link_indices_pairs, device=self.device, dtype=torch.int64)  # (num_pairs, 2)

        self.link_spheres_trans = None
        self.link_spheres_jacob = None
        self.link_spheres_hessi = None

        self.hessian = False

    def forward_spheres_batch(self, q: torch.Tensor):
        """
        Args:
            q: torch (B, Dof)
        Returns:
            self.link_spheres_trans: torch (B, num_link, m_sphere, 4), :3 pos, 1 radius,
              B组关节构型下的机械臂球的拟合，k为连杆数，max_m为所有连杆中最大的球数，前3个维度表示球心，最后一个维度表示球的半径。
        """
        B = q.shape[0]
        trans_batch = self.kin.get_joint_pose_batch(self.link_names)  # (B, num_link, 4, 4)

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

    def forward_jacobian_batch(self, q: torch.Tensor):
        """
        Args:
            q: torch (B, Dof)
        Returns:
            self.link_spheres_jacob: torch (B, num_link, m, 6, Dof)
            self.link_spheres_hessi: torch (B, num_link, m, 6, Dof, Dof)
        """
        B, _ = q.shape
        num_dof = self.kin.dof
        num_link = self.num_link
        num_sphere = self.link_spheres_trans.shape[2]

        # 连杆编号映射为连杆名
        link_dicts = self.link_names.repeat(num_sphere).reshape(1, -1)  # (1, num_link * num_sphere)
        link_names = link_dicts.repeat(B, 0)  # (B, num_link * num_sphere)

        # 求所有位置的雅可比矩阵
        link_spheres = self.link_spheres_trans.view(B, num_link * num_sphere, 4)  # (B, num_link*num_sphere, 4)
        jacobian = self.kin.jacobian_batch(link_names, link_spheres[:, :, :3])  # (B, num_link*num_sphere, 6, num_dof)
        self.link_spheres_jacob = jacobian.view(B, num_link, num_sphere, 6, num_dof)

        if self.hessian:
            # 求所有位置的海森矩阵
            jac = jacobian.view(B * self.num_link * num_sphere, 6, num_dof)  # (B*num_link*num_sphere, 6, num_dof)
            hessian = self.kin.hessian_batch(jac)  # (B*num_link*num_sphere, 6, num_dof, num_dof)
            self.link_spheres_hessi = hessian.view(B, num_link, num_sphere, 6, num_dof, num_dof)

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

    def get_jacobian_batch(
            self, closest_link_sphere_indices, closest_obs_indices
    ):
        """
        Notes:
            The input is the output of get_distances_batch
        Args:
            closest_link_sphere_indices: torch.Tensor(B, k), Indices of the ball closest to the obstacle on links
            closest_obs_indices: torch.Tensor(B, k), Indices of the ball closest to the links on obstacle
        Returns:
            jacobian_dq_f: torch(B, num_link, num_dof)
            hessian_dq_f: torch(B, num_link, num_dof, num_dof)
        """
        B, num_link = closest_link_sphere_indices.shape
        num_dof = self.kin.dof

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

        if self.hessian:
            # 计算海森矩阵
            # 求所有位置的海森矩阵
            link_indices = closest_link_sphere_indices[:, :, None, None, None, None].\
                repeat(1, 1, 1, 6, num_dof, num_dof)  # (B, num_link, 1, 6, num_dof,num_dof)
            hessian = torch.gather(self.link_spheres_hessi, dim=2, index=link_indices).\
                squeeze(2)  # (B, num_link, 6, num_dof, num_dof)
            lin_hessian = hessian[:, :, :3, :, :]  # (B, num_link, 3, num_dof, num_dof)

            part_1 = -torch.matmul(jacobian_dq_f[:, :, :].unsqueeze(-1),
                                    jacobian_dq_f[:, :, :].unsqueeze(2)) \
                                        / normal_pos_norm[:, :, None, None]  # (B, num_link, num_dof, num_dof)
            part_2 = torch.matmul(torch.transpose(lin_jac, 2, 3), lin_jac
                                  ) / normal_pos_norm[:, :, None, None]  # (B, num_link, num_dof, num_dof)
            part_3 = (normal_pos[:, :, :, None, None] * lin_hessian).sum(dim=2) / \
                        normal_pos_norm[:, :, None, None]  # (B, num_link, num_dof, num_dof)

            hessian_dq_f = part_1 + part_2 + part_3
        else:
            hessian_dq_f = None

        return jacobian_dq_f, hessian_dq_f

    def get_self_distances_batch(self, q):
        """
        Notes:
            self.kin.forward_kinematics_batch(q) should be called first
        Args:
            q: torch (B, DOF)
        Returns:
            min_distances: torch.Tensor (B, (k-1)*(k-2)/2), the minimal distance between links
            closest_link_sphere_indices: torch.Tensor (B, (k-1)*(k-2)/2, 2), Indices of the ball closest to the other balls of the links
        """
        batch_size = q.shape[0]
        num_pairs = self.self_collision_link_pairs.shape[0]  # 连杆对数

        # 用于存储结果
        centers = self.link_spheres_trans[:, :, :, :3]  # (batch, k, m, 3)
        radii = self.link_spheres_trans[:, :, :, 3:]  # (batch, k, m, 1)

        # Gather pairs
        link1_centers = centers[:, self.self_collision_link_pairs[:, 0]]  # (batch, num_pairs, m, 3)
        link2_centers = centers[:, self.self_collision_link_pairs[:, 1]]  # (batch, num_pairs, m, 3)
        link1_radii = radii[:, self.self_collision_link_pairs[:, 0]]      # (batch, num_pairs, m, 1)
        link2_radii = radii[:, self.self_collision_link_pairs[:, 1]]      # (batch, num_pairs, m, 1)

        # Broadcast for distance computation
        diff = link1_centers.unsqueeze(-2) - link2_centers.unsqueeze(-3)  # (batch, num_pairs, m, m, 3)
        dists = torch.norm(diff, dim=-1)  # (batch, num_pairs, m, m)
        adjusted_dists = dists -\
            (link1_radii.unsqueeze(-2) + link2_radii.unsqueeze(-3)).squeeze(-1)  # (batch, num_pairs, m, m)

        # Find minimum distances
        min_dists, min_indices = torch.min(adjusted_dists.view(batch_size, num_pairs, -1), dim=-1)  # (batch, num_pairs)

        # Compute corresponding indices in m1 and m2
        m = link2_centers.shape[-2]  # Number of spheres in link2
        row_indices = min_indices // m  # Indices in link1
        col_indices = min_indices % m  # Indices in link2

        # Combine closest positions
        closest_link_sphere_indices = torch.stack([row_indices, col_indices], dim=-1)  # (batch, num_pairs, 2)

        return min_dists, closest_link_sphere_indices

    def get_self_jacobian_batch(self, closest_link_sphere_indices):
        """
        Notes:
            The input is the output of get_self_distances_batch
        Args:
            closest_link_sphere_indices: torch.Tensor (B, (k-1)*(k-2)/2, 2), Indices of the ball closest to the other balls of the links
        Returns:
            jacobian_dq_f: torch (B, (k-1)*(k-2)/2, num_dof)
            hessian_dq_f: torch (B, (k-1)*(k-2)/2, num_dof, num_dof)
        """
        B, _, _ = closest_link_sphere_indices.shape

        link_indices = self.self_collision_link_pairs.unsqueeze(0).repeat(B, 1, 1)  # (B, num_pairs, 2)
        shpere_indices = closest_link_sphere_indices
        batch_indices = torch.arange(B).view(B, 1, 1).expand_as(link_indices)

        # 计算雅可比矩阵
        # 求所有位置的雅可比矩阵
        jacobian = self.link_spheres_jacob[batch_indices, link_indices, shpere_indices]  # (B, num_pairs, 2, 6, num_dof)
        lin_jac = jacobian[:, :, :, :3, :]  # (B, num_pairs, 2, 3, num_dof)
        delta_jac = lin_jac[:, :, 0, :, :] - lin_jac[:, :, 1, :, :]  # (B, num_pairs, 3, num_dof)

        spheres = self.link_spheres_trans[batch_indices, link_indices, shpere_indices]  # (B, num_pairs, 2, 4)

        normal_pos = spheres[:, :, 0, :3] - spheres[:, :, 1, :3]  # (B, num_pairs, 3)
        normal_pos_norm = torch.norm(normal_pos, dim=-1)  # (B, num_pairs)

        jacobian_dq_f = torch.matmul(normal_pos[:, :, None, :], delta_jac).squeeze(
            2) / normal_pos_norm[:, :, None]  # (B, num_pairs, num_dof)

        if self.hessian:
            # 计算海森矩阵
            # 求所有位置的海森矩阵
            hessian = self.link_spheres_hessi[batch_indices, link_indices,
                                              shpere_indices]  # (B, num_pairs, 2, 6, num_dof, num_dof)
            lin_hessian = hessian[:, :, :, :3, :]   # (B, num_pairs, 2, 3, num_dof, num_dof)
            delta_hessian = lin_hessian[:, :, 0, :, :, :] - \
                lin_hessian[:, :, 1, :, :, :]  # (B, num_pairs, 3, num_dof, num_dof)

            part_1 = -torch.matmul(jacobian_dq_f[:, :, :].unsqueeze(-1),
                                    jacobian_dq_f[:, :, :].unsqueeze(2)) \
                                        / normal_pos_norm[:, :, None, None]  # (B, num_pairs, num_dof, num_dof)
            part_2 = torch.matmul(torch.transpose(delta_jac, 2, 3), delta_jac
                                  ) / normal_pos_norm[:, :, None, None]  # (B, num_pairs, num_dof, num_dof)
            part_3 = (normal_pos[:, :, :, None, None] * delta_hessian).sum(dim=2) / \
                        normal_pos_norm[:, :, None, None]  # (B, num_pairs, num_dof, num_dof)

            hessian_dq_f = part_1 + part_2 + part_3
        else:
            hessian_dq_f = None

        return jacobian_dq_f, hessian_dq_f


# 示例用法：
if __name__ == "__main__":
    # # 平面碰撞检查示例：初始化类并测试查询功能（暂时注释掉具体调用以避免错误）
    # image_path = "/app/src/scripts/collision_maps/L-obstacle_2.png"  # 示例 PNG 图片路径
    # calculator = PlanerDistanceFieldCalculator(image_path, [[-5, 5], [-5, 5]])

    # # 查询距离和最近障碍物位置
    # x_real, y_real = -5.0, -4.0  # 示例查询点（米）
    # distance, nearest_obstacle_pos = calculator.query_nearest_obstacle(x_real, y_real)
    # print(f"最近障碍物距离: {distance:.2f} 米, 最近障碍物位置: {nearest_obstacle_pos}")
    pass
