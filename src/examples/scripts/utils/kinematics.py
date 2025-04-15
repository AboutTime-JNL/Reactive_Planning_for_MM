import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def inv_trans_batch(trans):
    """
    **Input**

    - trans: (n, 4, 4)
    """
    res = torch.eye(4).to(trans.device)
    res = res[None, :, :].repeat(trans.shape[0], 1, 1)
    trans_T = trans[:, :3, :3].transpose(dim0=2, dim1=1)
    res[:, :3, :3] = trans_T
    res[:, :3, 3] = -torch.matmul(trans_T, trans[:, :3, 3].unsqueeze(2)).squeeze(2)
    return res


def transform_mdh_batch(a, alpha, q, d):
    """
    Optimized transform_mdh_batch to handle batch and joint dimensions.

    **Input**
    - a: (1, dof) or (b, dof)
    - alpha: (1, dof) or (b, dof)
    - q: (b, dof)
    - d: (1, dof) or (b, dof)

    **Output**
    - trans: (b, dof, 4, 4)
    """
    b, dof = q.shape
    device = q.device

    c = torch.cos(q)  # (b, dof)
    s = torch.sin(q)  # (b, dof)
    c_alpha = torch.cos(alpha)  # (1, dof) or (b, dof)
    s_alpha = torch.sin(alpha)  # (1, dof) or (b, dof)

    trans = torch.zeros((b, dof, 4, 4), device=device)

    # Fill transformation matrix
    trans[:, :, 0, 0] = c
    trans[:, :, 0, 1] = -s
    trans[:, :, 0, 3] = a
    trans[:, :, 1, 0] = s * c_alpha
    trans[:, :, 1, 1] = c * c_alpha
    trans[:, :, 1, 2] = -s_alpha
    trans[:, :, 1, 3] = -s_alpha * d
    trans[:, :, 2, 0] = s * s_alpha
    trans[:, :, 2, 1] = c * s_alpha
    trans[:, :, 2, 2] = c_alpha
    trans[:, :, 2, 3] = c_alpha * d
    trans[:, :, 3, 3] = 1.0

    return trans


class MMKinematics:
    def __init__(self, dtype=torch.float32, device='cpu'):
        self.dtype = dtype
        self.device = device
        self.tensor_args = {'dtype': self.dtype, 'device': self.device}

        self.link_names = ["base_link", "y_base_link", "x_base_link", "w_base_link", "jaka_base_link",
                           "jaka_shoulder_pan_link", "jaka_shoulder_lift_link", "jaka_elbow_link",
                           "jaka_wrist_1_link", "jaka_wrist_2_link", "jaka_wrist_3_link"]
        self.map_link2idx = {}
        for i, name in enumerate(self.link_names):
            self.map_link2idx[name] = i
        self.num_link = len(self.link_names)

        # link name 索引
        self.link_names = np.array(self.link_names)
        self.sorted_idx = np.argsort(self.link_names)  # 排序后的索引
        self.link_names = self.link_names[self.sorted_idx]  # 排序后的 keys

        self.link_idx = np.array(list(self.map_link2idx.values()))

        # MDH参数
        pi = torch.pi
        self.a = torch.as_tensor([0, 0, 0, 0.1395, 0, 0, 0.36, 0.3035, 0, 0], **self.tensor_args).unsqueeze(0)
        self.alpha = torch.as_tensor([-0.5 * pi, -0.5 * pi, -0.5 * pi, 0, 0, 0.5 * pi, 0,
                                     0, 0.5 * pi, -0.5 * pi], **self.tensor_args).unsqueeze(0)
        self.d_offset = torch.as_tensor([0, 0, 0, 0.4145, 0.12015, 0, 0, -0.1135, 0.1135, 0.107],
                                        **self.tensor_args).unsqueeze(0)
        self.q_offset = torch.as_tensor([-0.5 * pi, -0.5 * pi, -0.5 * pi, 0, 0, 0,
                                        0, 0, 0, 0], **self.tensor_args).unsqueeze(0)
        self.dof = 9
        self.trans_batch = None

        self.qmin = np.asarray([-99, -99, -pi, -pi, -pi, -pi, -pi, -pi, -pi])
        self.qmax = np.asarray([99, 99, pi, pi, pi, pi, pi, pi, pi])

        self.ee_link = 'jaka_wrist_3_link'

        # 碰撞球拟合
        self.sphere_info = {
            'w_base_link': np.asarray([[0., 0., 0.207, 0.35],
                                       [0.1395, 0., 0.535, 0.05]]),
            'jaka_shoulder_pan_link': np.asarray([[0., 0.14415, 0., 0.05],
                                                   [0., 0.14415, 0., 0.]]),
            'jaka_shoulder_lift_link': np.asarray([[0.18, 0., -0.14415, 0.05],
                                                   [0.36, 0., -0.14415, 0.05]]),
            'jaka_elbow_link': np.asarray([[0., 0., 0., 0.05],
                                             [0.15175, 0., 0., 0.05]]),
            'jaka_wrist_1_link': np.asarray([[0., 0., 0., 0.05],
                                             [0., 0., 0.1135, 0.05]]),
            'jaka_wrist_3_link': np.asarray([[0., 0., 0., 0.05],
                                             [0., 0., -0.107, 0.05]])
        }

    def forward_kinematics_batch(self, q: torch.Tensor):
        """
        **Input**

        - q: (b, 9)

        **Output**

        - self.trans_batch: (b, 10, 4, 4)

        """
        q = q.clone().to(self.device)
        batch, _ = q.shape

        q_sum = self.q_offset.clone().expand(batch, -1)
        d_sum = self.d_offset.clone().expand(batch, -1)

        d_sum[:, 1] += q[:, 0]  # x_base_link
        d_sum[:, 0] += q[:, 1]  # y_base_link
        q_sum[:, 2] += q[:, 2]  # w_base_link
        q_sum[:, 4:] += q[:, 3:]  # jaka_link

        # Compute all transformation matrices for joints
        all_transforms = transform_mdh_batch(self.a[:, :], self.alpha[:, :], q_sum, d_sum)  # Result: (b, dof, 4, 4)

        # Add identity matrix at the beginning to represent base frame
        identity = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch, 1, 4, 4)  # (b, 1, 4, 4)
        self.trans_batch = torch.cat([identity, all_transforms], dim=1)  # (b, dof + 2, 4, 4)

        # Cumulatively multiply along the joint axis to compute forward kinematics
        for i in range(self.trans_batch.shape[1] - 1):
            self.trans_batch[:, i + 1] = torch.matmul(self.trans_batch[:, i, :, :], self.trans_batch[:, i + 1, :, :])

        return

    def link2idx(self, link_name: np.ndarray) -> np.ndarray:
        '''_summary_

        Args:
            link_name (np.ndarray): string 构成的 np.ndarray

        Returns:
            np.ndarray: int 构成的 np.ndarray
        '''
        indices = np.searchsorted(self.link_names, link_name)
        indices = self.sorted_idx[indices]

        return self.link_idx[indices]

    def get_joint_pose(self, link_name):
        idx = self.map_link2idx[link_name]
        if self.trans_batch is not None:
            return self.trans_batch[:, idx, :, :]
        else:
            print('warning: forward kinematics should be called')
            return torch.eye(4)

    def get_joint_pose_batch(self, link_name: np.ndarray) -> torch.Tensor:
        '''_summary_

        Args:
            link_name (np.ndarray): (n,)

        Returns:
            torch.Tensor: (B, n, 4, 4)
        '''
        if self.trans_batch is not None:
            batch = self.trans_batch.shape[0]
            link_name = link_name.reshape(1, -1).repeat(batch, 0)
            idx = torch.as_tensor(self.link2idx(link_name), dtype=torch.int64, device=self.device)  # (B,n)
            idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)
            trans_batch = torch.gather(self.trans_batch, 1, idx)
            return trans_batch
        else:
            print('warning: forward kinematics should be called')
            return torch.eye(4)

    def jacobian_batch(self, link_name, p_e=None) -> torch.Tensor:
        """
        **Inputs: **

        - link_name: numpy of link names (B, n)
        - pts: tensor (B, n, 3), in base coordinate, P_base

        **Outputs: **

        - jacobian: (B, n, 6, dof)

        **Note**

        - forward_kinematics should be called firstly
        """
        c_idx = torch.as_tensor(self.link2idx(link_name), dtype=torch.int64, device=self.device)  # (B, n)

        if p_e is None:
            idx = c_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)  # B*n*4*4
            e_pose = torch.gather(self.trans_batch, 1, idx)
            p_e = e_pose[:, :, :3, 3]  # (B, n, 3)

        B, n, _ = p_e.shape

        # Extract positions and z-axes for all joints (batch_size, dof, 3)
        p_i = self.trans_batch[:, 3:, :3, 3].unsqueeze(1)  # (B, 1, dof, 3)
        z_i = self.trans_batch[:, 3:, :3, 2].unsqueeze(1).repeat(1, n, 1, 1)  # (B, n, dof, 3)

        # Compute linear and angular Jacobians for all joints
        diff_p = p_e.unsqueeze(2) - p_i  # (B, n, dof, 3)
        lin_jac = torch.cross(z_i, diff_p, dim=-1)  # (B, n, dof, 3)
        ang_jac = z_i  # (B, n, dof, 3)

        # Mask out inactive joints
        joint_mask = torch.arange(self.num_link, device=self.device).repeat(
            B, n, 1) <= c_idx.unsqueeze(-1)  # (B, n, dof)
        joint_mask = joint_mask.unsqueeze(-1)  # (B, n, dof, 1)
        joint_mask = joint_mask[:, :, 3:, :]  # (B, n, dof, 1)

        # Apply mask to zero out inactive joints
        lin_jac = lin_jac * joint_mask  # (B, n, dof, 3)
        ang_jac = ang_jac * joint_mask  # (B, n, dof, 3)

        jacobian = torch.zeros((B, n, 6, self.dof), **self.tensor_args)
        jacobian[:, :, 0, 0] = 1
        jacobian[:, :, 1, 1] = 1
        jacobian[:, :, :3, 2] = lin_jac[:, :, 0, :]
        jacobian[:, :, :3, 3:] = lin_jac[:, :, 2:, :].transpose(-1, -2)
        jacobian[:, :, 3:, 2] = ang_jac[:, :, 0, :]
        jacobian[:, :, 3:, 3:] = ang_jac[:, :, 2:, :].transpose(-1, -2)

        return jacobian


if __name__ == '__main__':
    panda = MMKinematics()
    q = torch.as_tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]).repeat(4, 1)
    link_name = np.array([['panda_link1'], ['panda_link3'], ['panda_link6'], ['panda_hand']])
    p_e = torch.Tensor([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9]], [[1.0, 1.1, 1.2]]]).to(panda.device)
    panda.forward_kinematics_batch(q)
    J = panda.jacobian_batch(link_name, p_e)
    print(J)
