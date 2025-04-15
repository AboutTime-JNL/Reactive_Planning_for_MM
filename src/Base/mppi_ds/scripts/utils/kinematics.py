import numpy as np
import torch


class MobileBaseKinematics:
    def __init__(self, device='cpu'):
        self.link_names = ['virtual0', 'virtual1', 'mobile_base']
        self.map_link2idx = {}
        for i, name in enumerate(self.link_names):
            self.map_link2idx[name] = i

        self.device = device

        self.dof = 2
        self.trans_list_batch = []

        self.qmin = np.asarray([999, 999])
        self.qmax = np.asarray([999, 999])

        self.dq_max = np.asarray([0.1, 0.1]) * 10.
        self.dq_min = np.asarray([0.1, 0.1]) * 10.

        self.ee_link = 'mobile_base'

        self.sphere_info = {
            "mobile_base": np.asarray([[0.0, 0, 0.145, 0.40]]),
            # "mobile_base": np.asarray([[0.0, 0, 0, 0.]]),
        }

    def forward_kinematics_batch(self, q):
        """
        **Input**

        - q: (b, dof)
        """
        self.trans_list_batch = []
        current_trans = torch.eye(4).to(self.device)
        current_trans = current_trans[None, :, :].repeat(q.shape[0], 1, 1)
        self.trans_list_batch.append(current_trans)

        current_trans[:, 0, 3] = q[:, 0]
        self.trans_list_batch.append(current_trans)
        current_trans[:, 1, 3] = q[:, 1]
        self.trans_list_batch.append(current_trans)

        return self.trans_list_batch

    def jacobian_batch(self, link_name, pts=None):
        """
        **Note**

        - forward_kinematics_batch should be called firstly

        Args:
            link_name:
            pts: torch (B, 3), in base coordinate, P_base

        Returns:
            lin_jac: torch (B, 3, Dof)
            ang_jac: torch (B, 3, Dof)
        """
        c_idx = self.map_link2idx[link_name]

        if pts is None:
            e_pose = self.get_joint_pose_idx(c_idx, batch=True)
            p_e = e_pose[:, :3, 3]  # (B, 3)
        else:
            p_e = pts

        B = p_e.shape[0]

        lin_jac, ang_jac = torch.zeros([B, 3, self.dof]).to(self.device), torch.zeros([B, 3, self.dof]).to(self.device)
        lin_jac[:, 0, 0] = 1
        lin_jac[:, 1, 1] = 1

        jac = torch.cat([lin_jac, ang_jac], dim=1)

        return jac

    def get_joint_pose(self, link_name):
        idx = self.map_link2idx[link_name]
        return self.get_joint_pose_idx(idx)

    def get_joint_pose_idx(self, idx):
        if len(self.trans_list_batch):
            return self.trans_list_batch[idx]
        else:
            print('warning: forward kinematics should be called')
            return torch.eye(4)


class PlanarKinematics:
    def __init__(self, device='cpu'):
        self.link_names = ['link0', 'link1']
        self.map_link2idx = {}
        for i, name in enumerate(self.link_names):
            self.map_link2idx[name] = i

        self.device = device

        self.dof = 2
        self.trans_list_batch = []

        self.qmin = np.asarray([-np.pi * 2., -np.pi * 0.98])
        self.qmax = np.asarray([np.pi * 2., np.pi * 0.98])

        self.dq_max = np.asarray([0.1, 0.1])
        self.dq_min = np.asarray([0.1, 0.1])

        self.ee_link = 'link1'

        self.link_length = np.asarray([0.1, 0.1])

        sphere = np.zeros([50, 4])
        sphere[:, 0] = np.linspace(0, 0.1, 50)
        sphere[:, 3] = 0.1

        self.sphere_info = {
            "link0": sphere,
            "link1": sphere,
        }

    def forward_kinematics_batch(self, q):
        """
        **Input**

        - q: (b, dof)
        """
        self.trans_list_batch = []
        current_trans = torch.eye(4).to(self.device)
        current_trans = current_trans[None, :, :].repeat(q.shape[0], 1, 1)
        self.trans_list_batch.append(current_trans)

        current_trans[:, 0, 3] = torch.cos(q[:, 0]) * self.link_length[0]
        current_trans[:, 1, 3] = torch.sin(q[:, 0]) * self.link_length[0]
        # TODO: 旋转矩阵
        self.trans_list_batch.append(current_trans)
        self.trans_list_batch.append(current_trans)

        return self.trans_list_batch

    def jacobian_batch(self, link_name, pts=None):
        """
        **Note**

        - forward_kinematics_batch should be called firstly

        Args:
            link_name:
            pts: torch (B, 3), in base coordinate, P_base

        Returns:
            lin_jac: torch (B, 3, Dof)
            ang_jac: torch (B, 3, Dof)
        """
        c_idx = self.map_link2idx[link_name]

        if pts is None:
            e_pose = self.get_joint_pose_idx(c_idx, batch=True)
            p_e = e_pose[:, :3, 3]  # (B, 3)
        else:
            p_e = pts

        B = p_e.shape[0]

        lin_jac, ang_jac = torch.zeros([B, 3, self.dof]).to(self.device), torch.zeros([B, 3, self.dof]).to(self.device)
        lin_jac[:, 0, 0] = 1
        lin_jac[:, 1, 1] = 1

        jac = torch.cat([lin_jac, ang_jac], dim=1)

        return jac

    def get_joint_pose(self, link_name):
        idx = self.map_link2idx[link_name]
        return self.get_joint_pose_idx(idx)

    def get_joint_pose_idx(self, idx):
        if len(self.trans_list_batch):
            return self.trans_list_batch[idx]
        else:
            print('warning: forward kinematics should be called')
            return torch.eye(4)
