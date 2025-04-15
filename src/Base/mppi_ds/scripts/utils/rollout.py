import torch

from utils.collision import SphereNNModel


def generalized_sigmoid(x, y_min, y_max, x0, x1, k):
    return y_min + (y_max - y_min) / (1 + torch.exp(k * (-x + (x0 + x1) / 2)))


class Rollout:
    def __init__(self, qf, nn_model: SphereNNModel, params, dt):
        self.tensor_args = params
        self.qf = qf
        self.n_dof = qf.shape[0]

        self.dt = dt

        self.dst_thr = 0.2
        self.nn_model = nn_model
        self.kin = self.nn_model.kin

    def dynamics(self, state, perturbed_action, t):
        """
        Args:
            state: (B, n_dof)
            perturbed_action: (b, nu), nu=n_dof-1
            t:
        Return:
            state: (B, n_dof)
        """
        batch_size = state.shape[0]
        # Constants
        dist_low, dist_high = 0.0, 0.5
        k_sigmoid = 30
        ln_min, ln_max = 0, 1
        ltau_min, ltau_max = 1, 3

        # Compute nominal velocity
        nominal_velocity = self.qf - state
        nominal_velocity_norm = torch.norm(nominal_velocity, dim=-1, keepdim=True)
        nominal_velocity_normalized = nominal_velocity / nominal_velocity_norm

        # Compute kinematics and distances
        self.kin.forward_kinematics_batch(state)
        min_distances, closest_link_sphere_indices, closest_obs_indices = (
            self.nn_model.get_distances_batch(state, self.nn_model.obs)
        )  # (B, num_link), (B, num_dof, num_link)
        gradient_dq_f = self.nn_model.get_gradient_batch(closest_link_sphere_indices, closest_obs_indices)

        # Compute closest distance and gradient
        min_idx = torch.argmin(min_distances, dim=1, keepdim=True)  # (B, 1)
        distance = min_distances.gather(1, min_idx).squeeze(1) - self.dst_thr

        nn_grad = gradient_dq_f.gather(2, min_idx[:, None, :].repeat(1, self.n_dof, 1)).squeeze(2)  # (B, num_dof)
        nn_grad_norm = nn_grad / nn_grad.norm(2, 1, keepdim=True)  # (B, num_dof)

        # QR decomposition and sign adjustment
        basis_eye_temp = torch.eye(self.n_dof, **self.tensor_args).repeat(batch_size, 1, 1)
        basis_eye_temp[:, :, 0] = nn_grad.to(torch.float32)
        E, R = torch.linalg.qr(basis_eye_temp)
        signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))  # (B, N)
        signs = signs * signs[:, 0].unsqueeze(-1)
        E = E * signs.unsqueeze(-2)  # (B, N, N) * (B, 1, N),
        E = E.to(**self.tensor_args)
        E[:, :, 0] = nn_grad_norm

        # Calculate activations
        dotproduct = (nn_grad_norm * nominal_velocity_normalized).sum(dim=-1)
        l_vel = generalized_sigmoid(dotproduct, 0, 1, -0.2, 0.0, 100)
        l_n = generalized_sigmoid(
            distance, ln_min, ln_max, dist_low, dist_high, k_sigmoid
        )

        # Compute activation terms
        collision_activation = 1 - l_n[:, None]
        velocity_activation = 1 - l_vel[:, None]
        goal_activation = (state - self.qf).norm(p=0.5, dim=1).clamp(0, 1).unsqueeze(1)
        goal_activation[goal_activation < 0.3] = 0
        total_activation = collision_activation * velocity_activation * goal_activation

        # Compute policy velocity
        policy_value = torch.matmul(
            E[:, :, 1:], perturbed_action[:, :, None]
        )  # (B, D, D-1) * (B, D-1, 1)=(B, D, 1)
        policy_velocity = (
            policy_value * nominal_velocity_norm[:, None, :]
        )  # magnitude of nominal velocity
        policy_velocity = policy_velocity.squeeze(-1)
        total_velocity = (
            policy_velocity + (1 - total_activation) * nominal_velocity
        )  # magnitude of 2x nominal velocity

        # Compute modulation matrix
        l_n_vel = l_vel * 1 + (1 - l_vel) * l_n
        l_tau = generalized_sigmoid(
            distance, ltau_max, ltau_min, dist_low, dist_high, k_sigmoid
        )
        D = l_tau.repeat_interleave(self.n_dof).reshape((batch_size, self.n_dof)).diag_embed(0, 1, 2)
        D[:, 0, 0] = l_n_vel
        M = E @ D @ E.transpose(1, 2)

        mod_velocity = (M @ total_velocity.unsqueeze(2)).squeeze()
        mod_velocity_norm = torch.norm(mod_velocity, dim=-1).reshape(-1, 1)
        mod_velocity_norm[mod_velocity_norm <= 0.5] = 1
        mod_velocity = torch.nan_to_num(mod_velocity / mod_velocity_norm)

        # Slow down and repulsion for collision case
        mod_velocity[distance < 0] *= 0.1
        repulsion_velocity = E[:, :, 0] * nominal_velocity_norm * 0.05
        mod_velocity[distance < 0] += repulsion_velocity[distance < 0]

        # Update state
        state = state + self.dt * mod_velocity

        self.closest_dist = distance
        self.move_dist = mod_velocity.norm(p=2, dim=-1) * self.dt

        return state

    def running_cost(self, state, action, t):
        """
        Args:
            state: (B, DOF)
            action: (B, nu)
            t:
        Returns:
            cost: (B)
        """
        cost = 1000 * (self.closest_dist < 0) + 1 * self.move_dist

        return cost

    def terminal_state_cost(self, states, actions):
        """
        Args:
            states: (K x T x nx)
            actions: (K x T x nu)
        Returns:
            cost: (K)
        """
        goal_cost = 10 * (states[:, -1, :] - self.qf).norm(p=2, dim=-1)

        len_cost = -5 * (states[:, -1, :] - states[:, 0, :]).norm(p=2, dim=-1)

        return goal_cost + len_cost
