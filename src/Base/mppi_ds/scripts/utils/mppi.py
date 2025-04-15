import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


class MPPI:
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(
        self,
        dynamics,
        running_cost,
        nx,
        noise_sigma,
        noise_mu=None,
        u_min=None,
        u_max=None,
        u_init=None,
        U_init=None,
        num_samples=100,
        horizon=15,
        device="cpu",
        terminal_state_cost=None,
        lambda_=1.0,
        sample_null_action=False,
        noise_abs_cost=False,
    ):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.device = device
        self.dtype = torch.float32

        # mode
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost

        # param
        self.nx = nx
        self.K = num_samples
        self.T = horizon
        self.lambda_ = lambda_

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost

        # sampled results from last command
        self.state = None
        self.cost_total = None
        self.rollout_cost = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None
        self.min_action = None

        # dimensions of state and control
        self.nu = noise_sigma.shape[0]
        self.noise_sigma = noise_sigma.view(self.nu, self.nu).to(self.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_mu = (
            torch.zeros(self.nu, device=self.device, dtype=self.dtype)
            if noise_mu is None
            else noise_mu.view(self.nu).to(self.device)
        )

        self.noise_dist = MultivariateNormal(
            self.noise_mu, covariance_matrix=self.noise_sigma
        )

        self.u_init = (
            torch.zeros(self.nu, device=self.device, dtype=self.dtype)
            if u_init is None
            else u_init.to(self.device)
        )

        self.U = (
            self.noise_dist.sample((self.T,))
            if U_init is None
            else U_init.to(self.device)
        )

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        # make sure if any of them is specified, both are specified
        if self.u_max is not None or self.u_min is not None:
            if self.u_max is None:
                self.u_max = -self.u_min
            if self.u_min is None:
                self.u_min = -self.u_max

            # 确保 u_min 和 u_max 是张量
            self.u_min = self.u_min.clone().detach().to(device=self.device, dtype=self.dtype).view(self.nu)
            self.u_max = self.u_max.clone().detach().to(device=self.device, dtype=self.dtype).view(self.nu)

    def shift_nominal_trajectory(self):
        """
        Shift the nominal trajectory forward one step
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)  # (T, nu)
        self.U[-1] = self.u_init

    def command(self, state, shift_nominal_trajectory=True):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :param shift_nominal_trajectory: Whether to roll the nominal trajectory forward one step. This should be True
        if the command is to be executed. If the nominal trajectory is to be refined then it should be False.
        :returns action: (nu) best action
        """

        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.device)

        cost_total = self._compute_total_cost_batch()

        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1.0 / eta) * self.cost_total_non_zero

        perturbations = []
        for t in range(self.T):
            perturbations.append(
                torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
            )
        perturbations = torch.stack(perturbations)
        self.U = self.U + perturbations  # (T, nu)

        min_cost_idx = torch.argmin(cost_total)
        action = self.actions[min_cost_idx, 0]
        self.min_action = self.actions[min_cost_idx]

        return action

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):
        """
        Args:
            perturbed_actions: (K, T, nu)
        """
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.device, dtype=self.dtype)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        states = [state]
        actions = []

        for t in range(T):
            u = perturbed_actions[:, t]
            state = self.F(state, u, t)
            c = self.running_cost(state, u, t)
            cost_total = cost_total + c

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_total = cost_total + c
        return cost_total, states, actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.noise_dist.rsample((self.K, self.T))  # (K x T x nu), self.U (T x nu)

        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + noise
        if self.sample_null_action:
            perturbed_action[self.K - 2] = self.U
        if self.min_action is not None:
            perturbed_action[-1] = self.min_action

        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)
        self.noise = self.perturbed_action - self.U

        self.rollout_cost, self.states, self.actions = self._compute_rollout_costs(
            self.perturbed_action
        )

        # noise cost
        if self.min_action is not None:
            noise_cost = 0.005 * torch.sum(torch.norm(self.perturbed_action - self.min_action, dim=-1), dim=-1)
        else:
            noise_cost = torch.zeros_like(self.rollout_cost)

        self.cost_total = self.rollout_cost + noise_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            for t in range(self.T):
                u = action[:, slice(t * self.nu, (t + 1) * self.nu)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                action[:, slice(t * self.nu, (t + 1) * self.nu)] = cu
        return action
