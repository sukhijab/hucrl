"""Run the inverted-pendulum using MB-MPO."""
from dotmap import DotMap

from exps.inverted_pendulum import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
    get_true_model_with_agent
)
from exps.inverted_pendulum.plotters import (
    plot_pendulum_trajectories,
    set_figure_params,
)
from exps.inverted_pendulum.util import get_mbmpo_parser, get_mpc_parser
from exps.util import train_and_evaluate
import torch
import numpy as np
from exps.inverted_pendulum.util import PendulumReward, PendulumModel
from rllib.model.abstract_model import AbstractModel
from rllib.algorithms.mpc.cem_shooting import CEMShooting
from rllib.environment.gym_environment import GymEnvironment
from rllib.algorithms.mpc.gradient_based_solver import GradientBasedSolver

class PendulumReward(AbstractModel):
    def __init__(self):
        super(PendulumReward, self).__init__(
            dim_state=(3, ),
            dim_action=(1, ),
            model_kind="rewards"
        )
        self.max_torque = 2.0

    def forward(self, state, action, next_state=None):
        def angle_normalize(angle):
            return ((angle + torch.pi) % (2 * torch.pi)) - torch.pi
        cos_th, sin_th, thdot = state[..., 0].unsqueeze(-1), state[..., 1].unsqueeze(-1), state[..., 2].unsqueeze(-1)
        th = angle_normalize(torch.atan2(sin_th, cos_th))
        u = torch.clip(action, -1, 1) * self.max_torque
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        # costs = jnp.expand_dims(costs, 0)
        return -costs, torch.diag_embed(torch.zeros_like(costs))

class Pendulum(AbstractModel):
    def __init__(self, g=10.0):
        super().__init__(
            dim_state=(3, ),
            dim_action=(1, ),
        )
        self.g = g
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

    def forward(self, state, action, next_state=None):
        def angle_normalize(angle):
            return ((angle + torch.pi) % (2 * torch.pi)) - torch.pi
        cos_th, sin_th, thdot = state[..., 0].unsqueeze(-1), state[..., 1].unsqueeze(-1), state[..., 2].unsqueeze(-1)
        th = angle_normalize(torch.atan2(sin_th, cos_th))

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        u = torch.clip(action, -1, 1) * self.max_torque
        #costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        # costs = jnp.expand_dims(costs, 0)
        newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = torch.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        state = torch.cat((torch.cos(newth), torch.sin(newth), newthdot), axis=-1)
        # state = jnp.expand_dims(state, 0)
        #state = tfd.MultivariateNormalDiag(state, 0.01 * jnp.ones_like(state))
        return state, torch.diag_embed(torch.zeros_like(state))

PLAN_HORIZON, SIM_TRAJECTORIES = 8, 16

steps = 400
dynamical_model = Pendulum()
reward_model = PendulumReward()
solver = CEMShooting(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        num_model_steps=50,
        num_particles=50,
        num_elites=5,
        num_iter=10,

)
env = GymEnvironment("PendulumSwingUp-v0")
obs = env.reset()
for i in range(steps):
    action = solver(torch.from_numpy(obs))[0]
    obs, reward, done, info = env.step(action.detach().numpy()*dynamical_model.max_torque)
    env.render()

env.close()

# parser = get_mpc_parser()
# # parser = get_mbmpo_parser()
# parser.description = "Run Swing-up Inverted Pendulum using Model-Based MPC."
# parser.set_defaults(
#     action_cost=ACTION_COST,
#     train_episodes=TRAIN_EPISODES,
#     environment_max_steps=ENVIRONMENT_MAX_STEPS,
#     plan_horizon=PLAN_HORIZON,
#     sim_num_steps=ENVIRONMENT_MAX_STEPS,
#     sim_initial_states_num_trajectories=SIM_TRAJECTORIES // 2,
#     sim_initial_dist_num_trajectories=SIM_TRAJECTORIES // 2,
#     model_learn_num_iter=0,
#     seed=1,
#     mpc_num_iter=10,
#     mpc_num_particles=10,
#     mpc_num_elites=5,
# )
#
# args = parser.parse_args()
# params = DotMap(vars(args))
# environment, agent = get_true_model_with_agent(params)
# set_figure_params(serif=True, fontsize=9)
# train_and_evaluate(
#     agent, environment, params, plot_callbacks=[plot_pendulum_trajectories]
# )
