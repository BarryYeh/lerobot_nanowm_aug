"""Planning environment adapters.

Each adapter wraps an existing DINO-WM environment and exposes a uniform
interface for the MPC planning loop:

    env.sample_random_init_goal_states(seed) -> (init_state, goal_state)
    env.prepare(seed, state) -> (obs_dict, state)
    env.step(action) -> (obs_dict, reward, done, info)
    env.eval_state(goal_state, cur_state) -> {success, state_dist}
    env.render_state(seed, state) -> obs_dict   # for goal rendering
"""

from planning.envs.registry import create_planning_env

__all__ = ["create_planning_env"]
