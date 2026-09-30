"""Environment factory for planning.

Uses the MuJoCo / pymunk wrappers from src/environments/. These render exactly
the textures and geometry the world model was trained on, so the diffusion cost
in VAE latent space is meaningful.

We expose only the envs whose planning loop produces non-zero success rates
under our DINO-WM-aligned protocol:

  - point_maze: U-maze with MuJoCo rendering. 40% (2/5) at horizon=5,
    replan_every=5, CEM 100/10/30, DDIM 20.
  - pusht: pushing-T task. Requires goal_source='dset' (random goals are
    typically not reachable in goal_H planner steps); 40% (2/5) under the same
    CEM protocol.

Wall and the deformable envs (rope, granular) are intentionally not registered:
  - wall: world model converges to CEM loss ~0.14 (vs point_maze's 0.0003) at
    15k training steps; even goal_source='dset' produces 0% success. The
    bottleneck is world-model accuracy, not planning.
  - rope/granular: depend on NVIDIA FleX bindings (pyflex) whose static archives
    require gcc-7-era toolchain; building against gcc 13 / glibc 2.31 surfaces
    runtime ABI failures (`__powf_finite`, `cudaSetupArgument`, stack-canary
    mismatches). See git history for the build-out attempt.
"""


def create_planning_env(env_name: str, device: str = "cpu", **kwargs):
    """Create a planning-compatible environment by name."""
    if env_name == "point_maze":
        from environments.pointmaze.point_maze_wrapper import PointMazeWrapper
        from environments.pointmaze.maze_model import U_MAZE
        return PointMazeWrapper(maze_spec=U_MAZE, reward_type="sparse", reset_target=False)

    if env_name == "pusht":
        from environments.pusht.pusht_wrapper import PushTWrapper
        return PushTWrapper(with_velocity=True, with_target=True)

    raise ValueError(f"Unknown planning env: {env_name}")
