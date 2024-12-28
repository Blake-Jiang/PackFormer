from gymnasium.envs.registration import register

register(
    id='BinPacking3D-v0',
    entry_point='envs.binpacking3d_env:BinPacking3DEnv',
    max_episode_steps=1000,
)
