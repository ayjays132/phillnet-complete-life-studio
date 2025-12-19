from gymnasium.envs.registration import register
from .life_game_env import LifeGameEnv, MAX_AGE

register(
    id="LifeGameEnv-v0",
    entry_point="life_game_env.life_game_env:LifeGameEnv",
    max_episode_steps=MAX_AGE,
)
