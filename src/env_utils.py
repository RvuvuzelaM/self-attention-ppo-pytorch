import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


def make_env_with_wrappers(env_name, render_mode=None):
    env = gym.make(env_name, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        noop_max=0,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env


def make_env_function(env_name):
    def _thunk():
        return make_env_with_wrappers(env_name)

    return _thunk
