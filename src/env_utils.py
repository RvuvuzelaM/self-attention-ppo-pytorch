import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


class EpisodicLifeEnv(gym.Wrapper):
    """Treat life loss as episode end (but only reset on true done)."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip reward to {-1, 0, +1} using sign."""

    def reward(self, reward):
        return np.sign(reward)


def make_env_with_wrappers(env_name, render_mode=None):
    env = gym.make(env_name, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = EpisodicLifeEnv(env)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = ClipRewardWrapper(env)
    env = FrameStackObservation(env, stack_size=4)
    return env


def make_env_function(env_name):
    def _thunk():
        return make_env_with_wrappers(env_name)

    return _thunk
