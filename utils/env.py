import os

import gym
import numpy as np
import torch
# from gym.spaces.box import Box
# from gym.wrappers.clip_action import ClipAction
# from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
#                                                      EpisodicLifeEnv,
#                                                      FireResetEnv,
#                                                      MaxAndSkipEnv,
#                                                      NoopResetEnv, WarpFrame)
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
#                                               VecEnvWrapper)
# from stable_baselines3.common.vec_env.vec_normalize import \
#     VecNormalize as VecNormalize_

######## LOOK into below from the original repo from Chen
######## from a2c_ppo_acktr import utils
######## from a2c_ppo_acktr.envs import make_vec_envs

class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False



def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

# def make_env(env_id, seed, rank, log_dir, allow_early_resets):
#     def _thunk():
#         if env_id.startswith("dm"):
#             _, domain, task = env_id.split('.')
#             env = dmc2gym.make(domain_name=domain, task_name=task)
#             env = ClipAction(env)
#         else:
#             env = gym.make(env_id)

#         is_atari = hasattr(gym.envs, 'atari') and isinstance(
#             env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
#         if is_atari:
#             env = NoopResetEnv(env, noop_max=30)
#             env = MaxAndSkipEnv(env, skip=4)

#         env.seed(seed + rank)

#         if str(env.__class__.__name__).find('TimeLimit') >= 0:
#             env = TimeLimitMask(env)

#         #if log_dir is not None:
#         env = Monitor(env,
#                       os.path.join(log_dir, str(rank)),
#                       allow_early_resets=allow_early_resets)

#         if is_atari:
#             if len(env.observation_space.shape) == 3:
#                 env = EpisodicLifeEnv(env)
#                 if "FIRE" in env.unwrapped.get_action_meanings():
#                     env = FireResetEnv(env)
#                 env = WarpFrame(env, width=84, height=84)
#                 env = ClipRewardEnv(env)
#         elif len(env.observation_space.shape) == 3:
#             raise NotImplementedError(
#                 "CNN models work only for atari,\n"
#                 "please use a custom wrapper for a custom pixel input env.\n"
#                 "See wrap_deepmind for an example.")

#         # If the input has shape (W,H,3), wrap for PyTorch convolutions
#         obs_shape = env.observation_space.shape
#         if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
#             env = TransposeImage(env, op=[2, 0, 1])

#         return env

#     return _thunk
