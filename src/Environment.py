import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
from Config.config1_ddqn import config
import numpy as np
from framebuffer import FrameBuffer
from PIL import Image
import atari_wrappers

class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _to_gray_scale(self, rgb, channel_weights=[0.8, 0.1, 0.1]):
        return np.dot(np.asarray(rgb[...,:3]), channel_weights)


    def observation(self, img):
        """what happens to each observation"""

        img_cropped = img[10:-20]
        img_resized = Image.fromarray(img_cropped).resize((config['W'], config['H']), Image.LANCZOS)
        img_resized = np.array(img_resized)
        
        img_gray = self._to_gray_scale(img_resized)

        img_normalized = img_gray.reshape(self.img_size).astype(np.float32) / 255.0
        return img_normalized
    
def PrimaryAtariWrap(env, clip_rewards=True, skip=4):
    # assert 'NoFrameskip' in env.spec.id

    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=skip)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env

def make_env(clip_rewards=True, seed=None, skip=4, render_mode=None):
    env = None
    if render_mode is not None:
        env = gym.make(config['env_name'], render_mode=render_mode)
    else:
        env = gym.make(config['env_name'])  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards, skip)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env
