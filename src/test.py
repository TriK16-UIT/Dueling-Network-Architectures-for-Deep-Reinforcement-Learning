import gym
from common.atari_wrappers import make_atari, wrap_deepmind
from gym.wrappers import FrameStack
env = make_atari("BreakoutNoFrameskip-v4", "haha", capture_video=False, video_frequency=0)
env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=False)
# env.seed(0)
n1 = []
for _ in range(1):
    n1.append(env.reset(seed=42)[0])
    done = False
    while not done:
        next , reward, done, _ = env.step(env.action_space.sample())
        n1.append(next[0])
env.seed(0)
n2 = []
for _ in range(1):
    n2.append(env.reset()[0])
    done = False
    while not done:
        next , reward, done, _ = env.step(env.action_space.sample())
        n2.append(next[0])
print(n1[0] == n2[0]) # trueprint(n1[1] == n2[1]) # true
print(n1[2] == n2[2]) # sometimes true????