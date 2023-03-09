import gym
from policy.dqn import CustomDQNPolicy, CustomFeaturesExtractor
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")
policy = DQN(CustomDQNPolicy, env, learning_rate=0.001, verbose=1)
policy.learn(total_timesteps=3000000)
policy.save("policy")
