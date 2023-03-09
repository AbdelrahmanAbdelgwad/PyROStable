"""main module of the library"""
from stable_baselines3 import DQN
import gymnasium as gym  # type: ignore


def test():
    """test function"""

    env = gym.make("CartPole-v1", render_mode="human")
    episodes = 3
    done = False
    model = DQN.load("dqn_cartpole.zip")
    for episode in range(episodes):
        obs, _ = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _, _ = env.step(action)
            env.render()
            print(done, "   ", episode)
        done = False
    env.close()


if __name__ == "__main__":
    test()
