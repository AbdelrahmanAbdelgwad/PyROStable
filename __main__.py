"""main module of the library"""
from dataclasses import dataclass
import gymnasium as gym  # type: ignore
from classic_control.pid import PID
from classic_control.full_state_feedback import FSFB


@dataclass
class Param:
    """PID parameters"""

    kp_angle: float = 1
    ki_angle: float = 0.001
    kd_angle: float = 1.5
    kp_pos: float = 0.05
    ki_pos: float = 0
    kd_pos: float = 0


def test_PID():
    """testing PID function"""
    score: int = 0
    env = gym.make("CartPole-v1", render_mode="human")
    episodes = 3
    done = False
    param = Param()
    angle_pid = PID(param.kp_angle, param.ki_angle, param.kd_angle)
    pos_pid = PID(param.kp_pos, param.ki_pos, param.kd_pos)
    for episode in range(episodes):
        obs, _ = env.reset()
        while not done:
            #         action = env.action_space.sample()
            angle_u = angle_pid.compute(0, obs[2])
            pos_u = pos_pid.compute(0, obs[0])
            control_action = angle_u + pos_u
            if control_action > 0:
                action = 0
            else:
                action = 1
            obs, reward, done, _, _ = env.step(action)
            score += reward
            env.render()
            print(f"score: {score}", "  ", episode)
        done = False
    env.close()


def test_FSFB():
    """testing FSFB function"""
    score: int = 0
    env = gym.make("CartPole-v1", render_mode="human")
    episodes = 3
    done = False
    controller = FSFB([5, 1, 10, 1])
    for episode in range(episodes):
        obs, _ = env.reset()
        while not done:
            #         action = env.action_space.sample()
            control_action = controller.compute(obs)
            if control_action > 0:
                action = 0
            else:
                action = 1
            obs, reward, done, _, _ = env.step(action)
            score += reward
            env.render()
            print(f"score: {score}", "  ", episode)
        done = False
    env.close()


if __name__ == "__main__":
    test_FSFB()
