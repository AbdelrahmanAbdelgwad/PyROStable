import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 4):
        super().__init__(observation_space=observation_space, features_dim=features_dim)

        self.network = nn.Sequential(
            nn.Linear(in_features=4, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=4),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations)  # type: ignore


class CustomQNetwork(nn.Module):
    def __init__(self, features_dim: int, action_space: gym.spaces.Discrete):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, action_space.n)

    def forward(self, features: th.Tensor) -> th.Tensor:
        x = th.relu(self.fc1(features))
        x = th.relu(self.fc2(x))
        q_values = self.q_out(x)

        return q_values

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        features = self(observation)
        q_values = self.q_out(features)
        return q_values, features

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)


class CustomDQNPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        learning_rate: float = 1e-4,
    ):

        super().__init__(
            observation_space,
            action_space,
            learning_rate,
        )

        self.features_extractor = CustomFeaturesExtractor(
            observation_space=observation_space, features_dim=4
        )
        self.q_net = CustomQNetwork(
            features_dim=self.features_extractor.features_dim, action_space=action_space
        )

    def forward(self, obs):
        features = self.extract_features(obs)
        q_values = self.q_net(features)
        return q_values

    def extract_features(self, observations: th.Tensor) -> th.Tensor:
        return self.features_extractor(observations)
