"""Implementation of the full state feedback controller"""
from typing import List
import numpy as np


class FSFB:
    """Full state feedback controller class"""

    def __init__(self, gain: List):
        """creates a full state feedback controller using provided gains

        Args:
            gain (List): Diagonal elements of the gain matrix provide in a list
        """
        self.k = gain

    def compute(self, state_vector: np.ndarray):
        """Computes the control action based on the given state

        Args:
            state_vector (np.ndarray): Measured state vector of the environment

        Returns:
            float: control action provided based on the state and the gains
        """
        action: float = 0
        for index, state in enumerate(state_vector):
            action += self.k[index] * state
        action = -action
        return action

    def set_fsfb(self, gain: List):
        """sets the gain of full state feedback controller

        Args:
            gain (List): Diagonal elements of the gain matrix provide in a list
        """
        self.k = gain
