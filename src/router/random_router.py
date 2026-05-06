import random

from .actions import RouterAction, TaskState
from .base import Router


class RandomRouter(Router):
    """Picks a uniformly random action from the action space. Used as a control baseline."""

    def __init__(self, action_space: list[RouterAction], seed: int = 42):
        self._action_space = action_space
        self._rng = random.Random(seed)

    def select_action(self, state: TaskState) -> RouterAction:
        return self._rng.choice(self._action_space)

    def update(self, state: TaskState, action: RouterAction, reward: float) -> None:
        pass
