from abc import ABC, abstractmethod

from .actions import RouterAction, TaskState


class Router(ABC):
    @abstractmethod
    def select_action(self, state: TaskState) -> RouterAction:
        """Choose an (agent_role, tool_name) action for the given task state."""
        ...

    @abstractmethod
    def update(self, state: TaskState, action: RouterAction, reward: float) -> None:
        """Update router internal state after observing reward. No-op for static routers."""
        ...
