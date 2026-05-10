from __future__ import annotations

import os

import numpy as np

from .actions import RouterAction, TaskState
from .base import Router


class BanditRouter(Router):
    """
    LinUCB contextual bandit router.

    State vector x: TaskState.to_float_array()
      → task_type one-hot (7) + 4 scalar features + task_embedding (384) = 395-dim

    Per-action matrices (n_actions × d):
        A_a  (d×d)  initialised to identity
        b_a  (d,)   initialised to zeros

    Selection: argmax_a [ θ_aᵀ·x + α·√(xᵀ·A_a⁻¹·x) ]   where θ_a = A_a⁻¹·b_a
    Update:    A_a += x·xᵀ,  b_a += r·x   (called once per workflow)
    """

    def __init__(
        self,
        action_space: list[RouterAction],
        alpha: float = 0.25,
        state_dim: int = 395,
        state_path: str | None = None,
    ) -> None:
        n = len(action_space)
        self._action_space = list(action_space)
        self._alpha = alpha
        self._state_dim = state_dim
        self._state_path = state_path
        self._action_idx: dict[tuple, int] = {
            (a.agent_role, a.tool_name): i for i, a in enumerate(action_space)
        }
        self._A = np.stack([np.eye(state_dim, dtype=np.float64)] * n)   # (n, d, d)
        self._b = np.zeros((n, state_dim), dtype=np.float64)             # (n, d)

        if state_path and os.path.exists(state_path):
            self._load(state_path)

    # ------------------------------------------------------------------
    # Router interface
    # ------------------------------------------------------------------

    def select_action(self, state: TaskState) -> RouterAction:
        x = np.array(state.to_float_array(), dtype=np.float64)
        scores = np.empty(len(self._action_space))
        for i, (A, b) in enumerate(zip(self._A, self._b)):
            theta = np.linalg.solve(A, b)
            ucb = self._alpha * np.sqrt(x @ np.linalg.solve(A, x))
            scores[i] = theta @ x + ucb
        return self._action_space[int(np.argmax(scores))]

    def update(self, state: TaskState, action: RouterAction, reward: float) -> None:
        idx = self._action_idx.get((action.agent_role, action.tool_name))
        if idx is None:
            print(f"[BanditRouter] Warning: action ({action.agent_role}, {action.tool_name}) "
                  "not in action_space — update ignored")
            return
        x = np.array(state.to_float_array(), dtype=np.float64)
        self._A[idx] += np.outer(x, x)
        self._b[idx] += reward * x
        if self._state_path:
            self._save(self._state_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, path: str) -> None:
        np.savez(path, A=self._A, b=self._b)

    def _load(self, path: str) -> None:
        data = np.load(path)
        n, d = len(self._action_space), self._state_dim
        if data["A"].shape != (n, d, d) or data["b"].shape != (n, d):
            raise ValueError(
                f"Loaded bandit state shape {data['A'].shape} does not match "
                f"current action_space/state_dim ({n}, {d}, {d})"
            )
        self._A = data["A"]
        self._b = data["b"]
