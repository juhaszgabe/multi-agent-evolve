from .actions import RouterAction, TaskState
from .base import Router


class BanditRouter(Router):
    """
    TODO: LinUCB contextual bandit router.

    State vector x: TaskState.to_float_array()
      → task_type one-hot (7) + 4 scalar features + task_embedding (384) = 395-dim

    Action space: defined by Config.action_space (list of RouterAction dicts).

    LinUCB update rule (per action a):
        A_a  +=  x · xᵀ           (d×d matrix)
        b_a  +=  r · x             (d-dim vector)

    Action selection:
        θ_a  =  A_a⁻¹ · b_a
        p_a  =  θ_aᵀ · x  +  α · √(xᵀ · A_a⁻¹ · x)
        choose argmax_a p_a

    Notes:
    - Initialize A_a = I (identity), b_a = 0 for every action.
    - α controls exploration–exploitation tradeoff (tune via cross-validation).
    - router.update() should be called ONCE per step with the terminal reward
      (buffered in orchestrator state, not called with 0-placeholder mid-workflow).
    """

    def select_action(self, state: TaskState) -> RouterAction:
        raise NotImplementedError("BanditRouter not yet implemented")

    def update(self, state: TaskState, action: RouterAction, reward: float) -> None:
        raise NotImplementedError("BanditRouter not yet implemented")
