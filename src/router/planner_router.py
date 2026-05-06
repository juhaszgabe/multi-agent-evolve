from .actions import RouterAction, TaskState
from .base import Router


class PlannerRouter(Router):
    """Delegates every routing decision to the Planner's suggested_tool hint.

    Note: select_action accepts an extra `plan_step` parameter not present on
    the Router ABC. The orchestrator detects isinstance(router, PlannerRouter)
    and passes the plan step. This keeps TaskState free of plan content so the
    bandit can use the same state representation without modifications.
    """

    def select_action(self, state: TaskState, plan_step: dict | None = None) -> RouterAction:  # type: ignore[override]
        if plan_step is None:
            return RouterAction(agent_role="data_analysis", tool_name="python_sandbox")
        return RouterAction(
            agent_role=plan_step.get("agent_role", "data_analysis"),
            tool_name=plan_step.get("suggested_tool"),
        )

    def update(self, state: TaskState, action: RouterAction, reward: float) -> None:
        pass  # PlannerRouter does not learn
