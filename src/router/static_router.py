from .actions import RouterAction, TaskState
from .base import Router


class StaticRouter(Router):
    """Maps task categories to fixed RouterActions. Used for hand-coded baselines.

    The mapping dict must contain a "default" key as fallback.

    Example:
        StaticRouter({
            "default":       RouterAction("data_analysis", "python_sandbox"),
            "aggregation":   RouterAction("data_analysis", "sql_query"),
            "visualization": RouterAction("visualization",  "chart_generator"),
        })
    """

    def __init__(self, mapping: dict[str, RouterAction]):
        if "default" not in mapping:
            raise ValueError("StaticRouter mapping must contain a 'default' key")
        self._mapping = mapping

    def select_action(self, state: TaskState, task_category: str = "default") -> RouterAction:  # type: ignore[override]
        return self._mapping.get(task_category, self._mapping["default"])

    def update(self, state: TaskState, action: RouterAction, reward: float) -> None:
        pass
