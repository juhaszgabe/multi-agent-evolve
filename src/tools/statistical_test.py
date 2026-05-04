import time
from scipy import stats
from .base import ToolResult


def statistical_test(test: str, **kwargs) -> ToolResult:
    start = time.time()
    try:
        if test == "t_test":
            stat, p = stats.ttest_ind(kwargs["group1"], kwargs["group2"])
            interpretation = f"p={p:.4f}: {'significant difference' if p < 0.05 else 'no significant difference'} at α=0.05"
        elif test == "chi_square":
            stat, p = stats.chi2_contingency(kwargs["contingency_table"])[:2]
            interpretation = f"p={p:.4f}: {'significant association' if p < 0.05 else 'no significant association'} at α=0.05"
        elif test == "anova":
            stat, p = stats.f_oneway(*kwargs["groups"])
            interpretation = f"p={p:.4f}: {'significant difference' if p < 0.05 else 'no significant difference'} between groups at α=0.05"
        elif test == "correlation":
            stat, p = stats.pearsonr(kwargs["x"], kwargs["y"])
            interpretation = f"r={stat:.4f}, p={p:.4f}: {'significant correlation' if p < 0.05 else 'no significant correlation'}"
        else:
            return ToolResult(success=False, output=None, error=f"Unknown test: {test}")

        return ToolResult(
            success=True,
            output={"statistic": float(stat), "p_value": float(p), "interpretation": interpretation},
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=(time.time() - start) * 1000)
