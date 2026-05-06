"""Req 6 — TaskStateExtractor produces correct TaskState from task description + metadata."""
import pytest
from state_extractor import TaskStateExtractor, TASK_TYPES
from router.actions import TaskState


@pytest.fixture
def extractor():
    return TaskStateExtractor()


def _meta(rows: int, cols: int, dtypes: dict | None = None) -> dict:
    dtypes = dtypes or {}
    return {"shape": {"rows": rows, "columns": cols}, "dtypes": dtypes}


# ---------------------------------------------------------------------------
# task_type one-hot
# ---------------------------------------------------------------------------

def test_task_type_sums_to_one(extractor):
    state = extractor.extract("compute the total revenue", _meta(100, 5), {})
    assert sum(state.task_type) == 1.0


def test_task_type_length(extractor):
    state = extractor.extract("anything", _meta(10, 3), {})
    assert len(state.task_type) == len(TASK_TYPES)


def test_aggregation_keyword_detected(extractor):
    state = extractor.extract("what is the total revenue by region?", _meta(500, 5), {})
    idx = TASK_TYPES.index("aggregation")
    assert state.task_type[idx] == 1.0


def test_stat_keyword_detected(extractor):
    # "correlation" is a stat keyword; no compare/aggregation keywords present
    state = extractor.extract("run a t-test on the salary distributions", _meta(200, 8), {})
    idx = TASK_TYPES.index("stat")
    assert state.task_type[idx] == 1.0


def test_default_to_text_when_no_match(extractor):
    state = extractor.extract("tell me about this dataset", _meta(50, 3), {})
    idx = TASK_TYPES.index("text")
    assert state.task_type[idx] == 1.0


# ---------------------------------------------------------------------------
# Temporal keyword
# ---------------------------------------------------------------------------

def test_temporal_keyword_detected(extractor):
    state = extractor.extract("show monthly revenue trend over time", _meta(1000, 4), {})
    assert state.has_temporal_keyword is True


def test_no_temporal_keyword(extractor):
    state = extractor.extract("compare sales by region", _meta(100, 4), {})
    assert state.has_temporal_keyword is False


# ---------------------------------------------------------------------------
# CSV size bucket
# ---------------------------------------------------------------------------

def test_small_csv_bucket(extractor):
    state = extractor.extract("task", _meta(rows=500, cols=5), {})
    assert state.csv_size_bucket == 0


def test_medium_csv_bucket(extractor):
    state = extractor.extract("task", _meta(rows=50_000, cols=5), {})
    assert state.csv_size_bucket == 1


def test_large_csv_bucket(extractor):
    state = extractor.extract("task", _meta(rows=200_000, cols=5), {})
    assert state.csv_size_bucket == 2


# ---------------------------------------------------------------------------
# Column bucket
# ---------------------------------------------------------------------------

def test_narrow_column_bucket(extractor):
    state = extractor.extract("task", _meta(100, cols=3), {})
    assert state.num_columns_bucket == 0


def test_medium_column_bucket(extractor):
    state = extractor.extract("task", _meta(100, cols=10), {})
    assert state.num_columns_bucket == 1


def test_wide_column_bucket(extractor):
    state = extractor.extract("task", _meta(100, cols=25), {})
    assert state.num_columns_bucket == 2


# ---------------------------------------------------------------------------
# Numeric target detection
# ---------------------------------------------------------------------------

def test_numeric_target_detected(extractor):
    meta = _meta(100, 3, dtypes={"revenue": "float64", "region": "object"})
    state = extractor.extract("what is the average revenue?", meta, {})
    assert state.has_numeric_target is True


def test_no_numeric_target(extractor):
    meta = _meta(100, 2, dtypes={"name": "object", "city": "object"})
    state = extractor.extract("list all cities", meta, {})
    assert state.has_numeric_target is False


# ---------------------------------------------------------------------------
# Embedding placeholder
# ---------------------------------------------------------------------------

def test_embedding_is_384_zeros(extractor):
    state = extractor.extract("any task", _meta(10, 2), {})
    assert len(state.task_embedding) == 384
    assert all(v == 0.0 for v in state.task_embedding)


# ---------------------------------------------------------------------------
# to_float_array
# ---------------------------------------------------------------------------

def test_to_float_array_length(extractor):
    state = extractor.extract("count rows", _meta(100, 5), {})
    arr = state.to_float_array()
    # 7 task_type + 4 scalars + 384 embedding = 395
    assert len(arr) == 395


def test_to_float_array_all_floats(extractor):
    state = extractor.extract("count rows", _meta(100, 5), {})
    arr = state.to_float_array()
    assert all(isinstance(v, float) for v in arr)
