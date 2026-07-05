import re
import time

from vectora_core.trace.schema import generate_trace_id, is_valid_trace_id, TRACE_ID_PATTERN


def test_format():
    trace_id = generate_trace_id()
    assert TRACE_ID_PATTERN.fullmatch(trace_id), f"Invalid trace ID: {trace_id}"


def test_prefix():
    assert generate_trace_id().startswith("vct_")


def test_timestamp_is_recent():
    before = int(time.time() * 1000)
    trace_id = generate_trace_id()
    after = int(time.time() * 1000)
    ts = int(trace_id.split("_")[1])
    assert before <= ts <= after


def test_uniqueness():
    ids = {generate_trace_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_is_valid_true():
    assert is_valid_trace_id("vct_1749648123456_a3f7b2c1")


def test_is_valid_false():
    assert not is_valid_trace_id("vct_20260628_ab12")   # old date format
    assert not is_valid_trace_id("vct_1749648123456_ab")  # suffix too short
    assert not is_valid_trace_id("not_a_trace_id")
