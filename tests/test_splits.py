"""Pure-Python unit tests for event-level split safety (src/data/splits.py).

Runnable without any third-party dependency:
    python tests/test_splits.py
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load splits.py directly (bypassing src/data/__init__.py) so this test runs
# without torch/h5py/numpy. splits.py is intentionally pure standard-library.
_spec = importlib.util.spec_from_file_location(
    "splits", REPO_ROOT / "src" / "data" / "splits.py"
)
_splits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_splits)

assert_disjoint_event_split = _splits.assert_disjoint_event_split
assert_no_duplicate_events = _splits.assert_no_duplicate_events
partition_events_by_assignment = _splits.partition_events_by_assignment


def _expect_runtime_error(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_overlap_raises():
    _expect_runtime_error(lambda: assert_disjoint_event_split({1, 2}, {2, 3}, {4}))
    _expect_runtime_error(lambda: assert_disjoint_event_split({1}, None, {1}))
    _expect_runtime_error(lambda: assert_disjoint_event_split({1}, {2}, {1, 2}))


def test_disjoint_passes():
    assert_disjoint_event_split({1, 2}, {3, 4}, {5, 6})
    assert_disjoint_event_split({1}, None, None)
    assert_disjoint_event_split([], [], [])


def test_duplicate_events_raise():
    _expect_runtime_error(lambda: assert_no_duplicate_events([1, 1, 2]))
    assert_no_duplicate_events([1, 2, 3])


def test_partition_events():
    p = partition_events_by_assignment(
        {1: "train", 2: "train", 3: "val", 4: "test"},
        expected_splits={"train", "val", "test"},
    )
    assert p["train"] == {1, 2}
    assert p["val"] == {3}
    assert p["test"] == {4}
    try:
        partition_events_by_assignment({1: "bogus"}, {"train", "val"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown split name")


def main() -> None:
    tests = [
        ("test_overlap_raises", test_overlap_raises),
        ("test_disjoint_passes", test_disjoint_passes),
        ("test_duplicate_events_raise", test_duplicate_events_raise),
        ("test_partition_events", test_partition_events),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{sys.argv[0]}: {len(tests) - n_fail}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
