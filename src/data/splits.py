"""Event-level split safety utilities (no heavy ML dependency).

These functions enforce the core leakage guarantee of the paper pipeline:
a typhoon event may appear in exactly ONE of train / val / test. They are
deliberately pure (standard-library only) so they can be unit-tested without
torch, h5py, or numpy.

The gold rule:
    train_typhoon_ids ∩ val_typhoon_ids  = ∅
    train_typhoon_ids ∩ test_typhoon_ids = ∅
    val_typhoon_ids   ∩ test_typhoon_ids = ∅
"""

from __future__ import annotations

from typing import Collection, Iterable, Optional, Sequence


def _to_set(ids: Optional[Iterable[object]]) -> set:
    if ids is None:
        return set()
    return set(ids)


def assert_disjoint_event_split(
    train_ids: Optional[Iterable[object]],
    val_ids: Optional[Iterable[object]] = None,
    test_ids: Optional[Iterable[object]] = None,
) -> None:
    """Raise RuntimeError if any two event sets share a typhoon ID.

    Args:
        train_ids: Typhoon IDs assigned to training.
        val_ids: Typhoon IDs assigned to validation (optional).
        test_ids: Typhoon IDs assigned to testing (optional).

    Raises:
        RuntimeError: If any pair of sets overlaps. The message lists the
            offending IDs so the error is immediately diagnosable.
    """
    train = _to_set(train_ids)
    val = _to_set(val_ids)
    test = _to_set(test_ids)

    pairs = (
        ("train", "val", train, val),
        ("train", "test", train, test),
        ("val", "test", val, test),
    )
    for name_a, name_b, set_a, set_b in pairs:
        overlap = set_a & set_b
        if overlap:
            ids = sorted(overlap, key=str)
            raise RuntimeError(
                f"Event-level split leakage detected: {len(overlap)} typhoon "
                f"ID(s) appear in both {name_a} and {name_b} splits: {ids}. "
                f"The same typhoon must not appear in multiple splits."
            )


def assert_no_duplicate_events(ids: Sequence[object]) -> None:
    """Raise RuntimeError if a typhoon ID appears more than once in a list.

    Duplicate IDs usually indicate a metadata-sidecar misalignment (e.g. a
    sample_idx / typhoon_id mapping error), which would silently break the
    per-event grouping guarantee.
    """
    seen = set()
    dupes = []
    for sid in ids:
        if sid in seen:
            dupes.append(sid)
        seen.add(sid)
    if dupes:
        raise RuntimeError(
            f"Duplicate typhoon ID(s) in event list: {sorted(set(dupes), key=str)}. "
            f"Each event must be unique."
        )


def partition_events_by_assignment(
    id_to_split: dict,
    expected_splits: Optional[Collection[str]] = None,
) -> dict:
    """Group typhoon IDs by their assigned split.

    Args:
        id_to_split: Mapping of typhoon_id -> split name (e.g. 'train', 'val').
        expected_splits: Optional allow-list of split names; any other name
            raises ValueError.

    Returns:
        Dict split name -> set of typhoon IDs.
    """
    partitions: dict = {}
    for tid, split in id_to_split.items():
        if expected_splits is not None and split not in expected_splits:
            raise ValueError(
                f"Unknown split name {split!r} for typhoon {tid!r}. "
                f"Expected one of {sorted(expected_splits)}."
            )
        partitions.setdefault(split, set()).add(tid)
    return partitions
