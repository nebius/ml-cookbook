"""Retention policy for committed and incomplete checkpoint steps."""

from collections.abc import Iterable


def plan_checkpoint_pruning(
    steps: Iterable[int], committed_step: int, keep_last: int
) -> list[tuple[int, str]]:
    """Select stale partials and committed steps that should be deleted."""
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative")

    stale_partials = sorted(step for step in steps if step > committed_step)
    doomed = [(step, "stale partial") for step in stale_partials]

    # keep_last=0 means keep every committed checkpoint, but uncommitted steps
    # above the marker remain garbage and are still selected above.
    if keep_last > 0:
        committed = sorted(step for step in steps if step <= committed_step)
        doomed.extend((step, "retention") for step in committed[:-keep_last])

    return doomed
