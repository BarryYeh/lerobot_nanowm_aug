from types import SimpleNamespace

import pytest

from sample.rollout_selection import select_rollout_slices


def dataset():
    return SimpleNamespace(
        slice_mode="exhaustive", slice_indices=[3, 0, 1, 2, 4],
        all_slices=[SimpleNamespace(traj_idx=t, start_frame=s)
                    for t, s in [(7, 0), (7, 1), (8, 0), (9, 0), (10, 0)]],
        data_source=SimpleNamespace(get_seq_length=lambda t: {7: 105, 8: 100, 9: 99, 10: 120}[t]),
    )


def test_selection_keeps_validation_order_and_accepts_exact_action_headroom():
    assert select_rollout_slices(dataset(), 3, 20, 5, True) == [1, 3, 4]
    assert select_rollout_slices(dataset(), 2, 20, 5) == [1, 2]


def test_insufficient_clips_fail_instead_of_succeeding_with_no_videos():
    with pytest.raises(ValueError, match="Only found 0"):
        select_rollout_slices(dataset(), 1, 100, 5)
    with pytest.raises(ValueError, match="Only found 3"):
        select_rollout_slices(dataset(), 4, 20, 5, True)
