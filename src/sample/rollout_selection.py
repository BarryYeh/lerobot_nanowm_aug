"""Deterministic selection of recorded clips for open-loop rollout previews."""


def select_rollout_slices(dataset, num_samples, rollout_length, frame_interval,
                          unique_trajectories=False):
    if min(num_samples, rollout_length, frame_interval) < 1:
        raise ValueError("Sample count, rollout length, and frame interval must be positive")
    if dataset.slice_mode != "exhaustive":
        raise ValueError("Rollout requires exhaustive slice_mode")
    selected, seen = [], set()
    for i, slice_idx in enumerate(dataset.slice_indices):
        info = dataset.all_slices[slice_idx]
        if unique_trajectories and info.traj_idx in seen:
            continue
        # Actions are grouped in complete frame_interval blocks, including the last.
        if dataset.data_source.get_seq_length(info.traj_idx) - info.start_frame < rollout_length * frame_interval:
            continue
        selected.append(i)
        seen.add(info.traj_idx)
        if len(selected) == num_samples:
            break
    if len(selected) < num_samples:
        raise ValueError(f"Only found {len(selected)} valid clips; requested {num_samples}. "
                         "Reduce rollout_length or num_samples.")
    return selected
