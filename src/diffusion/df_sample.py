"""
Diffusion Forcing Scheduling Utilities for NanoWM
Adapted from: https://github.com/kwsong0113/diffusion-forcing-transformer

This module provides scheduling matrix generation for Diffusion Forcing sampling,
where each frame can have independent noise levels during the denoising process.
"""
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple, Literal


# =============================================================================
# Scheduling Matrix Generation
# =============================================================================

def generate_full_sequence_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray:
    """
    Full sequence schedule: all tokens denoise together (traditional diffusion).
    
    Args:
        sampling_timesteps: Number of DDIM sampling steps
        horizon: Number of tokens (frames) to generate
    
    Returns:
        schedule: [sampling_timesteps + 1, horizon] matrix of DDIM step indices
    """
    # Each row: all tokens have the same noise level
    # Decreasing from sampling_timesteps to 0
    return np.arange(sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)


def generate_pyramid_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray:
    """
    Pyramid schedule: tokens denoise with staggered start times.
    Earlier tokens start denoising first, later tokens are delayed.
    
    Args:
        sampling_timesteps: Number of DDIM sampling steps
        horizon: Number of tokens (frames) to generate
    
    Returns:
        schedule: [total_steps, horizon] matrix of DDIM step indices (NOT actual timesteps)
                  These indices will be converted to actual timesteps by ddim_idx_to_timestep()
    
    Example (sampling_timesteps=4, horizon=4, num_timesteps=1000):
        DDIM indices returned by this function:
            Step 0: [4, 4, 4, 4]  ->  After conversion: [999, 999, 999, 999]
            Step 1: [3, 4, 4, 4]  ->  After conversion: [749, 999, 999, 999]
            Step 2: [2, 3, 4, 4]  ->  After conversion: [499, 749, 999, 999]
            Step 3: [1, 2, 3, 4]  ->  After conversion: [249, 499, 749, 999]
            Step 4: [0, 1, 2, 3]  ->  After conversion: [ -1, 249, 499, 749]  # Frame 0 done
            Step 5: [0, 0, 1, 2]  ->  After conversion: [ -1,  -1, 249, 499]
            Step 6: [0, 0, 0, 1]  ->  After conversion: [ -1,  -1,  -1, 249]
            Step 7: [0, 0, 0, 0]  ->  After conversion: [ -1,  -1,  -1,  -1]  # All done
    """
    # Total steps = sampling_timesteps + horizon - 1
    # Each token's denoising is delayed by its index
    total_steps = sampling_timesteps + horizon
    schedule = np.zeros((total_steps, horizon), dtype=np.int64)
    
    for step in range(total_steps):
        for token in range(horizon):
            # Each token's denoising delay = token index
            effective_step = step - token
            if effective_step < 0:
                schedule[step, token] = sampling_timesteps  # Not started yet, pure noise
            elif effective_step >= sampling_timesteps:
                schedule[step, token] = 0  # Completed
            else:
                schedule[step, token] = sampling_timesteps - effective_step
    
    return schedule


def generate_sequential_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray:
    """
    Sequential schedule: fully denoise one frame before starting the next.
    The self-forcing way to auto-regressively generate videos.
    
    Args:
        sampling_timesteps: Number of DDIM sampling steps
        horizon: Number of tokens (frames) to generate
    
    Returns:
        schedule: [total_steps, horizon] matrix of DDIM step indices (NOT actual timesteps)
                  These indices will be converted to actual timesteps by ddim_idx_to_timestep()
    
    Example (sampling_timesteps=4, horizon=3, num_timesteps=1000):
        DDIM indices -> After conversion to actual timesteps:
            Step 0:  [4, 4, 4]  ->  [999, 999, 999]  # All noise
            Step 1:  [3, 4, 4]  ->  [749, 999, 999]  # Frame 0 denoising
            Step 2:  [2, 4, 4]  ->  [499, 999, 999]
            Step 3:  [1, 4, 4]  ->  [249, 999, 999]
            Step 4:  [0, 4, 4]  ->  [ -1, 999, 999]  # Frame 0 done (-1 = clean)
            Step 5:  [0, 3, 4]  ->  [ -1, 749, 999]  # Frame 1 starts
            Step 6:  [0, 2, 4]  ->  [ -1, 499, 999]
            Step 7:  [0, 1, 4]  ->  [ -1, 249, 999]
            Step 8:  [0, 0, 4]  ->  [ -1,  -1, 999]  # Frame 1 done
            Step 9:  [0, 0, 3]  ->  [ -1,  -1, 749]  # Frame 2 starts
            Step 10: [0, 0, 2]  ->  [ -1,  -1, 499]
            Step 11: [0, 0, 1]  ->  [ -1,  -1, 249]
            Step 12: [0, 0, 0]  ->  [ -1,  -1,  -1]  # All done
    
    """
    total_steps = sampling_timesteps * horizon + 1
    schedule = np.full((total_steps, horizon), sampling_timesteps, dtype=np.int64)
    
    for token in range(horizon):
        # Each token starts denoising after all previous tokens are done
        start_step = token * sampling_timesteps
        for s in range(sampling_timesteps + 1):
            step_idx = start_step + s
            if step_idx < total_steps:
                # Set this token's noise level for all subsequent steps
                schedule[step_idx:, token] = sampling_timesteps - s
    
    return schedule


def ddim_idx_to_timestep(
    indices: torch.Tensor, 
    num_timesteps: int, 
    sampling_timesteps: int
) -> torch.Tensor:
    """
    Convert DDIM step indices [0, sampling_timesteps] to actual timesteps [-1, num_timesteps-1].
    -1 means fully clean (no noise).
    
    Args:
        indices: DDIM step indices
        num_timesteps: Total diffusion timesteps (e.g., 1000)
        sampling_timesteps: Number of DDIM sampling steps (e.g., 50)
    
    Returns:
        timesteps: Actual diffusion timesteps, -1 for clean frames
    """
    real_steps = torch.linspace(-1, num_timesteps - 1, sampling_timesteps + 1).long()
    real_steps = real_steps.to(indices.device)
    return real_steps[indices.clamp(0, sampling_timesteps)]


def generate_scheduling_matrix(
    num_frames: int,
    num_timesteps: int,
    sampling_timesteps: int,
    mode: Literal["full_sequence", "pyramid", "sequential"] = "pyramid",
    n_context_frames: int = 0,
) -> torch.Tensor:
    """
    Generate the scheduling matrix for Diffusion Forcing sampling.
    
    Args:
        num_frames: Number of frames to generate
        num_timesteps: Total diffusion timesteps (e.g., 1000)
        sampling_timesteps: Number of DDIM sampling steps (e.g., 50)
        mode: Scheduling mode:
            - "full_sequence": Traditional diffusion, all frames denoise together
            - "pyramid": Pyramid schedule, frames overlap with 1-step delay (faster)
            - "sequential": Fully serial, one frame completes before next starts (slowest)
        n_context_frames: Number of context frames (will be set to -1 in schedule)
    
    Returns:
        scheduling_matrix: [num_steps, num_frames] tensor of actual timesteps
                          -1 indicates clean (no noise) frames
    """
    # Generate the schedule only for frames we actually need to denoise.
    # Previously the schedule was generated for the full horizon and context frames were
    # masked to -1 afterwards, which wasted n_context_frames * sampling_timesteps steps in
    # sequential mode (context frames do not need denoising steps budgeted for them).
    num_gen_frames = num_frames - n_context_frames
    if num_gen_frames <= 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be > n_context_frames ({n_context_frames})"
        )

    if mode == "full_sequence":
        gen_matrix = generate_full_sequence_schedule(sampling_timesteps, num_gen_frames)
    elif mode == "pyramid":
        gen_matrix = generate_pyramid_schedule(sampling_timesteps, num_gen_frames)
    elif mode == "sequential":
        gen_matrix = generate_sequential_schedule(sampling_timesteps, num_gen_frames)
    else:
        raise ValueError(f"Unknown scheduling mode: {mode}")

    # Prepend n_context_frames columns of DDIM idx 0 (→ actual timestep -1 = clean after conversion).
    if n_context_frames > 0:
        context_cols = np.zeros((gen_matrix.shape[0], n_context_frames), dtype=np.int64)
        gen_matrix = np.concatenate([context_cols, gen_matrix], axis=1)

    matrix = torch.from_numpy(gen_matrix).long()
    matrix = ddim_idx_to_timestep(matrix, num_timesteps, sampling_timesteps)

    return matrix


# =============================================================================
# Convenience Functions
# =============================================================================

def dfot_sample(
    diffusion,
    model,
    shape: Tuple[int, ...],
    context: Optional[Tensor] = None,
    n_context_frames: int = 0,
    scheduling_mode: Literal["full_sequence", "pyramid", "sequential"] = "pyramid",
    num_sampling_steps: int = 50,
    model_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
    progress: bool = True,
    eta: float = 0.0,
    clip_denoised: bool = False,
    n_generate_frames: Optional[int] = None,
    history_stabilization_level: float = 0.0,
) -> Tensor:
    """
    Convenience function for Diffusion Forcing sampling.

    Args:
        diffusion: NanoWM's GaussianDiffusion object
        model: NanoWM model
        shape: Output shape [B, F, C, H, W]
        context: Optional context frames [B, F_ctx, C, H, W]
        n_context_frames: Number of context frames
        scheduling_mode: Scheduling mode:
            - "full_sequence": Traditional diffusion, all frames denoise together
            - "pyramid": Pyramid schedule with overlap (DFoT)
            - "sequential": Fully serial, one frame at a time (Self-Forcing)
        num_sampling_steps: Number of DDIM sampling steps
        model_kwargs: Additional model arguments
        device: Device to run on
        progress: Whether to show progress bar
        eta: DDIM eta parameter (0 = deterministic)
        clip_denoised: Whether to clip denoised samples to [-1, 1]
        n_generate_frames: If set, truncate the scheduling matrix once this many
            frames (after context) are fully denoised. Useful in sliding-window
            rollout where only the first generated frame is kept per step.
        history_stabilization_level: DFoT stabilization level in [0, 1). 0 disables.

    Returns:
        samples: Generated video [B, F, C, H, W]

    Example:
        >>> from diffusion import create_diffusion
        >>> from sample.utils.df_sample import dfot_sample
        >>>
        >>> diffusion = create_diffusion(timestep_respacing="50")
        >>>
        >>> # Unconditional generation with pyramid schedule
        >>> samples = dfot_sample(
        ...     diffusion, model,
        ...     shape=(4, 16, 4, 32, 32),
        ...     scheduling_mode="pyramid",
        ... )
        >>>
        >>> # Video prediction with sequential (self-forcing)
        >>> samples = dfot_sample(
        ...     diffusion, model,
        ...     shape=(4, 16, 4, 32, 32),
        ...     context=first_4_frames,
        ...     n_context_frames=4,
        ...     scheduling_mode="sequential",  # One frame at a time
        ... )
    """
    if device is None:
        device = next(model.parameters()).device

    batch_size, num_frames = shape[:2]

    # Generate scheduling matrix
    scheduling_matrix = generate_scheduling_matrix(
        num_frames=num_frames,
        num_timesteps=diffusion.num_timesteps,
        sampling_timesteps=num_sampling_steps,
        mode=scheduling_mode,
        n_context_frames=n_context_frames,
    )

    # Early stopping: truncate schedule once target frames are fully denoised
    if n_generate_frames is not None:
        target_frame_idx = n_context_frames + n_generate_frames - 1
        target_column = scheduling_matrix[:, target_frame_idx]
        clean_rows = (target_column == -1).nonzero(as_tuple=True)[0]
        if len(clean_rows) > 0:
            truncate_at = clean_rows[0].item() + 1
            scheduling_matrix = scheduling_matrix[:truncate_at]

    if history_stabilization_level > 0.0 and n_context_frames > 0 and context is not None:
        assert 0.0 < history_stabilization_level < 1.0
        t_stab = int(round(history_stabilization_level * (diffusion.num_timesteps - 1)))
        scheduling_matrix[:, :n_context_frames] = t_stab
        t_stab_tensor = torch.full((batch_size, n_context_frames), t_stab, device=context.device, dtype=torch.long)
        context = context.clone()
        context[:, :n_context_frames] = diffusion.q_sample(context[:, :n_context_frames], t_stab_tensor)

    # Call the diffusion's dfot_sample_loop
    return diffusion.dfot_sample_loop(
        model=model,
        shape=shape,
        scheduling_matrix=scheduling_matrix,
        context=context,
        n_context_frames=n_context_frames,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
        eta=eta,
    )
