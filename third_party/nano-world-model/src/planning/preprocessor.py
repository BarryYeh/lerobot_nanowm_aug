"""Preprocessor for observations."""

import torch
from typing import Dict


class Preprocessor:
    """Preprocessor for transforming observations and actions before planning."""

    def __init__(
        self,
        image_size: int = 256,
        normalize: bool = True,
        action_mean: torch.Tensor = None,
        action_std: torch.Tensor = None,
        device: str = "cuda",
    ):
        """
        Initialize preprocessor.

        Args:
            image_size: Target image size (images will be resized if needed)
            normalize: Whether to normalize images to [-1, 1]
            action_mean: Per-dim mean for action normalization (optional).
            action_std:  Per-dim std  for action normalization (optional).
            device: Device to place action stats on.
        """
        self.image_size = image_size
        self.normalize = normalize
        self.device = device
        self.action_mean = action_mean.to(device) if action_mean is not None else None
        self.action_std = action_std.to(device) if action_std is not None else None

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None or self.action_std is None:
            return action
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None or self.action_std is None:
            return action
        return action * self.action_std + self.action_mean

    def transform_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Transform observations for model input.

        Args:
            obs: Observation dict
                - 'visual': [B, T, C, H, W] or [B, T, H, W, C] images
                - 'proprio': [B, T, D] or None

        Returns:
            Transformed observation dict
        """
        transformed = {}

        # Process visual observations
        if "visual" in obs:
            visual = obs["visual"]

            # Handle channel ordering (ensure [B, T, C, H, W])
            if visual.ndim == 5:
                # Check if channels are last
                if visual.shape[-1] == 3 or visual.shape[-1] == 1:
                    # [B, T, H, W, C] -> [B, T, C, H, W]
                    visual = visual.permute(0, 1, 4, 2, 3)

            # Normalize to [-1, 1] if needed
            if self.normalize and visual.dtype == torch.uint8:
                visual = visual.float() / 255.0 * 2.0 - 1.0
            elif self.normalize and visual.max() > 1.0:
                visual = visual / 255.0 * 2.0 - 1.0

            transformed["visual"] = visual

        # Process proprioceptive observations
        if "proprio" in obs:
            transformed["proprio"] = obs["proprio"]

        return transformed

    def inverse_transform_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Inverse transform (denormalize) observations.

        Args:
            obs: Transformed observation dict

        Returns:
            Original scale observation dict
        """
        inverse = {}

        if "visual" in obs:
            visual = obs["visual"]
            if self.normalize:
                # [-1, 1] -> [0, 255]
                visual = ((visual + 1.0) / 2.0 * 255.0).clamp(0, 255)
            inverse["visual"] = visual

        if "proprio" in obs:
            inverse["proprio"] = obs["proprio"]

        return inverse
