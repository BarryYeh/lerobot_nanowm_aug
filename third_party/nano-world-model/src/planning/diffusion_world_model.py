"""Diffusion world model wrapper for planning."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusion.df_sample import dfot_sample
from diffusion.gaussian_diffusion import GaussianDiffusion
from latent_codecs import LatentCodec


class DiffusionWorldModel(nn.Module):
    """Wrapper to make diffusion model compatible with planning interface."""

    def __init__(
        self,
        model: nn.Module,
        latent_codec: LatentCodec,
        diffusion: GaussianDiffusion,
        args,
    ):
        """
        Initialize diffusion world model.

        Args:
            model: NanoWM transformer model
            latent_codec: visual latent encoder/decoder adapter
            diffusion: Gaussian diffusion process
            args: Training/sampling config
        """
        super().__init__()
        self.model = model
        self.latent_codec = latent_codec
        self.vae = getattr(latent_codec, "vae", None)  # kept for SD-VAE debugging/back-compat
        self.diffusion = diffusion
        self.args = args
        self.device = next(model.parameters()).device
        self.vae_scale_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", None)
        self.vae_precision = getattr(latent_codec, "precision", getattr(args.experiment.infra, "vae_precision", "fp32"))

    @torch.no_grad()
    def encode_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode observations to latent embeddings.

        Args:
            obs: Observation dict
                - 'visual': [B, T, C, H, W] visual observations
                - 'proprio': [B, T, D] proprioceptive states (optional)

        Returns:
            Embeddings dict
                - 'visual': [B, T, D_visual] visual feature embeddings
                - 'proprio': [B, T, D_proprio] proprioceptive embeddings (or None)
        """
        frames = obs["visual"]  # [B, T, C, H, W]
        B, T, C, H, W = frames.shape

        frames_flat = frames.reshape(B * T, C, H, W)
        latents = self.latent_codec.encode(frames_flat)

        # Get feature embeddings from NanoWM encoder
        # Use the model's patch embedding and positional encoding
        latents = latents.reshape(B, T, *latents.shape[1:])  # [B, T, C_lat, H_lat, W_lat]

        # For planning, we use the raw latents as visual features
        # Flatten spatial dimensions to get feature vectors
        C_lat, H_lat, W_lat = latents.shape[2:]
        z_visual = latents.reshape(B, T, C_lat * H_lat * W_lat)  # [B, T, D_visual]

        # Proprioceptive embeddings (pass through if available)
        z_proprio = obs.get("proprio", None)

        return {"visual": z_visual, "proprio": z_proprio}

    @torch.no_grad()
    def rollout(
        self,
        obs_0: Dict[str, torch.Tensor],
        act: torch.Tensor,
        num_sampling_steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Autoregressive rollout: generates chunks of (num_frames - n_context)
        new frames at a time, feeding the most recent context window to the
        next chunk until the full horizon is covered.

        ``NanoWM.forward`` shifts each action by one frame, so the first
        action that should affect a generated frame belongs at index
        ``n_context - 1`` in a sampled chunk.  Keeping that alignment here is
        important for models trained with more than one context frame (for
        example the CSGO checkpoints).
        """
        context_frames = obs_0["visual"]  # [B, T_ctx, C, H, W]
        if context_frames.ndim != 5:
            raise ValueError(
                f"obs_0['visual'] must have shape [B, T, C, H, W], got "
                f"{tuple(context_frames.shape)}"
            )
        if context_frames.shape[1] == 0:
            raise ValueError("obs_0['visual'] must contain at least one context frame")
        if act.ndim != 3:
            raise ValueError(
                f"act must have shape [B, horizon, action_dim], got "
                f"{tuple(act.shape)}"
            )

        B, context_len, C, H, W = context_frames.shape
        if act.shape[0] != B:
            raise ValueError(
                f"obs_0 and act batch sizes must match, got {B} and {act.shape[0]}"
            )
        horizon = act.shape[1]

        if num_sampling_steps is None:
            num_sampling_steps = self.args.model.num_sampling_steps

        num_frames = self.args.model.num_frames
        n_context = self.args.model.n_context_frames
        if n_context < 1 or n_context >= num_frames:
            raise ValueError(
                "model.n_context_frames must be in [1, model.num_frames), "
                f"got n_context_frames={n_context}, num_frames={num_frames}"
            )
        gen_per_chunk = num_frames - n_context
        scheduling_mode = self.args.model.scheduling_mode

        # Encode the supplied observations.  Planning normally starts from a
        # single live frame, while some checkpoints were trained with a larger
        # context window.  Left-pad the initial history by repeating its first
        # frame so the temporal order (oldest -> newest) is preserved.
        ctx_flat = context_frames.reshape(B * context_len, C, H, W)
        ctx_latents = self.latent_codec.encode(ctx_flat)
        # ctx_latents: [B, T_ctx, C_lat, H_lat, W_lat]
        ctx_latents = ctx_latents.reshape(B, context_len, *ctx_latents.shape[1:])

        if context_len >= n_context:
            history = ctx_latents[:, -n_context:]
        else:
            pad = ctx_latents[:, :1].expand(
                -1, n_context - context_len, -1, -1, -1
            )
            history = torch.cat([pad, ctx_latents], dim=1)

        # Preserve the last supplied observation as the rollout's initial
        # frame; ``history`` is only the conditioning window used internally.
        all_latents = [ctx_latents[:, -1:]]
        act_offset = 0

        while act_offset < horizon:
            chunk_len = min(gen_per_chunk, horizon - act_offset)

            # Always generate a full chunk (num_frames) to match temp_embed size,
            # but only keep chunk_len new frames.  The model shifts actions by
            # one frame, therefore prefix the planned actions with dummy values
            # for all but the final context slot.
            planned_actions = act[:, act_offset : act_offset + chunk_len]
            prefix = torch.zeros(
                B,
                n_context - 1,
                act.shape[2],
                device=act.device,
                dtype=act.dtype,
            )
            suffix = torch.zeros(
                B,
                num_frames - (n_context - 1) - chunk_len,
                act.shape[2],
                device=act.device,
                dtype=act.dtype,
            )
            act_chunk = torch.cat([prefix, planned_actions, suffix], dim=1)

            # Context latent for this chunk
            cur_ctx = history  # [B, n_context, C_lat, H_lat, W_lat]

            shape = [B, num_frames, *cur_ctx.shape[2:]]

            chunk_latents = dfot_sample(
                diffusion=self.diffusion,
                model=self.model,
                shape=shape,
                context=cur_ctx,
                n_context_frames=n_context,
                model_kwargs={"action": act_chunk},
                scheduling_mode=scheduling_mode,
                num_sampling_steps=num_sampling_steps,
                eta=eta,
                history_stabilization_level=self.args.experiment.diffusion.history_stabilization_level,
            )  # [B, total_frames, C_lat, H_lat, W_lat]

            # Keep only the newly generated frames (skip context)
            new_latents = chunk_latents[:, n_context : n_context + chunk_len]
            all_latents.append(new_latents)
            history = torch.cat([history, new_latents], dim=1)[:, -n_context:]
            act_offset += chunk_len

        # Concatenate: [context_frame] + [all generated chunks]
        generated_latents = torch.cat(all_latents, dim=1)  # [B, 1+horizon, ...]

        B_out, T_total, C_lat, H_lat, W_lat = generated_latents.shape
        z_visual = generated_latents.reshape(B_out, T_total, C_lat * H_lat * W_lat)
        z_obses = {"visual": z_visual, "proprio": None}

        return z_obses, None

    def forward(self, *args, **kwargs):
        """Forward pass delegates to rollout."""
        return self.rollout(*args, **kwargs)
