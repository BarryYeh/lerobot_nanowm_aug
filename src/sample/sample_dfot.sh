#!/bin/bash

# Diffusion Forcing video prediction sampling
# Uses first frame(s) as context to predict future frames
#
# All settings are defined in the config file. 
# Use command line args only to override specific settings.

# Required: checkpoint path
CKPT="./results/020-NanoWM-S-2-F4S1-point_maze/checkpoints/latest.ckpt"

# Config file (contains all default settings)
CONFIG="../configs/dino_wm/point_maze_sample_dfot.yaml"

# Run sampling
# - Most settings come from config file
# - Only override what you need via command line
python sample_dfot.py \
    --config $CONFIG \
    --ckpt $CKPT \
    --verbose

# =============================================================================
# Optional: Override specific settings via command line
# =============================================================================
# python sample_dfot.py \
#     --config $CONFIG \
#     --ckpt $CKPT \
#     --num_samples 20 \
#     --n_context_frames 2 \
#     --scheduling_mode sequential \
#     --verbose

