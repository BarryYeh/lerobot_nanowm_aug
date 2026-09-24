"""Reusable helpers for the NanoWM PushT augmentation prototype.

Named ``datagen`` (not ``utils``): ``utils`` collides with nano-world-model's own
top-level ``src/utils`` package (a hard dependency of ``imagined_rollout``) and with the
repo-root ``utils/`` doc-check scripts -- both would shadow / be shadowed here.

Two independent groups; import only what you need so the module stays usable from BOTH
venvs:

  * ``action_convert``       -- NanoWM <-> diffusion-policy action converter (pure numpy)
  * ``coverage_from_frame``  -- coverage/pose from an RGB frame (numpy + cv2)
        both load in the nanowm conda env and the lerobot venv.

  * ``dataset_converter``    -- ``LeRobotDatasetConverter`` base class
  * ``convert_*`` / ``pack_*`` -- scenario converters (lerobot venv only; need LeRobotDataset)
        dispatched by ``run_converter.py``.

This ``__init__`` deliberately imports nothing, so ``from datagen import action_convert``
does not drag in ``lerobot`` on the nanowm side.
"""
