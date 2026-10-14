#!/usr/bin/env python
"""Single entry point for every scenario converter in this package.

    uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py <name> [options]
    uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py <name> --help
    uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py --list

<name> is one of the registered converters (see --list). Everything after <name> is
parsed by that converter's own argparse (its --help lists its flags). Works the same
run as a script or as `python -m datagen.run_converter`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen import (  # noqa: E402, F401  -- imported for side-effect registration
    convert_pusht_imgonly,
    convert_pusht_latent,
    convert_pusht_noise,
    convert_pushtnoise_denoise,
    pack_imagined,
)
from datagen.dataset_converter import LeRobotDatasetConverter  # noqa: E402


def _print_list() -> None:
    print("converters:")
    for n in sorted(LeRobotDatasetConverter.registry):
        doc = LeRobotDatasetConverter.registry[n].help_text().strip().splitlines()
        print(f"  {n:20s} {doc[0] if doc else ''}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "--list", "list"):
        print(__doc__)
        _print_list()
        sys.exit(0)
    name, rest = argv[0], argv[1:]
    if name not in LeRobotDatasetConverter.registry:
        print(f"unknown converter: {name!r}\n")
        _print_list()
        sys.exit(2)
    LeRobotDatasetConverter.registry[name].main(rest)


if __name__ == "__main__":
    main()
