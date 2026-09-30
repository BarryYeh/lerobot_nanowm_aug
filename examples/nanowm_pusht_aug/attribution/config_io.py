"""Config loading with `extends:` resolution -- deliberately free of lerobot imports so the shell
recipes can read a resolved value without paying for `import lerobot` on every call.

    python examples/nanowm_pusht_aug/attribution/config_io.py <config.yaml> finetune.save_freq
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

EXPAND_KEYS = ("ckpt", "pool_index", "oracle_npz", "out_dir", "nanowm_python", "dataset_dir", "lerobot_home")


def _merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins. Lists and scalars are replaced, never appended."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, out_dir_copy: bool = True) -> dict:
    """Load a yaml config, resolving `extends:` (relative to the config's own directory) so shared
    values live in one file, then snapshot the RESOLVED config into out_dir for reproducibility."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    seen = [path.resolve()]
    while "extends" in cfg:
        parent = (path.parent / cfg.pop("extends")).resolve()
        if parent in seen:
            raise ValueError(f"circular extends: {parent}")
        seen.append(parent)
        cfg = _merge(yaml.safe_load(parent.read_text()), cfg)
    for key in EXPAND_KEYS:
        if isinstance(cfg.get(key), str):
            cfg[key] = str(Path(cfg[key]).expanduser())
    if not cfg.get("out_dir"):
        return cfg
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if out_dir_copy:
        text = yaml.safe_dump(cfg, sort_keys=False)
        dest = out / "config.yaml"
        if dest.exists() and dest.read_text() != text:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            bak = out / f"config.{ts}.yaml"
            n = 1
            while bak.exists():
                bak = out / f"config.{ts}.{n}.yaml"
                n += 1
            dest.rename(bak)
        dest.write_text(text)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from datagen.provenance import provenance_entry  # noqa: E402

        entry = provenance_entry(
            None,
            config=str(path),
            config_sha1=hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest(),
        )
        with (out / "config_provenance.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
    return cfg


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: config.py <config.yaml> <dotted.key>")
    node = load_config(sys.argv[1], out_dir_copy=False)
    for key in sys.argv[2].split("."):
        node = node[key]
    print(node)


if __name__ == "__main__":
    main()
