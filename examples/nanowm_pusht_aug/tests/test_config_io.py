"""Tests for attribution.config_io."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml
from config_io import load_config


def test_extends_merge_nested_list_scalar(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "nested": {"a": 1, "b": 2},
                "lst": [1, 2],
            }
        )
    )
    child.write_text(
        yaml.safe_dump(
            {
                "extends": "base.yaml",
                "device": "cuda",
                "nested": {"b": 9, "c": 3},
                "lst": [9],
            }
        )
    )
    cfg = load_config(child, out_dir_copy=False)
    assert cfg["device"] == "cuda"
    assert cfg["nested"] == {"a": 1, "b": 9, "c": 3}
    assert cfg["lst"] == [9]


def test_circular_extends_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({"extends": "b.yaml", "x": 1}))
    b.write_text(yaml.safe_dump({"extends": "a.yaml", "y": 2}))
    try:
        load_config(a, out_dir_copy=False)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "circular" in str(e)


def test_expand_keys_tilde(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"out_dir": str(tmp_path / "out"), "ckpt": "~/models/x"}))
    cfg = load_config(cfg_path, out_dir_copy=False)
    assert cfg["ckpt"] == str(tmp_path / "home" / "models" / "x")


def test_cli_dotted_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"finetune": {"save_freq": 42}, "out_dir": str(tmp_path / "o")}))
    script = Path(__file__).resolve().parents[1] / "attribution" / "config_io.py"
    r = subprocess.run(
        [sys.executable, str(script), str(cfg_path), "finetune.save_freq"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "42"


def test_out_dir_copy_writes_resolved(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    out = tmp_path / "run"
    base.write_text(yaml.safe_dump({"device": "cpu", "shared": 1}))
    child.write_text(yaml.safe_dump({"extends": "base.yaml", "device": "cuda", "out_dir": str(out)}))
    cfg = load_config(child, out_dir_copy=True)
    dumped = yaml.safe_load((out / "config.yaml").read_text())
    assert dumped == cfg
    assert dumped["device"] == "cuda"
    assert dumped["shared"] == 1


def test_config_snapshot_no_overwrite_same(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    out = tmp_path / "o"
    cfg_path.write_text(yaml.safe_dump({"device": "cuda", "out_dir": str(out)}))
    load_config(cfg_path, out_dir_copy=True)
    load_config(cfg_path, out_dir_copy=True)
    assert list(out.glob("config.*.yaml")) == []
    lines = (out / "config_provenance.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_config_snapshot_backup_on_change(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    out = tmp_path / "o"
    cfg_path.write_text(yaml.safe_dump({"device": "cpu", "out_dir": str(out)}))
    load_config(cfg_path, out_dir_copy=True)
    old = (out / "config.yaml").read_text()
    cfg_path.write_text(yaml.safe_dump({"device": "cuda", "out_dir": str(out)}))
    load_config(cfg_path, out_dir_copy=True)
    baks = list(out.glob("config.*.yaml"))
    assert len(baks) == 1
    assert baks[0].read_text() == old
    assert "cuda" in (out / "config.yaml").read_text()


def test_out_dir_copy_false_writes_nothing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    out = tmp_path / "o"
    cfg_path.write_text(yaml.safe_dump({"device": "cpu", "out_dir": str(out)}))
    load_config(cfg_path, out_dir_copy=False)
    assert not (out / "config.yaml").exists()
    assert not (out / "config_provenance.jsonl").exists()


def test_sys_path_grows_once(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    out = tmp_path / "o"
    cfg_path.write_text(yaml.safe_dump({"device": "cpu", "out_dir": str(out)}))
    load_config(cfg_path, out_dir_copy=True)
    n = len(sys.path)
    load_config(cfg_path, out_dir_copy=True)
    assert len(sys.path) == n


def test_config_field_is_absolute(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "o"
    cfg_path = Path("c.yaml")
    cfg_path.write_text(yaml.safe_dump({"device": "cpu", "out_dir": str(out)}))
    load_config(cfg_path, out_dir_copy=True)
    import json

    line = json.loads((out / "config_provenance.jsonl").read_text().strip().splitlines()[0])
    assert Path(line["config"]).is_absolute()
    assert Path(line["config"]) == (tmp_path / "c.yaml").resolve()
