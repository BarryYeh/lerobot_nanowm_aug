"""Tests for closed_loop_rollout._wait proc liveness."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("gym_pusht")

from closed_loop_rollout import _wait  # noqa: E402


def test_wait_raises_when_proc_dies(tmp_path: Path) -> None:
    missing = tmp_path / "frames_1.npz"
    (tmp_path / "wm_worker.log").write_text("line1\nline2\nbad boom\n")
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    t0 = time.time()
    with pytest.raises(RuntimeError, match="code 3") as ei:
        _wait(missing, timeout=10.0, proc=proc)
    assert time.time() - t0 < 3.0
    assert "bad boom" in str(ei.value)


def test_wait_ok_when_file_written(tmp_path: Path) -> None:
    target = tmp_path / "frames_1.npz"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import time; time.sleep(0.2); open({str(target)!r},'w').write('x')",
        ]
    )
    _wait(target, timeout=5.0, proc=proc)
    proc.wait(timeout=5)


def test_wait_stop_file(tmp_path: Path) -> None:
    missing = tmp_path / "frames_1.npz"
    stop = tmp_path / "stop"
    stop.write_text("")
    with pytest.raises(RuntimeError, match="worker stopped"):
        _wait(missing, timeout=5.0, stop=stop)


def test_wait_timeout_live_proc(tmp_path: Path) -> None:
    missing = tmp_path / "frames_1.npz"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        with pytest.raises(TimeoutError):
            _wait(missing, timeout=0.5, proc=proc)
    finally:
        proc.kill()
        proc.wait(timeout=5)
