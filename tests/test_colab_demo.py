import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_demo", ROOT / "scripts/prepare_colab_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_preview_uses_full_validation_split_and_preserves_order():
    lengths = np.full(2000, 120)
    order = np.random.RandomState(42).permutation(2000)
    lengths[order[1800]] = 99
    chosen = demo.validation_episodes(lengths, 4)
    assert chosen == order[1801:1805].tolist()
    assert not set(chosen).intersection(order[:1800])
    with pytest.raises(ValueError, match="validation episodes"):
        demo.validation_episodes(lengths, 201)


def test_extracts_only_explicit_observation_members_and_reruns(tmp_path):
    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("point_maze/obses/episode_001.pth", b"selected observation")
        z.writestr("point_maze/obses/episode_002.pth", b"unneeded observation")
    output = tmp_path / "data/obses/episode_001.pth"
    with zipfile.ZipFile(archive) as z:
        for _ in range(2):
            demo.extract_member(z, "point_maze/obses/episode_001.pth", output)
    assert output.read_bytes() == b"selected observation"
    assert sorted(p.name for p in output.parent.iterdir()) == ["episode_001.pth"]


def test_notebook_cells_are_plain_python_for_colab_and_local_jupyter():
    notebook = json.loads((ROOT / "colab_quickstart.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
