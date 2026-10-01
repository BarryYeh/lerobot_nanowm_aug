"""`configs/sailor_v1_reset.yaml`: expansion, pre-registered numbers and namespace pins (M10 / O10)."""

from __future__ import annotations

import pytest
from config_io import load_config

from conftest import REPO

CFG_PATH = REPO / "configs" / "sailor_v1_reset.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CFG_PATH, out_dir_copy=False)


def test_out_dir_is_expanded_and_under_datastorage(cfg):
    assert "~" not in cfg["out_dir"]
    assert cfg["out_dir"].endswith("datastorage/nanowm_pusht_aug/value_model/sailor_v1")
    assert "outputs" not in cfg["out_dir"].split("/")[-3:]


def test_stage1_design(cfg):
    s1 = cfg["stage1"]
    assert s1["episode_steps"] == 300
    assert s1["band"]["n"] == 0 and "seed" not in s1["band"]
    assert s1["reset"]["n"] == 2000
    assert s1["query_keep"] in {"all", "after_switch"}
    assert s1["query_ts"] == list(range(10, 101, 10))
    assert s1["episode_steps"] >= max(s1["query_ts"]) + cfg["label"]["B"]
    pr = s1["perturb"]
    assert pr["mode"] == "prefix" and pr["switch"] in {"all", "random"}
    assert pr["noise_hold"] == 1 and pr["sigmas"] == [0.1] and pr["probs"] == [1.0] and pr["max_z"] == 3.0


def test_seeds_equal_the_common_namespace_and_are_pairwise_distinct(cfg):
    ns = cfg["seed_namespace"]
    reset, anchors, test = (
        cfg["stage1"]["reset"]["seed"],
        cfg["stage2"]["anchors"]["seed"],
        cfg["test_anchors"]["seed"],
    )
    assert reset == ns["sailor_v1_stage1_reset"] == 730000
    assert anchors == ns["sailor_v1_stage2_anchor"] == 740000
    assert test == ns["sailor_v1_test_anchor"] == 750000
    assert len({reset, anchors, test}) == 3
    assert cfg["label"]["cont_seed"] == ns["sailor_v1_label_cont"] == 770000
    assert cfg["label"]["noise_ceiling"]["cont_seed"] == ns["sailor_v1_label_cont"] + 1
    assert cfg["stage1"]["policy_seed"] == reset
    assert (
        cfg["stage1"]["perturb"]["seed"] == reset and cfg["stage2"]["anchors"]["perturb"]["seed"] == anchors
    )
    assert cfg["stage2"]["torch_seed"] == anchors and cfg["test_anchors"]["torch_seed"] == test
    # +1 is the anchor source rollouts' action noise and +2 their anchor-time draws: the driver uses +3
    assert cfg["stage2"]["noise_seed"] == anchors + 3 and cfg["test_anchors"]["noise_seed"] == test + 3
    nc = cfg["label"]["noise_ceiling"]
    assert nc["sample_seed"] not in {cfg["label"]["cont_seed"] + 1, nc["cont_seed"] + 1}
    others = {v for k, v in ns.items() if not k.startswith("sailor_v1")}
    assert not {reset, anchors, test} & others
    # sampling of the Table D anchors / T2 checks live in their own namespaces
    assert len({reset, anchors, test, ns["sailor_v1_t2_checks"], ns["sailor_v1_label_cont"]}) == 5


def test_anchors_use_stage1_noise_settings(cfg):
    """D2: anchor source rollouts use the identical noise as stage 1 (only the seed namespace differs)."""
    a = {k: v for k, v in cfg["stage2"]["anchors"]["perturb"].items() if k != "seed"}
    s = {k: v for k, v in cfg["stage1"]["perturb"].items() if k != "seed"}
    assert a == s
    assert set(cfg["stage2"]["anchors"]["t_choices"]) == set(cfg["stage1"]["query_ts"])
    cp = cfg["label"]["cont_perturb"]  # T2 backup: continuation noised with the same sigma / hold
    assert cp["sigma"] == cfg["stage1"]["perturb"]["sigmas"][0] and cp["noise_hold"] == 1


def test_stage2_design(cfg):
    s2 = cfg["stage2"]
    assert s2["h_max"] % s2["replan_every_rounds"] == 0
    assert all(r % s2["replan_every_rounds"] == 0 for r in s2["query_rounds"])
    assert s2["inject_sigma"][0] == 0.0  # sigma 0.0 first: its lead_in.npz is replayed by the others
    assert s2["lead_na"] == cfg["label"]["na"] == 10 and s2["k"] == 4
    assert s2["noise_hold"] == 1 and s2["anchors"]["n"] == 250 and cfg["test_anchors"]["n"] == 100


def test_gate_is_empty_and_train_has_no_confusing_keys(cfg):
    assert cfg["gate"] == {}  # Opus fills it in T5; the report must fail closed meanwhile
    assert set(cfg["train"]) >= {"lr", "wd", "batch", "max_epochs", "patience", "mix_stage1", "split_seed"}
    assert cfg["label"]["batch"] == 100 and cfg["label"]["B"] == 200 and cfg["label"]["gamma"] == 0.995
    assert cfg["smoke"]["test_anchors"] >= 1


def test_every_key_read_by_run_value_model_sh_exists(cfg):
    sh = (REPO / "examples/nanowm_pusht_aug/scripts/run_value_model.sh").read_text()
    import re

    keys = set(re.findall(r'\by(?: |\s)"?\$?\(?([a-z_0-9]+(?:\.[a-z_0-9]+)+)\)?', sh))
    keys |= set(re.findall(r"\$\(y ([a-z_0-9.]+)\)", sh))
    assert {"stage2.torch_seed", "test_anchors.noise_seed", "encode.batch", "smoke.test_anchors"} <= keys
    for key in sorted(keys):
        if key.startswith("ckpt."):
            continue
        node = cfg
        for part in key.split("."):
            assert part in node, f"run_value_model.sh reads {key!r} but the config has no {part!r}"
            node = node[part]
