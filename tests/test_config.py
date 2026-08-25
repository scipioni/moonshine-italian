"""Config loader tests: schema validation, round-trip, profile resolution."""

from pathlib import Path

import yaml

from moonshine_it.config import (
    REPO_ROOT,
    VALID_HW_PROFILES,
    VALID_TRAIN_PROFILES,
    load_config,
    load_env,
    resolve_profile,
)


def test_config_loads_and_validates():
    cfg = load_config()
    assert cfg["base_model"]["id"] == "moonshine-ai/moonshine-streaming-small"


def test_round_trip_preserves_every_key():
    cfg = load_config()
    dumped = yaml.safe_dump(cfg, sort_keys=False)
    reloaded = yaml.safe_load(dumped)
    assert cfg == reloaded, "config must round-trip losslessly through YAML"


def test_all_profiles_resolve():
    cfg = load_config()
    for hw in VALID_HW_PROFILES:
        for tp in VALID_TRAIN_PROFILES:
            rp = resolve_profile(cfg, hw, tp)
            meta = rp.run_metadata()
            assert meta["hardware"] == hw
            assert meta["batch_size"] == cfg["hardware_profiles"][hw]["batch_size"]
            assert rp.gate in cfg["evaluation"]["gates"]


def test_rocm12g_vs_strix_differ_in_batch_size():
    cfg = load_config()
    small = resolve_profile(cfg, "rocm12g", "final")
    big = resolve_profile(cfg, "strix", "final")
    assert small.batch_size < big.batch_size


def test_env_parsing(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("# comment\nHF_TOKEN=abc123\nEMPTY=\n")
    values = load_env(env)
    assert values == {"HF_TOKEN": "abc123", "EMPTY": ""}


def test_paths_are_absolute_and_inside_repo():
    cfg = load_config()
    for key in ("data", "results", "artifacts"):
        from moonshine_it.config import absolute_path

        p = absolute_path(cfg, key)
        assert p.is_absolute() and REPO_ROOT in p.parents
