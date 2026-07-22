"""Tests for G1 single-shift OOD evaluation presets."""

import json

import pytest

from mjlab.tasks.ood import OOD_SHIFTS, configure_ood_evaluation
from mjlab.tasks.summarize_ood import summarize
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


@pytest.mark.parametrize("shift", OOD_SHIFTS)
def test_ood_presets_apply_one_shift_and_disable_training_randomization(shift):
  cfg = configure_ood_evaluation(unitree_g1_flat_env_cfg(), shift)

  assert not cfg.observations["actor"].enable_corruption
  assert cfg.curriculum == {}
  assert "foot_friction" not in cfg.events
  assert "base_com" not in cfg.events
  assert "encoder_bias" not in cfg.events
  assert "push_robot" not in cfg.events

  ood_terms = [name for name in cfg.events if name.startswith("ood_")]
  assert len(ood_terms) == (0 if shift == "nominal" else 1)


def test_ood_preset_values() -> None:
  payload = configure_ood_evaluation(unitree_g1_flat_env_cfg(), "payload_10kg").events[
    "ood_payload"
  ]
  assert payload.params["mass_range"] == (10.0, 10.0)
  assert payload.params["position"] == (0.0, 0.0, 0.2)

  friction = configure_ood_evaluation(
    unitree_g1_flat_env_cfg(), "foot_friction_0.15"
  ).events["ood_foot_friction"]
  assert friction.params["ranges"] == (0.15, 0.15)

  impulse = configure_ood_evaluation(
    unitree_g1_flat_env_cfg(), "impulse_severe"
  ).events["ood_impulse"]
  assert impulse.params["force_range"] == (-200.0, 200.0)
  assert impulse.params["torque_range"] == (-40.0, 40.0)
  assert impulse.params["duration_s"] == (0.08, 0.12)
  assert impulse.params["cooldown_s"] == (2.0, 3.0)


def test_summarize_ood_reports_mean_and_sample_std(tmp_path) -> None:
  paths = []
  for seed, value in enumerate((1.0, 2.0, 3.0)):
    path = tmp_path / f"seed_{seed}.json"
    path.write_text(
      json.dumps(
        {
          "task_id": "task",
          "ood_shift": "nominal",
          "seed": seed,
          "metrics": {"return": value},
        }
      )
    )
    paths.append(str(path))

  report = summarize(tuple(paths))
  assert report[0]["num_seeds"] == 3
  assert report[0]["metrics"]["return"] == {"mean": 2.0, "std": 1.0}
