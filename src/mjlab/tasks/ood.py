"""Single-shift out-of-distribution evaluation presets."""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

OodShift = Literal[
  "nominal",
  "payload_5kg",
  "payload_10kg",
  "foot_friction_0.15",
  "foot_friction_1.5",
  "joint_friction_0.5",
  "joint_friction_1.0",
  "impulse_moderate",
  "impulse_severe",
]

OOD_SHIFTS: tuple[OodShift, ...] = (
  "nominal",
  "payload_5kg",
  "payload_10kg",
  "foot_friction_0.15",
  "foot_friction_1.5",
  "joint_friction_0.5",
  "joint_friction_1.0",
  "impulse_moderate",
  "impulse_severe",
)


def configure_ood_evaluation(
  cfg: ManagerBasedRlEnvCfg,
  shift: OodShift,
) -> ManagerBasedRlEnvCfg:
  """Make an evaluation config with exactly one fixed OOD shift.

  Training randomization, observation corruption, pushes, and curricula are disabled
  first. The requested shift is then applied to every evaluation environment.
  """
  if shift not in OOD_SHIFTS:
    raise ValueError(f"Unknown OOD shift {shift!r}; expected one of {OOD_SHIFTS}.")

  foot_friction = cfg.events.pop("foot_friction", None)
  foot_asset_cfg = (
    foot_friction.params["asset_cfg"]
    if foot_friction is not None
    else SceneEntityCfg("robot")
  )
  cfg.events.pop("base_com", None)
  cfg.events.pop("encoder_bias", None)
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  cfg.observations["actor"].enable_corruption = False

  if shift == "nominal":
    return cfg

  if shift.startswith("payload_"):
    mass = 5.0 if shift == "payload_5kg" else 10.0
    cfg.events["ood_payload"] = EventTermCfg(
      mode="startup",
      func=dr.point_mass_payload,
      params={
        "mass_range": (mass, mass),
        "position": (0.0, 0.0, 0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    )
  elif shift.startswith("foot_friction_"):
    coefficient = 0.15 if shift == "foot_friction_0.15" else 1.5
    cfg.events["ood_foot_friction"] = EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": foot_asset_cfg,
        "operation": "abs",
        "ranges": (coefficient, coefficient),
        "shared_random": True,
      },
    )
  elif shift.startswith("joint_friction_"):
    friction = 0.5 if shift == "joint_friction_0.5" else 1.0
    cfg.events["ood_joint_friction"] = EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "abs",
        "ranges": (friction, friction),
        "shared_random": True,
      },
    )
  else:
    force, torque = (100.0, 20.0) if shift == "impulse_moderate" else (200.0, 40.0)
    cfg.events["ood_impulse"] = EventTermCfg(
      mode="step",
      func=envs_mdp.apply_body_impulse,
      params={
        "force_range": (-force, force),
        "torque_range": (-torque, torque),
        "duration_s": (0.08, 0.12),
        "cooldown_s": (2.0, 3.0),
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    )

  return cfg
