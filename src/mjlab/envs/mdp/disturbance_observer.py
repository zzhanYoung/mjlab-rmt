"""Momentum disturbance observers for policy observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim_data import TorchArray, WarpBridge
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


ObserverMode = Literal["full_order", "roam"]
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass(frozen=True)
class DisturbanceObserverVizCfg:
  """Viewer settings for disturbance-estimate arrows."""

  force_scale: float = 0.001
  torque_scale: float = 0.005
  joint_scale: float = 0.002
  max_arrow_length: float = 0.4
  top_k_joints: int = 8
  min_magnitude: float = 1.0e-3
  width: float = 0.012


_DEFAULT_VIZ_CFG = DisturbanceObserverVizCfg()


class disturbance_observer:
  r"""Estimate unmodeled generalized forces with a momentum observer.

  The observer follows the formulation in ``research_doc.md``::

      d_hat = xi + L M(q) qdot
      xi_dot = L (q_bias - tau - d_hat)

  Dynamics are evaluated on a private copy of the compile-time nominal model so
  domain-randomized plant parameters do not leak into the estimate. ``roam`` zeros
  the floating-base translational velocity before evaluating the nominal dynamics
  and omits those three generalized forces from its output.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._mode: ObserverMode = params.get("mode", "roam")
    if self._mode not in ("full_order", "roam"):
      raise ValueError(
        f"Unknown disturbance observer mode {self._mode!r}; "
        "expected 'full_order' or 'roam'."
      )

    asset_cfg: SceneEntityCfg = params.get("asset_cfg", SceneEntityCfg("robot"))
    self._asset: Entity = env.scene[asset_cfg.name]
    self._num_envs = env.num_envs
    self._step_dt = env.step_dt
    bandwidth_hz = float(params.get("bandwidth_hz", 5.0))
    if bandwidth_hz <= 0.0:
      raise ValueError("bandwidth_hz must be positive.")
    self._gain = 2.0 * math.pi * bandwidth_hz
    self._estimate_clip = float(params.get("estimate_clip", 1000.0))
    if self._estimate_clip <= 0.0:
      raise ValueError("estimate_clip must be positive.")

    self._debug_vis_enabled = bool(params.get("debug_vis", False))
    self._viz_cfg: DisturbanceObserverVizCfg = params.get(
      "viz_cfg", DisturbanceObserverVizCfg()
    )

    self._free_dof_ids = self._asset.indexing.free_joint_v_adr
    if len(self._free_dof_ids) != 6:
      raise ValueError(
        "disturbance_observer requires one floating-base free joint with 6 DoFs."
      )
    self._joint_dof_ids = self._asset.indexing.joint_v_adr
    if len(self._joint_dof_ids) != self._asset.num_joints:
      raise ValueError(
        "disturbance_observer currently supports one-DoF articulated joints only."
      )
    self._entity_dof_ids = torch.cat((self._free_dof_ids, self._joint_dof_ids), dim=0)
    self._output_dof_ids = (
      self._entity_dof_ids
      if self._mode == "full_order"
      else torch.cat((self._free_dof_ids[3:], self._joint_dof_ids), dim=0)
    )

    root_names = (
      ["base_force_x", "base_force_y", "base_force_z"]
      if self._mode == "full_order"
      else []
    )
    root_names += ["base_torque_x", "base_torque_y", "base_torque_z"]
    self._dof_names = root_names + list(self._asset.joint_names)

    nv = env.sim.mj_model.nv
    self._xi = torch.zeros((self._num_envs, nv), device=env.device)
    self._estimate_full = torch.zeros_like(self._xi)
    self._initialized = torch.zeros(self._num_envs, dtype=torch.bool, device=env.device)

    # Build the estimator from the host-side compile-time model. Startup domain
    # randomization only mutates the plant's per-world Warp model, so this copy stays
    # nominal for the lifetime of the observer.
    with wp.ScopedDevice(env.sim.wp_device):
      self._nominal_model = mjwarp.put_model(env.sim.mj_model)
      self._nominal_model.opt.run_collision_detection = False
      self._nominal_data = mjwarp.make_data(
        env.sim.mj_model,
        nworld=self._num_envs,
        nconmax=1,
        njmax=1,
      )
      self._momentum_wp = wp.zeros(
        (self._num_envs, nv), dtype=float, device=env.sim.wp_device
      )
    self._nominal = WarpBridge(self._nominal_data)
    self._momentum = TorchArray(self._momentum_wp)
    self._env = env

  @property
  def mode(self) -> ObserverMode:
    return self._mode

  @property
  def dof_names(self) -> list[str]:
    return list(self._dof_names)

  @property
  def estimate(self) -> torch.Tensor:
    """Latest policy-facing estimate."""
    return self._estimate_full[:, self._output_dof_ids]

  @property
  def output_dim(self) -> int:
    return len(self._output_dof_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    mode: ObserverMode = "roam",
    bandwidth_hz: float = 5.0,
    estimate_clip: float = 1000.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    debug_vis: bool = False,
    viz_cfg: DisturbanceObserverVizCfg = _DEFAULT_VIZ_CFG,
  ) -> torch.Tensor:
    del mode, bandwidth_hz, estimate_clip, asset_cfg, debug_vis, viz_cfg

    qvel = env.sim.data.qvel.clone()
    if self._mode == "roam":
      qvel[:, self._free_dof_ids[:3]] = 0.0

    self._nominal.qpos[:] = env.sim.data.qpos
    self._nominal.qvel[:] = qvel
    with wp.ScopedDevice(env.sim.wp_device):
      mjwarp.fwd_position(self._nominal_model, self._nominal_data, factorize=False)
      mjwarp.fwd_velocity(self._nominal_model, self._nominal_data)
      mjwarp.mul_m(
        self._nominal_model,
        self._nominal_data,
        self._momentum_wp,
        self._nominal_data.qvel,
      )

    momentum = self._momentum.clone()
    qfrc_bias = self._nominal.qfrc_bias.clone()
    qfrc_actuator = env.sim.data.qfrc_actuator.clone()

    active = self._initialized.clone()
    if active.any():
      d_hat = self._xi[active] + self._gain * momentum[active]
      self._xi[active] += (
        self._step_dt * self._gain * (qfrc_bias[active] - qfrc_actuator[active] - d_hat)
      )
      self._estimate_full[active] = self._xi[active] + self._gain * momentum[active]

    # Initialize xi so the first post-reset estimate is exactly zero instead of
    # producing an artificial momentum spike.
    new = ~active
    if new.any():
      self._xi[new] = -self._gain * momentum[new]
      self._estimate_full[new] = 0.0
      self._initialized[new] = True

    self._estimate_full = torch.nan_to_num(
      self._estimate_full,
      nan=0.0,
      posinf=self._estimate_clip,
      neginf=-self._estimate_clip,
    ).clamp_(-self._estimate_clip, self._estimate_clip)
    return self.estimate

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._xi[env_ids] = 0.0
    self._estimate_full[env_ids] = 0.0
    self._initialized[env_ids] = False

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled:
      return
    cfg = self._viz_cfg
    joint_ids = self._asset.indexing.joint_ids

    for env_idx in visualizer.get_env_indices(self._num_envs):
      if not self._initialized[env_idx]:
        continue
      root_pos = self._asset.data.root_link_pos_w[env_idx].cpu().numpy()
      root_quat = self._asset.data.root_link_quat_w[env_idx]

      if self._mode == "full_order":
        force = self._estimate_full[env_idx, self._free_dof_ids[:3]]
        self._draw_vector(
          visualizer,
          root_pos,
          force.cpu().numpy(),
          cfg.force_scale,
          (0.1, 0.9, 0.2, 0.9),
          "observer/base_force",
        )

      torque_b = self._estimate_full[env_idx, self._free_dof_ids[3:]]
      torque_w = quat_apply(root_quat.unsqueeze(0), torque_b.unsqueeze(0))[0]
      self._draw_vector(
        visualizer,
        root_pos,
        torque_w.cpu().numpy(),
        cfg.torque_scale,
        (0.8, 0.2, 0.9, 0.9),
        "observer/base_torque",
      )

      joint_residual = self._estimate_full[env_idx, self._joint_dof_ids]
      top_k = min(cfg.top_k_joints, len(joint_residual))
      if top_k == 0:
        continue
      top_ids = torch.topk(joint_residual.abs(), k=top_k).indices
      anchors = self._env.sim.data.xanchor[env_idx, joint_ids[top_ids]]
      axes = self._env.sim.data.xaxis[env_idx, joint_ids[top_ids]]
      for rank, local_id in enumerate(top_ids):
        value = joint_residual[local_id]
        vector = axes[rank] * value
        color = (0.9, 0.15, 0.1, 0.9) if value >= 0 else (0.1, 0.3, 0.9, 0.9)
        self._draw_vector(
          visualizer,
          anchors[rank].cpu().numpy(),
          vector.cpu().numpy(),
          cfg.joint_scale,
          color,
          f"observer/{self._asset.joint_names[int(local_id)]}",
        )

  def _draw_vector(
    self,
    visualizer: DebugVisualizer,
    start: np.ndarray,
    vector: np.ndarray,
    scale: float,
    color: tuple[float, float, float, float],
    label: str,
  ) -> None:
    cfg = self._viz_cfg
    magnitude = float(np.linalg.norm(vector))
    if magnitude < cfg.min_magnitude:
      return
    length = min(magnitude * scale, cfg.max_arrow_length)
    end = start + vector / magnitude * length
    visualizer.add_arrow(
      start=start,
      end=end,
      color=color,
      width=cfg.width,
      label=label,
    )


def _get_observer(
  env: ManagerBasedRlEnv,
  group_name: str,
  term_name: str,
) -> disturbance_observer:
  term_cfg = env.observation_manager.get_term_cfg(group_name, term_name)
  if not isinstance(term_cfg.func, disturbance_observer):
    raise TypeError(
      f"Observation term {group_name}/{term_name} is not a disturbance observer."
    )
  return cast(disturbance_observer, term_cfg.func)


def disturbance_estimate_rms(
  env: ManagerBasedRlEnv,
  group_name: str = "actor",
  term_name: str = "disturbance_estimate",
) -> torch.Tensor:
  """Per-environment RMS magnitude of the latest observer output."""
  estimate = _get_observer(env, group_name, term_name).estimate
  return torch.sqrt(torch.mean(torch.square(estimate), dim=-1))


def disturbance_estimate_peak(
  env: ManagerBasedRlEnv,
  group_name: str = "actor",
  term_name: str = "disturbance_estimate",
) -> torch.Tensor:
  """Per-environment peak absolute component of the latest observer output."""
  estimate = _get_observer(env, group_name, term_name).estimate
  return torch.max(torch.abs(estimate), dim=-1).values
