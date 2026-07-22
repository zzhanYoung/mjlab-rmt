"""Tests for the full-order and ROAM disturbance observers."""

from typing import cast
from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.disturbance_observer import disturbance_observer
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="2.0"/>
      <body name="link" pos="0 0 -0.2">
        <joint name="joint" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom name="link_geom" type="box" size="0.05 0.05 0.2" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="joint" gear="1.0"/>
  </actuator>
</mujoco>
"""


def _make_env(mode: str, debug_vis: bool = False) -> ManagerBasedRlEnv:
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=2,
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "disturbance_estimate": ObservationTermCfg(
            func=mdp.disturbance_observer,
            params={"mode": mode, "debug_vis": debug_vis},
          )
        }
      )
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=2,
    episode_length_s=1.0,
  )
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


@pytest.mark.parametrize(("mode", "output_dim"), [("full_order", 7), ("roam", 4)])
def test_disturbance_observer_shape_reset_and_finite(mode, output_dim):
  env = _make_env(mode)
  try:
    obs, _ = env.reset()
    estimate = cast(torch.Tensor, obs["actor"])
    assert estimate.shape == (2, output_dim)
    assert torch.count_nonzero(estimate) == 0
    metadata = get_base_metadata(env, "test")
    assert metadata["disturbance_observer_mode"] == mode
    dof_names = metadata["disturbance_observer_dof_names"]
    assert isinstance(dof_names, list)
    assert len(dof_names) == output_dim

    env.sim.data.qvel[:, :3] = torch.tensor(
      [[1.0, -0.5, 0.25], [-0.25, 0.5, -1.0]], device=env.device
    )
    env.sim.forward()
    estimate = cast(
      torch.Tensor,
      env.observation_manager.compute(update_history=True)["actor"],
    )
    assert torch.isfinite(estimate).all()

    env.observation_manager.reset(torch.tensor([0], device=env.device))
    estimate = cast(
      torch.Tensor,
      env.observation_manager.compute(update_history=True)["actor"],
    )
    torch.testing.assert_close(estimate[0], torch.zeros_like(estimate[0]))
  finally:
    env.close()


def test_roam_nominal_dynamics_zero_base_translation():
  env = _make_env("roam")
  try:
    env.reset()
    env.sim.data.qvel[:, :3] = 3.0
    env.observation_manager.compute(update_history=True)
    term_cfg = env.observation_manager.get_term_cfg("actor", "disturbance_estimate")
    assert isinstance(term_cfg.func, disturbance_observer)
    nominal_qvel = term_cfg.func._nominal.qvel  # noqa: SLF001
    torch.testing.assert_close(
      nominal_qvel[:, :3], torch.zeros_like(nominal_qvel[:, :3])
    )
  finally:
    env.close()


def test_observer_visualization_emits_arrows():
  env = _make_env("full_order", debug_vis=True)
  try:
    env.reset()
    env.sim.data.qvel[:, 0] = 1.0
    env.sim.forward()
    env.observation_manager.compute(update_history=True)
    visualizer = Mock()
    visualizer.get_env_indices.return_value = [0]
    env.observation_manager.debug_vis(visualizer)
    assert visualizer.add_arrow.call_count > 0
  finally:
    env.close()
