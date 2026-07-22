"""Evaluate a velocity-tracking policy under one fixed OOD shift."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.ood import OodShift, configure_ood_evaluation
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for velocity-policy evaluation."""

  wandb_run_path: str
  wandb_checkpoint_name: str | None = None
  num_envs: int = 1024
  device: str | None = None
  output_file: str | None = None
  log_root: str = "logs/rsl_rl"
  ood_shift: OodShift = "nominal"
  seed: int = 42


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, float]:
  """Evaluate one checkpoint and return aggregate episode metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = configure_ood_evaluation(load_env_cfg(task_id, play=False), cfg.ood_shift)
  agent_cfg = load_rl_cfg(task_id)
  if "twist" not in env_cfg.commands:
    raise ValueError(f"Task {task_id} is not a velocity-tracking task.")
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
  resume_path, _ = get_wandb_checkpoint_path(
    log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot: Entity = env.scene["robot"]
  actor_terms = env.observation_manager.active_terms["actor"]
  observer = None
  if "disturbance_estimate" in actor_terms:
    observer = env.observation_manager.get_term_cfg(
      "actor", "disturbance_estimate"
    ).func

  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  active_steps = torch.zeros(cfg.num_envs, device=device)
  episode_return = torch.zeros(cfg.num_envs, device=device)
  linear_sq_sum = torch.tensor(0.0, device=device)
  yaw_sq_sum = torch.tensor(0.0, device=device)
  velocity_samples = 0
  residual_sq_sum = torch.tensor(0.0, device=device)
  residual_count = 0
  residual_peak = torch.tensor(0.0, device=device)

  obs = wrapped.get_observations()
  print(f"[INFO] Running {cfg.num_envs} episodes with OOD shift {cfg.ood_shift!r}...")
  while not done_envs.all():
    command = env.command_manager.get_command("twist")
    assert command is not None
    command = command.clone()
    with torch.no_grad():
      actions = policy(obs)
    obs, rewards, dones, _ = wrapped.step(actions)

    active = ~done_envs
    n_active = int(active.sum().item())
    actual_linear = robot.data.root_link_lin_vel_b[:, :2]
    actual_yaw = robot.data.root_link_ang_vel_b[:, 2]
    linear_sq_sum += torch.sum(
      torch.square(command[active, :2] - actual_linear[active])
    )
    yaw_sq_sum += torch.sum(torch.square(command[active, 2] - actual_yaw[active]))
    velocity_samples += n_active
    active_steps += active.float()
    episode_return += torch.where(active, rewards, 0.0)

    if observer is not None:
      estimate = observer.estimate[active]
      residual_sq_sum += torch.sum(torch.square(estimate))
      residual_count += estimate.numel()
      if estimate.numel() > 0:
        residual_peak = torch.maximum(residual_peak, torch.max(torch.abs(estimate)))

    done_envs |= dones.bool() & ~done_envs

  metrics = {
    "linear_velocity_rmse": torch.sqrt(
      linear_sq_sum / max(2 * velocity_samples, 1)
    ).item(),
    "yaw_rate_rmse": torch.sqrt(yaw_sq_sum / max(velocity_samples, 1)).item(),
    "return": episode_return.mean().item(),
    "survival_rate": (active_steps / env.max_episode_length).mean().item(),
  }
  if observer is not None:
    metrics["residual_rms"] = torch.sqrt(
      residual_sq_sum / max(residual_count, 1)
    ).item()
    metrics["residual_peak"] = residual_peak.item()

  for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

  if cfg.output_file:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
      json.dump(
        {
          "task_id": task_id,
          "ood_shift": cfg.ood_shift,
          "seed": cfg.seed,
          "metrics": metrics,
        },
        f,
        indent=2,
      )
    print(f"[INFO] Metrics saved to {output_path}")

  wrapped.close()
  return metrics


def main() -> None:
  import mjlab.tasks  # noqa: F401

  velocity_tasks = [task for task in list_tasks() if "Velocity-Flat-Unitree-G1" in task]
  if not velocity_tasks:
    raise RuntimeError("No flat G1 velocity tasks are registered.")
  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(velocity_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_evaluate(task_id, args)


if __name__ == "__main__":
  main()
