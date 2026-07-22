from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_flat_tracking_env_cfg
from .rl_cfg import unitree_g1_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

_full_order_rl_cfg = unitree_g1_tracking_ppo_runner_cfg()
_full_order_rl_cfg.experiment_name = "g1_tracking_full_order"
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Full-Order",
  env_cfg=unitree_g1_flat_tracking_env_cfg(observer_mode="full_order"),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True, observer_mode="full_order"),
  rl_cfg=_full_order_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

_roam_rl_cfg = unitree_g1_tracking_ppo_runner_cfg()
_roam_rl_cfg.experiment_name = "g1_tracking_roam"
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-ROAM",
  env_cfg=unitree_g1_flat_tracking_env_cfg(observer_mode="roam"),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True, observer_mode="roam"),
  rl_cfg=_roam_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
