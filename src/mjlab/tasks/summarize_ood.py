"""Summarize OOD evaluation JSON files across seeds."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tyro

import mjlab


@dataclass(frozen=True)
class SummaryConfig:
  """Inputs for OOD result aggregation."""

  input_files: tuple[str, ...]
  """Evaluation JSON files produced by the velocity or tracking evaluator."""
  output_file: str | None = None
  """Optional path for the combined JSON report."""


def summarize(input_files: tuple[str, ...]) -> list[dict]:
  """Group results by task and shift, then compute seed mean and sample std."""
  grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
  for filename in input_files:
    payload = json.loads(Path(filename).read_text())
    key = (payload["task_id"], payload["ood_shift"])
    grouped[key].append(payload["metrics"])

  report = []
  for (task_id, shift), seed_metrics in sorted(grouped.items()):
    metric_names = set.intersection(*(set(metrics) for metrics in seed_metrics))
    metrics_summary = {}
    for name in sorted(metric_names):
      values = [float(metrics[name]) for metrics in seed_metrics]
      metrics_summary[name] = {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
      }
    report.append(
      {
        "task_id": task_id,
        "ood_shift": shift,
        "num_seeds": len(seed_metrics),
        "metrics": metrics_summary,
      }
    )
  return report


def main() -> None:
  cfg = tyro.cli(SummaryConfig, config=mjlab.TYRO_FLAGS)
  report = summarize(cfg.input_files)
  output = json.dumps(report, indent=2)
  print(output)
  if cfg.output_file:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n")


if __name__ == "__main__":
  main()
