from __future__ import annotations

from pathlib import Path

from .config import default_report_dir
from .exp1 import run_exp1_smoke
from .exp2 import run_exp2_smoke, run_exp2_synthetic_smoke
from .io_utils import markdown_table, write_json, write_text


def run_smoke(
    workspace: Path,
    *,
    device: str,
    data_root: Path,
    seed: int = 42,
    include_exp2: bool = True,
    synthetic_exp2: bool = False,
    disable_cudnn: bool = False,
    report_dir: Path | None = None,
) -> dict[str, object]:
    if disable_cudnn and device.startswith("cuda"):
        import torch

        torch.backends.cudnn.enabled = False
    report: dict[str, object] = {
        "device": device,
        "seed": seed,
        "cudnn_disabled": disable_cudnn,
        "exp1": run_exp1_smoke(workspace, seed=0),
        "exp2": None,
    }
    if include_exp2:
        report["exp2"] = (
            run_exp2_synthetic_smoke(workspace, device, seed=seed)
            if synthetic_exp2
            else run_exp2_smoke(workspace, data_root, device, seed=seed)
        )
    destination = report_dir or default_report_dir()
    write_json(destination / "smoke.json", report)
    exp1_rows = report["exp1"]["records"]
    lines = [
        "# GLEEP smoke test",
        "",
        "## EXP1 cached logits",
        "",
        markdown_table(
            ["Dataset", "Model", "LEEP", "Historical", "Delta", "GLEEP deterministic"],
            (
                (
                    row["dataset"], row["model"], f"{row['leep']:.12f}",
                    f"{row['historical_leep']:.12f}" if row["historical_leep"] is not None else "-",
                    f"{row['leep_absolute_delta']:.3e}" if row["leep_absolute_delta"] is not None else "-",
                    row["gleep_deterministic"],
                )
                for row in exp1_rows
            ),
        ),
        "",
        "GLEEP smoke runs on a real 10-class/32-logit slice with diagonal covariance so it finishes reliably; it is not compared with the full paper score.",
        "",
    ]
    if report["exp2"] is not None:
        lines.extend([
            "## EXP2 source/task smoke",
            "",
            markdown_table(
                ["Model", "Source", "Task", "Samples", "Cluster dim", "Prediction dim", "LEEP", "GLEEP"],
                (
                    (
                        row["model"], row["source"], row["task"], row["sample_count"],
                        row["clustering_dimensions"], row["prediction_dimensions"],
                        f"{row['scores']['LEEP']:.8f}", f"{row['scores']['GLEEP']:.8f}",
                    )
                    for row in report["exp2"]["records"]
                ),
            ),
            "",
            report["exp2"]["note"],
            "",
        ])
    write_text(destination / "smoke.md", "\n".join(lines))
    return report
