from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .config import (
    default_report_dir,
    exp1_metrics_root,
    exp2_results_root,
    load_config,
    published_path,
)
from .io_utils import load_json, markdown_table, portable_path, write_json, write_text
from .stats import common_ordered_values, kendall_tau_b, pearsonr


PUBLISHED_TOLERANCE = 0.0015


def _matches_published(calculated: float, published: float) -> bool:
    # The manuscript mixes rounding and truncation (for example 0.544895 is
    # printed as 0.544), so use a declared tolerance while retaining raw values.
    return abs(calculated - published) <= PUBLISHED_TOLERANCE


def _exp1_score_path(root: Path, relative_metric_path: str, dataset: str) -> Path:
    return root / Path(relative_metric_path) / f"{dataset}_metrics.json"


def verify_exp1(workspace: Path) -> dict[str, Any]:
    config = load_config()
    exp1_config = config["exp1"]
    paper = load_json(published_path("paper_tables.json"))
    accuracy = load_json(published_path("exp1_finetune_accuracy.json"))
    metric_root = exp1_metrics_root(workspace)
    model_order = exp1_config["published_models"]
    dataset_order = paper["dataset_order"]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for metric, relative_path in exp1_config["metric_paths"].items():
        kendall_values: list[float] = []
        pearson_values: list[float] = []
        for dataset_index, dataset in enumerate(dataset_order):
            score_path = _exp1_score_path(metric_root, relative_path, dataset)
            if not score_path.exists():
                rows.append({
                    "metric": metric,
                    "dataset": dataset,
                    "status": "missing",
                    "path": portable_path(score_path, workspace),
                })
                continue
            scores = load_json(score_path)
            keys, score_values, accuracy_values = common_ordered_values(
                scores, accuracy[dataset], model_order
            )
            if len(keys) < 2:
                rows.append({
                    "metric": metric,
                    "dataset": dataset,
                    "status": "insufficient",
                    "model_count": len(keys),
                    "path": portable_path(score_path, workspace),
                })
                continue
            kendall = kendall_tau_b(score_values, accuracy_values)
            pearson = pearsonr(score_values, accuracy_values)
            expected_kendall = paper["table1_kendall"][metric]["values"][dataset_index]
            expected_pearson = paper["table2_pearson"][metric]["values"][dataset_index]
            kendall_values.append(kendall)
            pearson_values.append(pearson)
            rows.append({
                "metric": metric,
                "dataset": dataset,
                "status": "ok",
                "model_count": len(keys),
                "models": keys,
                "kendall": kendall,
                "paper_kendall": expected_kendall,
                "kendall_matches": _matches_published(kendall, expected_kendall),
                "pearson": pearson,
                "paper_pearson": expected_pearson,
                "pearson_matches": _matches_published(pearson, expected_pearson),
                "path": portable_path(score_path, workspace),
            })
        expected_kendall_average = paper["table1_kendall"][metric]["average"]
        expected_pearson_average = paper["table2_pearson"][metric]["average"]
        kendall_average = sum(kendall_values) / len(kendall_values)
        pearson_average = sum(pearson_values) / len(pearson_values)
        summaries.append({
            "metric": metric,
            "dataset_count": len(kendall_values),
            "kendall_average": kendall_average,
            "paper_kendall_average": expected_kendall_average,
            "kendall_average_matches": _matches_published(kendall_average, expected_kendall_average),
            "pearson_average": pearson_average,
            "paper_pearson_average": expected_pearson_average,
            "pearson_average_matches": _matches_published(pearson_average, expected_pearson_average),
            "all_cells_match": all(
                row.get("kendall_matches", False) and row.get("pearson_matches", False)
                for row in rows
                if row.get("metric") == metric
            ),
        })

    return {
        "metric_root": portable_path(metric_root, workspace),
        "model_count_for_published_tables": len(model_order),
        "paper_claimed_model_count": 11,
        "rows": rows,
        "summaries": summaries,
    }


def _exp2_paths(root: Path, strategy: str, model: str, source: str, metric: str) -> tuple[Path, Path]:
    base = root / strategy / model / source
    return base / "metrics" / f"{metric}.json", base / "test_ACC" / f"{metric}_ACC.json"


def verify_exp2(workspace: Path) -> dict[str, Any]:
    config = load_config()
    exp2_config = config["exp2"]
    paper = load_json(published_path("paper_tables.json"))
    result_root = exp2_results_root(workspace)
    rows: list[dict[str, Any]] = []

    for strategy in exp2_config["strategies"]:
        for model in exp2_config["models"]:
            for source in exp2_config["sources"]:
                for metric in exp2_config["metrics"]:
                    score_path, accuracy_path = _exp2_paths(result_root, strategy, model, source, metric)
                    key = f"{strategy}|{model}|{source}|{metric}"
                    expected = paper["table3"][key]
                    if not score_path.exists() or not accuracy_path.exists():
                        rows.append({
                            "key": key,
                            "status": "missing",
                            "score_path": portable_path(score_path, workspace),
                            "accuracy_path": portable_path(accuracy_path, workspace),
                        })
                        continue
                    scores = load_json(score_path)
                    accuracies = load_json(accuracy_path)
                    keys, score_values, accuracy_values = common_ordered_values(scores, accuracies)
                    pearson = pearsonr(score_values, accuracy_values)
                    kendall = kendall_tau_b(score_values, accuracy_values)
                    rows.append({
                        "key": key,
                        "strategy": strategy,
                        "model": model,
                        "source": source,
                        "metric": metric,
                        "status": "ok",
                        "task_count": len(keys),
                        "first_task": keys[0],
                        "last_task": keys[-1],
                        "pearson": pearson,
                        "paper_pearson": expected["pearson"],
                        "pearson_matches": _matches_published(pearson, expected["pearson"]),
                        "kendall": kendall,
                        "paper_kendall": expected["kendall"],
                        "kendall_matches": _matches_published(kendall, expected["kendall"]),
                        "score_path": portable_path(score_path, workspace),
                        "accuracy_path": portable_path(accuracy_path, workspace),
                    })

    valid = [row for row in rows if row.get("status") == "ok"]
    averages: dict[str, dict[str, float]] = {}
    for metric in exp2_config["metrics"]:
        metric_rows = [row for row in valid if row["metric"] == metric]
        averages[metric] = {
            "pearson": sum(row["pearson"] for row in metric_rows) / len(metric_rows),
            "kendall": sum(row["kendall"] for row in metric_rows) / len(metric_rows),
        }

    anomalies = []
    incomplete = [row for row in valid if row["task_count"] != exp2_config["task_count"]]
    for row in incomplete:
        anomalies.append(
            f"{row['key']} contains {row['task_count']} pairs instead of {exp2_config['task_count']}."
        )
    swapped_a = next((row for row in valid if row["key"] == "Finetune|ResNet18|CIFAR10|LEEP"), None)
    swapped_b = next((row for row in valid if row["key"] == "Finetune|ResNet34|CIFAR10|LEEP"), None)
    if swapped_a and swapped_b:
        if (
            _matches_published(swapped_a["kendall"], swapped_b["paper_kendall"])
            and _matches_published(swapped_b["kendall"], swapped_a["paper_kendall"])
        ):
            anomalies.append(
                "The two CIFAR10 Finetune LEEP Kendall values match the opposite ResNet18/ResNet34 rows."
            )

    return {
        "result_root": portable_path(result_root, workspace),
        "rows": rows,
        "averages": averages,
        "anomalies": anomalies,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verification_markdown(report: dict[str, Any]) -> str:
    exp1 = report["exp1"]
    exp2 = report["exp2"]
    lines = [
        "# GLEEP published-result verification",
        "",
        f"Profile: `{report['profile']}`  ",
        f"Correlation: ordinary Kendall tau-b and Pearson correlation. Match tolerance: ±{PUBLISHED_TOLERANCE}.",
        "",
        "## Experiment 1: Tables 1 and 2",
        "",
        markdown_table(
            ["Metric", "Datasets", "Kendall", "Paper", "Pearson", "Paper", "All cells"],
            (
                (
                    row["metric"].upper(),
                    row["dataset_count"],
                    f"{row['kendall_average']:.6f}",
                    f"{row['paper_kendall_average']:.3f}",
                    f"{row['pearson_average']:.6f}",
                    f"{row['paper_pearson_average']:.3f}",
                    "PASS" if row["all_cells_match"] else "DIFF",
                )
                for row in exp1["summaries"]
            ),
        ),
        "",
        "The archived score files contain 10 models per complete dataset, while the manuscript says 11.",
        "The table reports GLEEP Pearson 0.615; the prose reports 0.618.",
        "",
        "## Experiment 2: Table 3",
        "",
        markdown_table(
            ["Strategy", "Model", "Source", "Metric", "Pairs", "Pearson", "Paper", "Kendall", "Paper", "Status"],
            (
                (
                    row.get("strategy", "-"),
                    row.get("model", "-"),
                    row.get("source", "-"),
                    row.get("metric", "-"),
                    row.get("task_count", "-"),
                    f"{row['pearson']:.6f}" if "pearson" in row else "-",
                    f"{row['paper_pearson']:.3f}" if "paper_pearson" in row else "-",
                    f"{row['kendall']:.6f}" if "kendall" in row else "-",
                    f"{row['paper_kendall']:.3f}" if "paper_kendall" in row else "-",
                    (
                        "PASS" if row.get("pearson_matches") and row.get("kendall_matches")
                        else ("DIFF" if row.get("status") == "ok" else row.get("status", "DIFF").upper())
                    ),
                )
                for row in exp2["rows"]
            ),
        ),
        "",
        markdown_table(
            ["Metric", "Average Pearson", "Paper", "Average Kendall", "Paper"],
            (
                (
                    metric,
                    f"{values['pearson']:.6f}",
                    "0.888" if metric == "LEEP" else "0.912",
                    f"{values['kendall']:.6f}",
                    "0.956" if metric == "LEEP" else "0.951",
                )
                for metric, values in exp2["averages"].items()
            ),
        ),
        "",
        "### Detected anomalies",
        "",
    ]
    lines.extend(f"- {item}" for item in exp2["anomalies"])
    lines.extend([
        "- The code iterates from 2 through 100 classes, so the normal count is 99 rather than 100.",
        "- `Retrain` freezes every non-classifier parameter and is therefore linear probing.",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
    ])
    return "\n".join(lines)


def run_verification(workspace: Path, report_dir: Path | None = None, profile: str = "published") -> dict[str, Any]:
    if profile != "published":
        raise ValueError("only the published profile is currently defined")
    exp1 = verify_exp1(workspace)
    exp2 = verify_exp2(workspace)
    core_metrics = {"leep", "logme", "sfda", "gleep"}
    core_rows = [row for row in exp1["summaries"] if row["metric"] in core_metrics]
    core_ok = all(
        row["kendall_average_matches"] and row["pearson_average_matches"]
        for row in core_rows
    )
    exp2_valid = [row for row in exp2["rows"] if row.get("status") == "ok"]
    exp2_match_count = sum(
        bool(row.get("pearson_matches") and row.get("kendall_matches")) for row in exp2_valid
    )
    verdict = (
        "PARTIAL REPRODUCTION: the four main Experiment 1 averages match, and "
        f"{exp2_match_count}/{len(exp2_valid)} Experiment 2 rows match both published rounded coefficients. "
        "Known NLEEP/PAC, task-count, and manuscript inconsistencies remain visible."
        if core_ok
        else "NOT REPRODUCED: one or more core Experiment 1 averages do not match."
    )
    report = {
        "profile": profile,
        "workspace": ".",
        "exp1": exp1,
        "exp2": exp2,
        "verdict": verdict,
    }
    destination = report_dir or default_report_dir()
    write_json(destination / "verification.json", report)
    write_text(destination / "verification.md", _verification_markdown(report))
    _write_csv(
        destination / "table1_table2_recomputed.csv",
        exp1["rows"],
        [
            "metric", "dataset", "status", "model_count", "kendall", "paper_kendall",
            "kendall_matches", "pearson", "paper_pearson", "pearson_matches", "path",
        ],
    )
    _write_csv(
        destination / "table3_recomputed.csv",
        exp2["rows"],
        [
            "strategy", "model", "source", "metric", "status", "task_count", "first_task",
            "last_task", "pearson", "paper_pearson", "pearson_matches", "kendall",
            "paper_kendall", "kendall_matches", "score_path", "accuracy_path",
        ],
    )
    return report
