from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    default_exp2_cache_dir,
    default_report_dir,
    load_config,
    runtime_data_root,
    workspace_root,
)


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(
        prog="python -m gleep_repro",
        description="Reproduce and audit the two main GLEEP experiments.",
    )
    parser.add_argument(
        "--workspace",
        help="GLEEP repository root (or set GLEEP_WORKSPACE_ROOT)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="audit artifacts, code health and environment")
    audit.add_argument("--report-dir", type=Path, default=default_report_dir())

    verify = commands.add_parser("verify", help="recompute Tables 1-3 from archived JSON")
    verify.add_argument("--profile", choices=["published"], default="published")
    verify.add_argument("--report-dir", type=Path, default=default_report_dir())

    smoke = commands.add_parser("smoke", help="run deterministic sampled score checks")
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--seed", type=int, default=config["exp2"]["seed"])
    smoke.add_argument("--data-root", type=Path)
    smoke.add_argument("--skip-exp2", action="store_true")
    smoke.add_argument(
        "--synthetic-exp2",
        action="store_true",
        help="validate EXP2 model/checkpoint/GPU wiring without downloading CIFAR100",
    )
    smoke.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="work around a broken local cuDNN installation during smoke only",
    )
    smoke.add_argument("--report-dir", type=Path, default=default_report_dir())

    run = commands.add_parser("run", help="run a recomputation or streaming training stage")
    experiments = run.add_subparsers(dest="experiment", required=True)

    exp1 = experiments.add_parser("exp1", help="recompute EXP1 GLEEP/LEEP")
    exp1.add_argument("--source", choices=["cached", "stream"], default="cached")
    exp1.add_argument("--metrics", nargs="+", default=["gleep", "leep"])
    exp1.add_argument("--models", nargs="+")
    exp1.add_argument("--datasets", nargs="+")
    exp1.add_argument("--seed", type=int, default=0)
    exp1.add_argument("--covariance-type", choices=["full", "tied", "diag", "spherical"], default="full")
    exp1.add_argument("--formula", choices=["legacy", "canonical"], default="legacy")
    exp1.add_argument("--batch-size", type=int, default=128)
    exp1.add_argument("--device", default="cuda")
    exp1.add_argument("--output-dir", type=Path)

    exp2 = experiments.add_parser("exp2", help="cache outputs, recompute scores, or stream training")
    exp2.add_argument("--mode", choices=["cache", "score", "train"], default="score")
    exp2.add_argument("--models", nargs="+", default=config["exp2"]["models"])
    exp2.add_argument("--sources", nargs="+", default=config["exp2"]["sources"])
    exp2.add_argument("--strategies", nargs="+", default=config["exp2"]["strategies"])
    exp2.add_argument("--metrics", nargs="+", default=config["exp2"]["metrics"])
    exp2.add_argument("--tasks", default="2:100")
    exp2.add_argument("--device", default="cuda")
    exp2.add_argument("--data-root", type=Path)
    exp2.add_argument("--seed", type=int, default=config["exp2"]["seed"])
    exp2.add_argument("--random-state", type=int, default=config["exp2"]["gmm_random_state"])
    exp2.add_argument("--covariance-type", choices=["full", "tied", "diag", "spherical"], default="full")
    exp2.add_argument("--semantics", choices=["published", "corrected"], default="published")
    exp2.add_argument("--formula", choices=["legacy", "canonical"], default="legacy")
    exp2.add_argument("--epochs", type=int, default=config["exp2"]["epochs"])
    exp2.add_argument("--batch-size-train", type=int, default=config["exp2"]["batch_size_train"])
    exp2.add_argument("--batch-size-test", type=int, default=config["exp2"]["batch_size_test"])
    exp2.add_argument("--learning-rate", type=float, default=config["exp2"]["learning_rate"])
    exp2.add_argument("--retain-checkpoints", choices=["none", "best"], default="none")
    exp2.add_argument("--cache-dir", type=Path, default=default_exp2_cache_dir())
    exp2.add_argument(
        "--score-input",
        choices=["forward", "cache"],
        default="forward",
        help="for score mode, run source-model forward or reuse --cache-dir",
    )
    exp2.add_argument("--output-dir", type=Path)

    package = commands.add_parser("package", help="build a minimal online-server archive")
    package.add_argument("--profile", choices=["server-minimal"], default="server-minimal")
    package.add_argument("--output", type=Path)

    artifacts = commands.add_parser("artifacts", help="audit or fetch ignored binary artifacts")
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", required=True)
    artifact_audit = artifact_commands.add_parser(
        "audit", help="verify local files against artifacts/index.json"
    )
    artifact_audit.add_argument("--verbose", action="store_true")
    fetch = artifact_commands.add_parser("fetch", help="download configured artifacts and verify SHA256")
    fetch.add_argument(
        "--profile",
        choices=["exp1", "exp2", "source-checkpoints", "all"],
        default="exp2",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    workspace = workspace_root(args.workspace)
    try:
        if args.command == "audit":
            from .audit import run_audit
            report = run_audit(workspace, args.report_dir)
            print(json.dumps({
                "report": str(args.report_dir / "audit.md"),
                "source_problems": len(report["source_problems"]),
                "torch_numpy_ok": report["environment"]["torch_numpy"]["ok"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "verify":
            from .verify import run_verification
            report = run_verification(workspace, args.report_dir, args.profile)
            print(report["verdict"])
            print(args.report_dir / "verification.md")
        elif args.command == "smoke":
            from .smoke import run_smoke
            data_root = args.data_root or runtime_data_root()
            run_smoke(
                workspace,
                device=args.device,
                data_root=data_root,
                seed=args.seed,
                include_exp2=not args.skip_exp2,
                synthetic_exp2=args.synthetic_exp2,
                disable_cudnn=args.disable_cudnn,
                report_dir=args.report_dir,
            )
            print(args.report_dir / "smoke.md")
        elif args.command == "run" and args.experiment == "exp1":
            from .exp1 import run_exp1
            result = run_exp1(
                workspace,
                source=args.source,
                metrics=args.metrics,
                models=args.models,
                datasets=args.datasets,
                seed=args.seed,
                covariance_type=args.covariance_type,
                formula=args.formula,
                batch_size=args.batch_size,
                device=args.device,
                output_dir=args.output_dir,
            )
            print(f"completed {len(result['records'])} EXP1 model/dataset records")
        elif args.command == "run" and args.experiment == "exp2":
            from .exp2 import parse_tasks, run_exp2_cache, run_exp2_scores, run_exp2_training
            tasks = parse_tasks(args.tasks)
            data_root = args.data_root or runtime_data_root()
            if args.mode == "cache":
                result = run_exp2_cache(
                    workspace,
                    models=args.models,
                    sources=args.sources,
                    data_root=data_root,
                    device=args.device,
                    seed=args.seed,
                    semantics=args.semantics,
                    batch_size=args.batch_size_test,
                    cache_dir=args.cache_dir,
                )
            elif args.mode == "score":
                result = run_exp2_scores(
                    workspace,
                    models=args.models,
                    sources=args.sources,
                    tasks=tasks,
                    metrics=args.metrics,
                    strategies=args.strategies,
                    data_root=data_root,
                    device=args.device,
                    seed=args.seed,
                    random_state=args.random_state,
                    covariance_type=args.covariance_type,
                    semantics=args.semantics,
                    formula=args.formula,
                    batch_size=args.batch_size_test,
                    output_dir=args.output_dir,
                    logits_cache_dir=args.cache_dir if args.score_input == "cache" else None,
                )
            else:
                result = run_exp2_training(
                    workspace,
                    models=args.models,
                    sources=args.sources,
                    strategies=args.strategies,
                    tasks=tasks,
                    data_root=data_root,
                    device=args.device,
                    seed=args.seed,
                    semantics=args.semantics,
                    epochs=args.epochs,
                    batch_size_train=args.batch_size_train,
                    batch_size_test=args.batch_size_test,
                    learning_rate=args.learning_rate,
                    retain_checkpoints=args.retain_checkpoints,
                    output_dir=args.output_dir,
                )
            print(f"completed {len(result['records'])} EXP2 records")
        elif args.command == "package":
            from .package_builder import build_server_package
            report = build_server_package(
                workspace,
                destination=args.output,
                profile=args.profile,
            )
            print(json.dumps({
                "archive": report["archive"],
                "size": report["archive_size"],
                "sha256": report["archive_sha256"],
            }, indent=2))
        elif args.command == "artifacts":
            from .artifacts import audit_artifacts, fetch_artifacts
            report = (
                audit_artifacts(workspace)
                if args.artifact_command == "audit"
                else fetch_artifacts(workspace, args.profile)
            )
            if args.artifact_command == "audit" and not args.verbose:
                report = {
                    key: report[key]
                    for key in ("index", "entry_count", "verified", "missing_or_invalid", "complete")
                }
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:  # pragma: no cover
            parser.error("unsupported command")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
