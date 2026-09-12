"""
Command-line interface (CLI) for DQAL.
Provides shift-left dataset validation, pre-commit checking, and CI gates.
"""

from __future__ import annotations
import sys
import os
import json
import argparse
from typing import Optional, List

import dqal
from dqal.standalone import check_quality
from dqal.trigger import QualityDecision


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="dqal",
        description="DQAL - Data-Quality-Aware Learning CLI for local and CI/CD data quality checks.",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {dqal.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: check
    check_parser = subparsers.add_parser(
        "check",
        help="Validate a dataset against an optional baseline distribution",
    )
    check_parser.add_argument(
        "data",
        type=str,
        help="Path to dataset to inspect (.csv, .tsv, .parquet, .json)",
    )
    check_parser.add_argument(
        "-b", "--baseline",
        type=str,
        default=None,
        help="Path to baseline training dataset (.csv, .tsv, .parquet, .json)",
    )
    check_parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to DQAL YAML configuration file",
    )
    check_parser.add_argument(
        "--compact",
        action="store_true",
        help="Display compact single-line metrics instead of full report card",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as a structured JSON object",
    )
    check_parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Treat WARNING status as failure (exit code 1) in addition to BLOCKED",
    )

    # Command: version
    subparsers.add_parser("version", help="Print DQAL version")

    return parser


def run_check(args: argparse.Namespace) -> int:
    """Execute data quality check from CLI arguments and return exit code."""
    if not os.path.exists(args.data):
        print(f"Error: Target data file not found: {args.data}", file=sys.stderr)
        return 1

    if args.baseline and not os.path.exists(args.baseline):
        print(f"Error: Baseline data file not found: {args.baseline}", file=sys.stderr)
        return 1

    if args.config and not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        report = check_quality(
            data=args.data,
            baseline=args.baseline,
            config=args.config,
        )
    except Exception as e:
        print(f"Error during data quality evaluation: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary(compact=args.compact))

    # Exit code determination
    if report.decision == QualityDecision.BLOCKED:
        return 2
    elif report.decision == QualityDecision.WARNING and args.fail_on_warning:
        return 1
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()

    raw_args = list(args) if args is not None else sys.argv[1:]
    if raw_args and not raw_args[0].startswith("-") and raw_args[0] not in ("check", "version"):
        raw_args = ["check"] + raw_args

    parsed_args = parser.parse_args(raw_args)

    if parsed_args.command == "version":
        print(f"dqal {dqal.__version__}")
        return 0
    elif parsed_args.command == "check":
        return run_check(parsed_args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
