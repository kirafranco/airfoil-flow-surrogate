"""Evaluate runs and baselines, and build the comparison table.

    python -m src.evaluate baseline free_stream --split test
    python -m src.evaluate baseline thin_airfoil --split test
    python -m src.evaluate run <run_id> --split test     # once per run; refused the second time
    python -m src.evaluate report                        # data/models/comparison_test.{csv,md}

In the container, prefix with `docker compose --profile ml run --rm ml`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from src.config import MODELS_ROOT
from src.dataset.airfrans import Dataset
from src.evaluate.report import write_report
from src.evaluate.runs import SPLITS, DatasetMismatch, RepeatedTestEvaluation, baseline_run, evaluate_run
from src.models.baselines import BASELINES

logger = logging.getLogger("src.evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="evaluate a trained run")
    run.add_argument("run_id")
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--device", default="cuda")
    baseline = commands.add_parser("baseline", help="evaluate a baseline")
    baseline.add_argument("name", choices=sorted(BASELINES))
    baseline.add_argument("--split", choices=SPLITS, required=True)
    report = commands.add_parser("report", help="write the comparison table")
    report.add_argument("--split", choices=SPLITS, default="test")
    args = parser.parse_args(argv)

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(name)s %(message)s", stream=sys.stdout)

    if args.command == "report":
        csv_path, md_path = write_report(MODELS_ROOT, args.split)
        print(md_path.read_text(encoding="utf-8"))
        logger.info("report_written csv=%s md=%s", csv_path, md_path)
        return 0

    dataset = Dataset.open()
    try:
        if args.command == "baseline":
            evaluate_run(baseline_run(args.name, dataset, MODELS_ROOT), dataset, args.split, device="cpu")
        else:
            evaluate_run(MODELS_ROOT / args.run_id, dataset, args.split, device=args.device)
    except (RepeatedTestEvaluation, DatasetMismatch) as exc:
        logger.error("evaluation_refused error=%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
