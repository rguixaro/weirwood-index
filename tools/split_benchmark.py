from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from weirwood_index.evaluation import load_benchmark
from weirwood_index.training import grouped_scene_folds


def _payload_for_cases(
    source: dict[str, Any],
    case_ids: set[str],
    *,
    fold: int,
    role: str,
) -> dict[str, Any]:
    payload = {key: value for key, value in source.items() if key != "cases"}
    payload["name"] = f"{source['name']}-fold-{fold:02d}-{role}"
    payload["split"] = "development"
    payload["status"] = "draft"
    payload["fold"] = {
        "number": fold,
        "role": role,
        "parent_benchmark": source["name"],
        "grouping": "scene_id",
    }
    payload["cases"] = [case for case in source["cases"] if case["id"] in case_ids]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic scene-grouped benchmark folds."
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source = json.loads(args.queries.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("cases"), list):
        raise SystemExit("scene-grouped splitting requires an object benchmark")
    benchmark = load_benchmark(args.queries)
    validation_folds = grouped_scene_folds(
        benchmark, folds=args.folds, seed=args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    all_ids = {case.id for case in benchmark}
    for fold, validation in enumerate(validation_folds, start=1):
        validation_ids = {case.id for case in validation}
        train_ids = all_ids - validation_ids
        for role, case_ids in (("train", train_ids), ("validation", validation_ids)):
            payload = _payload_for_cases(
                source, case_ids, fold=fold, role=role
            )
            path = args.output / f"fold-{fold:02d}-{role}.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"{path}: {len(payload['cases'])} cases")


if __name__ == "__main__":
    main()
