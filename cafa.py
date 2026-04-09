from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sympy as sp

from benchmark_loader import BenchmarkInstance, ORBenchmarkLoader
from lmstudio_client import LMStudioCAFAClient, LMStudioConfig

try:
    import gurobipy as gp
except Exception:  # pragma: no cover
    gp = None


GUROBI_PREFIX = """
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
m = gp.Model(env=env)
""".strip()

GUROBI_SUFFIX = """
m.optimize()
""".strip()


@dataclass
class RunConfig:
    dataset_root: str
    datasets: Optional[List[str]] = None
    output_dir: str = "output/benchmarks"
    max_instances: Optional[int] = None
    start_index: int = 0
    stop_index: Optional[int] = None
    skip_solve: bool = False
    save_prompts: bool = False


def clean_code(code: str) -> str:
    temp_code = re.sub(r"\)([a-zA-Z])", r")\n\1", code.strip())

    cleaned_lines: List[str] = []
    blocked_prefixes = ("```", "#", "import ", "from ", "env =", "m = gp.Model", "m.optimize")

    for raw_line in temp_code.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(blocked_prefixes):
            continue
        if line.startswith("m.addConstr") and not re.search(r"<=|>=", line):
            line = re.sub(r"<", "<=", line)
            line = re.sub(r">", ">=", line)
        if "==" in line:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).replace("{", "").replace("}", "")


def simplify_code(code: str) -> str:
    simplified_lines: List[str] = []

    for line in code.splitlines():
        if not (line.startswith("m.addConstr") or line.startswith("m.setObjective")) or "/" not in line:
            simplified_lines.append(line)
            continue

        obj_match = re.findall(r"m\.setObjective\(([^,]*)", line)
        if obj_match:
            expr = sp.sympify(obj_match[0])
            obj_mode_match = re.search(r"gp\.GRB\.(\w+)", line)
            if obj_mode_match:
                simplified_lines.append(f"m.setObjective({sp.simplify(expr)}, gp.GRB.{obj_mode_match.group(1)})")
                continue

        constr_match = re.findall(r"m\.addConstr\((.*)\)", line)
        if constr_match:
            expr_text = constr_match[0]
            op_match = re.search(r"\s*(>=|<=)\s*", expr_text)
            if op_match:
                expr = sp.sympify(expr_text)
                simplified_expr = str(sp.simplify(expr.lhs - expr.rhs))
                frac_match = re.search(r"^\((.*?)\)/", simplified_expr)
                if frac_match:
                    simplified_lines.append(f"m.addConstr({frac_match.group(1)} {op_match.group(1)} 0)")
                    continue

        simplified_lines.append(line)

    return "\n".join(simplified_lines)


def complement_code(code: str) -> str:
    return f"{GUROBI_PREFIX}\n{code}\n{GUROBI_SUFFIX}\n"


def execute_code(code: str) -> float:
    if gp is None:
        raise RuntimeError("gurobipy is not installed in this environment.")
    ex_locals: Dict[str, Any] = {}
    exec(code, None, ex_locals)
    return float(ex_locals["m"].objVal)


def extract_scalar_answer(answer: Any) -> Optional[float]:
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        return float(answer)
    if isinstance(answer, list):
        for item in answer:
            scalar = extract_scalar_answer(item)
            if scalar is not None:
                return scalar
        return None
    if isinstance(answer, dict):
        scalar_keys = ["objective", "obj", "objective_value", "optimal_value", "answer", "output", "value"]
        for key in scalar_keys:
            if key in answer:
                scalar = extract_scalar_answer(answer[key])
                if scalar is not None:
                    return scalar
        for value in answer.values():
            scalar = extract_scalar_answer(value)
            if scalar is not None:
                return scalar
        return None
    if isinstance(answer, str):
        matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", answer)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
    return None


def nearly_equal(pred: float, gold: float, rel_tol: float = 1e-4) -> bool:
    if math.isinf(pred) and math.isinf(gold):
        return True
    if gold == 0:
        return abs(pred) < rel_tol
    return abs(pred - gold) / abs(gold) < rel_tol


def prepare_instances(loader: ORBenchmarkLoader, config: RunConfig) -> List[BenchmarkInstance]:
    instances = list(loader.iter_instances(config.datasets))
    start = max(0, config.start_index)
    stop = config.stop_index if config.stop_index is not None else len(instances)
    selected = instances[start:stop]
    if config.max_instances is not None:
        selected = selected[: config.max_instances]
    return selected


def run_instance(instance: BenchmarkInstance, client: LMStudioCAFAClient, config: RunConfig) -> Dict[str, Any]:
    response = client.generate_code(instance)
    cleaned_code = clean_code(response.code)
    simplified_code = simplify_code(cleaned_code)

    record: Dict[str, Any] = {
        "dataset_name": instance.dataset_name,
        "instance_id": instance.instance_id,
        "question": instance.question,
        "expected_answer": instance.expected_answer,
        "answer_type": instance.answer_type,
        "problem_type": instance.problem_type,
        "difficulty": instance.difficulty,
        "metadata": instance.metadata,
        "prompt": response.prompt if config.save_prompts else None,
        "raw_text": response.raw_text,
        "raw_code": response.code,
        "clean_code": cleaned_code,
        "simplified_code": simplified_code,
    }

    if config.skip_solve:
        record["prediction"] = None
        record["gold_scalar"] = extract_scalar_answer(instance.expected_answer)
        record["correct"] = False
        return record

    executable_code = complement_code(simplified_code)
    prediction = execute_code(executable_code)
    gold = extract_scalar_answer(instance.expected_answer)

    record["prediction"] = prediction
    record["gold_scalar"] = gold
    record["correct"] = (gold is not None) and nearly_equal(prediction, gold)
    return record


def run_cafa(instances: Sequence[BenchmarkInstance], client: LMStudioCAFAClient, config: RunConfig) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for idx, instance in enumerate(instances, start=1):
        try:
            result = run_instance(instance, client, config)
            logging.info("[%d/%d] %s :: %s done", idx, len(instances), instance.dataset_name, instance.instance_id)
        except Exception as exc:
            result = {
                "dataset_name": instance.dataset_name,
                "instance_id": instance.instance_id,
                "question": instance.question,
                "expected_answer": instance.expected_answer,
                "answer_type": instance.answer_type,
                "prediction": None,
                "gold_scalar": extract_scalar_answer(instance.expected_answer),
                "correct": False,
                "error": str(exc),
            }
            logging.error("[%d/%d] %s :: %s failed: %s", idx, len(instances), instance.dataset_name, instance.instance_id, exc)
        results.append(result)

    return results


def summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    solved = sum(item.get("prediction") is not None for item in results)
    evaluable = sum(item.get("gold_scalar") is not None for item in results)
    correct = sum(bool(item.get("correct")) for item in results)

    per_dataset: Dict[str, Dict[str, Any]] = {}
    for item in results:
        name = item.get("dataset_name", "unknown")
        stats = per_dataset.setdefault(name, {"total": 0, "solved": 0, "evaluable": 0, "correct": 0})
        stats["total"] += 1
        stats["solved"] += int(item.get("prediction") is not None)
        stats["evaluable"] += int(item.get("gold_scalar") is not None)
        stats["correct"] += int(bool(item.get("correct")))

    for stats in per_dataset.values():
        stats["accuracy"] = (stats["correct"] / stats["evaluable"]) if stats["evaluable"] else 0.0

    return {
        "total": total,
        "solved": solved,
        "evaluable": evaluable,
        "correct": correct,
        "accuracy": (correct / evaluable) if evaluable else 0.0,
        "per_dataset": per_dataset,
    }


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_outputs(results: Sequence[Dict[str, Any]], summary: Dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    json_path = output_dir / f"cafa_benchmark_results_{timestamp}.json"
    pkl_path = output_dir / f"cafa_benchmark_results_{timestamp}.pkl"
    summary_path = output_dir / f"cafa_benchmark_summary_{timestamp}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(results), f, ensure_ascii=False, indent=2)
    save_pickle(list(results), pkl_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Saved results to %s", output_dir)


def parse_args() -> tuple[RunConfig, LMStudioConfig]:
    parser = argparse.ArgumentParser(description="Run CAFA on unified LLM4OR benchmark datasets")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the benchmark root folder")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names to run, e.g. NL4Opt IndustryOR")
    parser.add_argument("--output-dir", type=str, default=RunConfig.output_dir)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=None)
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")

    parser.add_argument("--model-name", type=str, default=LMStudioConfig.model_name)
    parser.add_argument("--base-url", type=str, default=LMStudioConfig.base_url)
    parser.add_argument("--api-key", type=str, default=LMStudioConfig.api_key)
    parser.add_argument("--temperature", type=float, default=LMStudioConfig.temperature)
    parser.add_argument("--timeout", type=int, default=LMStudioConfig.timeout)
    parser.add_argument("--max-retries", type=int, default=LMStudioConfig.max_retries)

    args = parser.parse_args()

    run_config = RunConfig(
        dataset_root=args.dataset_root,
        datasets=args.datasets,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
        start_index=args.start_index,
        stop_index=args.stop_index,
        skip_solve=args.skip_solve,
        save_prompts=args.save_prompts,
    )
    lm_config = LMStudioConfig(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    return run_config, lm_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_config, lm_config = parse_args()

    loader = ORBenchmarkLoader(run_config.dataset_root)
    instances = prepare_instances(loader, run_config)
    logging.info("Loaded %d instances from datasets: %s", len(instances), run_config.datasets or loader.available_datasets())

    client = LMStudioCAFAClient(lm_config)
    results = run_cafa(instances, client, run_config)
    summary = summarize_results(results)

    logging.info("Summary: %s", json.dumps(summary, indent=2))
    save_outputs(results, summary, run_config.output_dir)


if __name__ == "__main__":
    main()
