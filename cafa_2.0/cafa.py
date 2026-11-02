"""CAFA 2.0: generate -> validate -> build -> solve -> evaluate -> save."""
from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import datetime as dt
import json
import logging
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_loader import BenchmarkInstance, ORBenchmarkLoader
from lmstudio_client import LMStudioCAFAClient, LMStudioConfig, extract_code, instance_data


@dataclass
class RunConfig:
    dataset_root: str
    datasets: list[str] | None = None
    output_dir: str = "output/cafa_2.0"
    max_instances: int | None = None
    start_index: int = 0
    stop_index: int | None = None
    solver: str = "auto"
    time_limit: float = 30
    execution_timeout: float = 120
    skip_solve: bool = False
    code_file: str | None = None
    objective_key: str | None = None


def validate_code(code: str) -> ast.Module:
    """Check the small modeling contract without changing the returned Python."""
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Return exactly one function: def build_model(data)")
    fn = tree.body[0]
    if (fn.name != "build_model" or [a.arg for a in fn.args.args] != ["data"]
            or fn.decorator_list or fn.returns or fn.args.defaults or fn.args.kwonlyargs
            or fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.args[0].annotation):
        raise ValueError("Expected undecorated build_model(data) without annotations or defaults")
    allowed = {"Model", "add_variable", "add_integer_variable", "add_binary_variable",
               "add_linear_constraint", "minimize", "maximize", "get", "items", "keys", "values"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError("Imports, classes, and global declarations are outside the modeling contract")
        if isinstance(node, ast.Attribute) and (node.attr not in allowed or isinstance(node.ctx, ast.Store)):
            raise NotImplementedError(f"Unsupported attribute: {node.attr}; only the documented LP/MILP API is allowed")
    return tree


def extract_scalar_answer(answer, objective_key: str | None = None) -> float | None:
    """Accept an explicit objective, not the first number in a solution vector."""
    if isinstance(answer, dict):
        if objective_key:
            return extract_scalar_answer(answer.get(objective_key))
        for key in ("objective", "objective_value", "optimal_value", "obj"):
            if key in answer:
                return extract_scalar_answer(answer[key])
        candidates = [v for k, v in answer.items() if re.search(
            r"\b(objective|profit|cost|revenue|maximum|minimum|optimal|maximized|minimized|maximised|minimised)\b", k, re.I)]
        if len(candidates) == 1:
            return extract_scalar_answer(candidates[0])
        for key in ("answer", "output", "results", "value"):
            if len(answer) == 1 and key in answer:
                return extract_scalar_answer(answer[key])
        return None
    if isinstance(answer, list):
        return extract_scalar_answer(answer[0]) if len(answer) == 1 else None
    if isinstance(answer, bool) or not isinstance(answer, (int, float, str)):
        return None
    try:
        value = float(answer)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def solve_request(request: dict) -> dict:
    """Worker entry: preserve build/solver failures and export the model before solving."""
    record = {"stage": "validation", "status": "error"}
    try:
        tree = validate_code(request["code"])
        record["stage"] = "build"
        from ortools.math_opt.python import mathopt
        names = "sum range len enumerate zip min max abs float int sorted dict list tuple set ValueError NotImplementedError".split()
        namespace = {"mathopt": mathopt, "__builtins__": {n: getattr(builtins, n) for n in names}}
        exec(compile(tree, "<generated_model>", "exec"), namespace)
        model = namespace["build_model"](request["data"])
        if not isinstance(model, mathopt.Model):
            raise ValueError("build_model(data) must return a MathOpt Model")
        proto = model.export_model()
        record["model_proto"] = str(proto)
        fields = {field.name for field, _ in proto.ListFields()}
        if (fields - {"name", "variables", "objective", "linear_constraints", "linear_constraint_matrix"}
                or proto.objective.quadratic_coefficients.row_ids):
            raise NotImplementedError("Only one linear objective and linear constraints are supported")
        variables = list(model.variables())
        constraints = list(model.linear_constraints())
        for label, entities in (("variable", variables), ("constraint", constraints)):
            if any(not e.name for e in entities) or len({e.name for e in entities}) != len(entities):
                raise ValueError(f"Give every {label} a unique, nonempty name for result tracking")
        is_mip = any(v.integer for v in variables)
        solver = request["solver"]
        if solver == "auto":
            solver = "gscip" if is_mip else "glop"
        model_type = "IP" if variables and all(v.integer for v in variables) else "MILP" if is_mip else "LP"
        record.update(solver=solver, model_type=model_type,
                      variables=len(variables), constraints=len(constraints))
        if solver == "glop" and is_mip:
            raise ValueError("GLOP cannot solve integer models; use auto, gscip, or gurobi")
        record["stage"] = "solve"
        params = mathopt.SolveParameters(time_limit=dt.timedelta(seconds=request["time_limit"]))
        result = mathopt.solve(model, getattr(mathopt.SolverType, solver.upper()), params=params)
        feasible = result.has_primal_feasible_solution()
        bound = result.best_objective_bound()
        record.update(status=result.termination.reason.name.lower(), stage="complete",
                      termination_detail=result.termination.detail,
                      limit=result.termination.limit.name if result.termination.limit else None,
                      prediction=result.objective_value() if feasible else None,
                      best_bound=bound if math.isfinite(bound) else None,
                      solve_seconds=result.solve_time().total_seconds(),
                      variable_values={v.name: value for v, value in result.variable_values().items()} if feasible else {})
    except Exception as exc:
        record.update(status="unsupported" if isinstance(exc, NotImplementedError) else "error",
                      error=f"{type(exc).__name__}: {exc}")
    return record


def execute_code(code: str, data: dict, config: RunConfig) -> dict:
    """A fresh process bounds generation execution time and separates its namespace."""
    request = dict(code=code, data=data, solver=config.solver, time_limit=config.time_limit)
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(request), capture_output=True, text=True, encoding="utf-8",
            timeout=config.execution_timeout,
        )
        if process.returncode:
            return dict(status="error", stage="worker", error=f"Worker exit code {process.returncode}", worker_stderr=process.stderr)
        result = json.loads(process.stdout)
        if process.stderr:
            result["worker_stderr"] = process.stderr
        return result
    except subprocess.TimeoutExpired:
        return dict(status="timeout", stage="execution", error=f"Worker exceeded {config.execution_timeout:g}s")
    except json.JSONDecodeError:
        return dict(status="error", stage="worker", error="Worker returned invalid JSON",
                    worker_stdout=process.stdout, worker_stderr=process.stderr)


def run_instance(instance: BenchmarkInstance, client, config: RunConfig, folder: Path, index: int) -> dict:
    gold = extract_scalar_answer(instance.expected_answer, config.objective_key)
    record = dict(dataset_name=instance.dataset_name, instance_id=instance.instance_id,
                  question=instance.question, input_data=instance_data(instance),
                  expected_answer=instance.expected_answer, gold_scalar=gold,
                  answer_type=instance.answer_type, problem_type=instance.problem_type,
                  difficulty=instance.difficulty, source_files=instance.source_files,
                  status="error", stage="generate", prediction=None,
                  correct=None if config.skip_solve or gold is None else False, artifacts={})
    if gold is None:
        record["evaluation_note"] = "No unambiguous scalar objective; inspect expected_answer or set --objective-key."

    def save_artifact(kind: str, suffix: str, text: str):
        path = folder / f"{index:05d}.{suffix}"
        path.write_text(text, encoding="utf-8")
        record["artifacts"][kind] = path.name

    try:
        if config.code_file:
            raw_text = Path(config.code_file).read_text(encoding="utf-8")
            code = extract_code(raw_text)
        else:
            # Save the prompt before requesting generation, including on HTTP failures.
            prompt = client.build_prompt(instance)
            save_artifact("prompt", "prompt.txt", prompt)
            response = client.generate_code(instance, prompt=prompt)
            raw_text, code = response.raw_text, response.code
        save_artifact("raw_response", "response.txt", raw_text)
        save_artifact("code", "py", code)
        record["stage"] = "validation"
        validate_code(code)
        if config.skip_solve:
            record.update(status="generated", stage="complete")
        else:
            record["stage"] = "execution"
            record.update(execute_code(code, record["input_data"], config))
            model_proto = record.pop("model_proto", None)
            if model_proto is not None:
                save_artifact("model", "model.pbtxt", model_proto)
            if gold is not None:
                record["correct"] = (record["status"] == "optimal" and record["prediction"] is not None
                                     and math.isclose(record["prediction"], gold, rel_tol=1e-4, abs_tol=1e-4))
    except Exception as exc:
        record.update(status="unsupported" if isinstance(exc, NotImplementedError) else "error",
                      error=f"{type(exc).__name__}: {exc}")
    record["artifacts"]["result"] = f"{index:05d}.result.json"
    save_artifact("result", "result.json", json.dumps(record, ensure_ascii=False, indent=2))
    return record


def summarize_results(results: list[dict]) -> dict:
    def stats(rows):
        scored = sum(r["correct"] is not None for r in rows)
        correct = sum(r["correct"] is True for r in rows)
        return dict(total=len(rows), solved=sum(r["prediction"] is not None for r in rows),
                    scored=scored, correct=correct, accuracy=correct / scored if scored else None,
                    statuses=dict(Counter(r["status"] for r in rows)))
    per_dataset = {name: stats([r for r in results if r["dataset_name"] == name])
                   for name in sorted({r["dataset_name"] for r in results})}
    return dict(overall=stats(results), per_dataset=per_dataset)


def run_cafa(instances, client, config: RunConfig, folder: Path) -> list[dict]:
    results = []
    with (folder / "results.jsonl").open("w", encoding="utf-8") as stream:
        for index, instance in enumerate(instances, 1):
            record = run_instance(instance, client, config, folder, index)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()  # Completed instances survive an interrupted run.
            results.append(record)
            logging.info("[%d] %s :: %s | %s | objective=%s | correct=%s", index,
                         instance.dataset_name, instance.instance_id, record["status"], record["prediction"], record["correct"])
            if record.get("error"):
                logging.warning("    %s: %s", record["stage"], record["error"])
    return results


def parse_args() -> tuple[RunConfig, LMStudioConfig]:
    parser = argparse.ArgumentParser(description="CAFA 2.0: inspectable MathOpt generation for LP/MILP")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output-dir", default=RunConfig.output_dir)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--solver", choices=["auto", "glop", "gscip", "gurobi"], default="auto")
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--execution-timeout", type=float, default=120)
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--code-file", help="Replay one saved build_model function, without calling an LLM")
    parser.add_argument("--objective-key", help="Exact key identifying the objective in a gold answer dictionary")
    parser.add_argument("--model-name", default=LMStudioConfig.model_name)
    parser.add_argument("--base-url", default=LMStudioConfig.base_url)
    parser.add_argument("--api-key", default=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"))
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--zero-shot", action="store_true", help="Omit examples for a controlled prompt comparison")
    args = vars(parser.parse_args())
    if any(not math.isfinite(args[k]) or args[k] <= 0 for k in ("time_limit", "execution_timeout", "timeout", "max_retries")):
        parser.error("Time limits, timeout, and max-retries must be finite and positive")
    if args["start_index"] < 0 or (args["max_instances"] is not None and args["max_instances"] < 1):
        parser.error("start-index must be nonnegative and max-instances must be positive")
    if args["stop_index"] is not None and args["stop_index"] < args["start_index"]:
        parser.error("stop-index must be at least start-index")
    run = RunConfig(**{field.name: args[field.name] for field in fields(RunConfig)})
    lm = LMStudioConfig(**{field.name: args[field.name] for field in fields(LMStudioConfig)})
    return run, lm


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config, lm_config = parse_args()
    instances = list(ORBenchmarkLoader(config.dataset_root).iter_instances(config.datasets))
    instances = instances[config.start_index:config.stop_index]
    if config.max_instances is not None:
        instances = instances[:config.max_instances]
    if not instances:
        raise SystemExit("No instances selected; check dataset names, files, and slice options.")
    folder = Path(config.output_dir) / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True)
    lm_settings = {k: v for k, v in asdict(lm_config).items() if k != "api_key"}
    runtime = dict(python=sys.version, ortools=version("ortools"), openai=version("openai"))
    settings = dict(run=asdict(config), llm=lm_settings, runtime=runtime)
    (folder / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    logging.info("Selected %d instances; artifacts: %s", len(instances), folder)
    client = None if config.code_file else LMStudioCAFAClient(lm_config)
    results = run_cafa(instances, client, config, folder)
    summary = summarize_results(results)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Summary: %s", json.dumps(summary))


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        request = json.loads(sys.stdin.read())
        with contextlib.redirect_stdout(sys.stderr):
            result = solve_request(request)
        print(json.dumps(result))
    else:
        main()
