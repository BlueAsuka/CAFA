"""CAFA 2.0: generate -> validate -> build -> solve -> evaluate -> save."""
from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import datetime as dt
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace
from collections import Counter
from dataclasses import asdict, dataclass, fields
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_loader import BenchmarkInstance, ORBenchmarkLoader
from clients import CAFAClient, ClientConfig, DEFAULT_ENV_FILE, PROMPT_FILES, PROVIDERS, extract_code, instance_data


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
    repair_attempts: int = 0
    modeling_api: str = "mathopt"


def validate_code(code: str, modeling_api: str = "mathopt") -> ast.Module:
    """Check the small modeling contract without changing the returned Python."""
    tree = ast.parse(code)
    if not tree.body:
        raise ValueError("Return Python modeling statements that assign a MathOpt Model to m")
    allowed = {"Model", "add_variable", "add_integer_variable", "add_binary_variable",
               "add_linear_constraint", "minimize", "maximize", "get", "items", "keys", "values",
               "setdefault", "update", "append", "extend", "add", "split", "strip", "replace"}
    if modeling_api == "gurobi":
        allowed -= {"add_variable", "add_integer_variable", "add_binary_variable", "add_linear_constraint", "minimize", "maximize"}
        allowed |= {"addVar", "addConstr", "setObjective", "quicksum", "CONTINUOUS", "INTEGER",
                    "BINARY", "INFINITY", "MINIMIZE", "MAXIMIZE"}
    elif modeling_api != "mathopt":
        raise ValueError(f"Unknown modeling API: {modeling_api}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError("Imports, classes, and global declarations are outside the modeling contract")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Return)):
            raise ValueError("Use flat modeling statements; functions, lambdas, and returns are not allowed")
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in {"mathopt", "gp", "GRB", "data", "__builtins__"}:
            raise ValueError(f"Do not overwrite the runner-provided name: {node.id}")
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


def solve_gurobi(request: dict, tree: ast.Module, namespace: dict, record: dict) -> None:
    """Build and solve native gurobipy code with runner-owned model lifetimes."""
    if request["solver"] not in {"auto", "gurobi"}:
        raise ValueError("The Gurobi prompt requires --solver auto or --solver gurobi")
    import gurobipy as gp
    from gurobipy import GRB

    with contextlib.ExitStack() as stack:
        env = stack.enter_context(gp.Env(params={"OutputFlag": 0}))

        def create_model(*args, **kwargs):
            return stack.enter_context(gp.Model(*args, env=env, **kwargs))

        namespace.update(gp=SimpleNamespace(Model=create_model, quicksum=gp.quicksum), GRB=GRB)
        exec(compile(tree, "<generated_model>", "exec"), namespace)
        model = namespace.get("m")
        if not isinstance(model, gp.Model):
            raise ValueError("Modeling statements must leave a Gurobi Model in m")
        model.update()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.lp"
            model.write(str(path))
            record["model_text"] = path.read_text(encoding="utf-8")
        if model.NumQNZs or model.NumQConstrs or model.NumGenConstrs or model.NumSOS or model.NumObj > 1:
            raise NotImplementedError("Only one linear objective and linear constraints are supported")
        variables = model.getVars()
        constraints = model.getConstrs()
        for label, names in (("variable", [v.VarName for v in variables]),
                             ("constraint", [c.ConstrName for c in constraints])):
            if any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError(f"Give every {label} a unique, nonempty name for result tracking")
        integer_count = sum(v.VType in {GRB.INTEGER, GRB.BINARY} for v in variables)
        record.update(solver="gurobi", variables=len(variables), constraints=len(constraints),
                      model_type="IP" if variables and integer_count == len(variables) else "MILP" if integer_count else "LP")
        record["stage"] = "solve"
        model.setParam("TimeLimit", request["time_limit"])
        model.optimize()
        feasible = model.SolCount > 0
        statuses = {GRB.OPTIMAL: "optimal", GRB.INFEASIBLE: "infeasible",
                    GRB.INF_OR_UNBD: "infeasible_or_unbounded", GRB.UNBOUNDED: "unbounded"}
        prediction = model.ObjVal if feasible else None
        bound = model.ObjBound if integer_count else prediction if model.Status == GRB.OPTIMAL else None
        record.update(status=statuses.get(model.Status, "feasible" if feasible else "no_solution_found"),
                      stage="complete", termination_detail=f"Gurobi status {model.Status}",
                      limit="time" if model.Status == GRB.TIME_LIMIT else None,
                      prediction=prediction, best_bound=bound if bound is not None and math.isfinite(bound) and abs(bound) < GRB.INFINITY else None,
                      solve_seconds=model.Runtime,
                      variable_values={v.VarName: v.X for v in variables} if feasible else {})


def solve_request(request: dict) -> dict:
    """Worker entry: preserve build/solver failures and export the model before solving."""
    record = {"stage": "validation", "status": "error"}
    try:
        modeling_api = request.get("modeling_api", "mathopt")
        tree = validate_code(request["code"], modeling_api)
        record["stage"] = "build"
        names = "sum range len enumerate zip min max abs float int bool str round sorted reversed all any dict list tuple set ValueError NotImplementedError".split()
        namespace = {"data": request["data"],
                     "__builtins__": {n: getattr(builtins, n) for n in names}}
        if modeling_api == "gurobi":
            solve_gurobi(request, tree, namespace, record)
            return record
        from ortools.math_opt.python import mathopt
        namespace["mathopt"] = mathopt
        exec(compile(tree, "<generated_model>", "exec"), namespace)
        model = namespace.get("m")
        if not isinstance(model, mathopt.Model):
            raise ValueError("Modeling statements must leave a MathOpt Model in m")
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
        if isinstance(exc, ModuleNotFoundError):
            record["stage"] = "dependency"
        elif type(exc).__name__ == "GurobiError" and getattr(exc, "errno", None) == 10009:
            record["stage"] = "license"
        for frame in traceback.extract_tb(exc.__traceback__):
            if frame.filename == "<generated_model>":
                record["error_line"] = frame.lineno
    return record


def execute_code(code: str, data: dict, config: RunConfig) -> dict:
    """A fresh process bounds generation execution time and separates its namespace."""
    request = dict(code=code, data=data, solver=config.solver, time_limit=config.time_limit,
                   modeling_api=config.modeling_api)
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
                  difficulty=instance.difficulty, source_files=instance.source_files, generation_attempts=0,
                  status="error", stage="generate", prediction=None,
                  correct=None if config.skip_solve or gold is None else False, artifacts={})
    if gold is None:
        record["evaluation_note"] = "No unambiguous scalar objective; inspect expected_answer or set --objective-key."

    attempt_artifacts = None

    def save_artifact(kind: str, suffix: str, text: str):
        path = folder / f"{index:05d}.{suffix}"
        path.write_text(text, encoding="utf-8")
        record["artifacts"][kind] = path.name
        if attempt_artifacts is not None:
            snapshot = folder / f"{index:05d}.attempt{attempt:02d}.{suffix}"
            snapshot.write_text(text, encoding="utf-8")
            attempt_artifacts[kind] = snapshot.name

    repairs = config.repair_attempts if not config.code_file and not config.skip_solve else 0
    history = []
    code = ""
    outcome = {}
    for attempt in range(1, repairs + 2):
        previous = outcome
        outcome = dict(stage="generate", status="error", prediction=None)
        record["artifacts"] = {}
        attempt_artifacts = {} if repairs else None
        try:
            if config.code_file:
                raw_text = Path(config.code_file).read_text(encoding="utf-8")
                code = extract_code(raw_text)
            else:
                prompt = (client.build_prompt(instance) if attempt == 1 else
                          client.build_repair_prompt(instance, code, previous))
                save_artifact("prompt", "prompt.txt", prompt)
                record["generation_attempts"] += 1
                response = client.generate_code(instance, prompt=prompt)
                raw_text, code = response.raw_text, response.code
            save_artifact("raw_response", "response.txt", raw_text)
            save_artifact("code", "py", code)
            outcome["stage"] = "validation"
            validate_code(code, config.modeling_api)
            if config.skip_solve:
                outcome.update(status="generated", stage="complete")
            else:
                outcome["stage"] = "execution"
                outcome.update(execute_code(code, record["input_data"], config))
                model_proto = outcome.pop("model_proto", None)
                if model_proto is not None:
                    save_artifact("model", "model.pbtxt", model_proto)
                model_text = outcome.pop("model_text", None)
                if model_text is not None:
                    save_artifact("model", "model.lp", model_text)
        except Exception as exc:
            outcome.update(status="unsupported" if isinstance(exc, NotImplementedError) else "error",
                           error=f"{type(exc).__name__}: {exc}")
        if repairs:
            history.append(dict(attempt=attempt, **outcome, artifacts=attempt_artifacts))
        # Never consult correctness or gold to decide whether to request a revision.
        repairable = (outcome["status"] in {"infeasible", "unbounded", "infeasible_or_unbounded"}
                      or outcome["status"] in {"error", "unsupported"}
                      and outcome["stage"] in {"validation", "build", "solve"})
        if not repairable:
            break
    record.update(outcome)
    if repairs:
        record["attempts"] = history
    if not config.skip_solve and gold is not None:
        record["correct"] = (record["status"] == "optimal" and record["prediction"] is not None
                             and math.isclose(record["prediction"], gold, rel_tol=1e-4, abs_tol=1e-4))
    attempt_artifacts = None
    record["artifacts"]["result"] = f"{index:05d}.result.json"
    save_artifact("result", "result.json", json.dumps(record, ensure_ascii=False, indent=2))
    return record


def summarize_results(results: list[dict]) -> dict:
    def stats(rows):
        scored = sum(r["correct"] is not None for r in rows)
        correct = sum(r["correct"] is True for r in rows)
        return dict(total=len(rows), solved=sum(r["prediction"] is not None for r in rows),
                    scored=scored, correct=correct, accuracy=correct / scored if scored else None,
                    optimal_mismatch=sum(r["status"] == "optimal" and r["correct"] is False for r in rows),
                    first_attempt_statuses=dict(Counter(r["attempts"][0]["status"] if r.get("attempts")
                                                       else r["status"] for r in rows)),
                    generation_attempts=sum(r.get("generation_attempts", int("prompt" in r.get("artifacts", {}))) for r in rows),
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


def parse_args() -> tuple[RunConfig, ClientConfig]:
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
    parser.add_argument("--code-file", help="Replay saved flat modeling statements, without calling an LLM")
    parser.add_argument("--objective-key", help="Exact key identifying the objective in a gold answer dictionary")
    parser.add_argument("--repair-attempts", type=int, default=0,
                        help="Extra generations using execution feedback (default: 0; adds API calls)")
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument("--prompt", dest="prompt_style", choices=PROMPT_FILES, default="mathopt",
                        help="Prompt and generated-code API (default: mathopt)")
    parser.add_argument("--prompt-file", help="Custom template path, used with the API selected by --prompt")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Model configuration file (default: configs/.env)")
    parser.add_argument("--model-name")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float, help="Omitted by default for compatibility with reasoning models")
    parser.add_argument("--max-tokens", type=int, help="Output token limit, including reasoning tokens where applicable")
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
    if args["max_tokens"] is not None and args["max_tokens"] <= 0:
        parser.error("max-tokens must be positive")
    if args["repair_attempts"] < 0:
        parser.error("repair-attempts must be nonnegative")
    if args["temperature"] is not None and (not math.isfinite(args["temperature"]) or not 0 <= args["temperature"] <= 2):
        parser.error("temperature must be finite and between 0 and 2")
    if not Path(args["env_file"]).is_file() and args["env_file"] != str(DEFAULT_ENV_FILE):
        parser.error("env-file does not exist")
    if args["prompt_file"] and not Path(args["prompt_file"]).is_file():
        parser.error("prompt-file does not exist")
    args["modeling_api"] = args["prompt_style"]
    if args["modeling_api"] == "gurobi" and args["solver"] not in {"auto", "gurobi"}:
        parser.error("--prompt gurobi requires --solver auto or --solver gurobi")
    run = RunConfig(**{field.name: args[field.name] for field in fields(RunConfig)})
    lm = ClientConfig(**{field.name: args[field.name] for field in fields(ClientConfig)})
    return run, lm


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config, lm_config = parse_args()
    if config.modeling_api == "gurobi" and not config.skip_solve:
        try:
            import gurobipy as gp
            with gp.Env(params={"OutputFlag": 0}):
                pass
        except Exception as exc:
            raise SystemExit(f"Gurobi initialization failed before generation: {exc}. Install gurobipy and configure a valid license.") from exc
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
    template = lm_config.prompt_path.read_bytes()
    settings["prompt_file"] = str(lm_config.prompt_path)
    settings["prompt_sha256"] = hashlib.sha256(template).hexdigest()
    (folder / "prompt_template.txt").write_bytes(template)
    (folder / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    logging.info("Selected %d instances; artifacts: %s", len(instances), folder)
    client = None if config.code_file else CAFAClient(lm_config)
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
