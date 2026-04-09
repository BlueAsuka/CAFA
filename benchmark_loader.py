from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


# -----------------------------------------------------------------------------
# Normalized records used by CAFA
# -----------------------------------------------------------------------------


@dataclass
class BenchmarkInstance:
    dataset_name: str
    instance_id: str
    question: str
    expected_answer: Any = None
    answer_type: str = "unknown"
    problem_type: Optional[str] = None
    difficulty: Optional[str] = None
    code_example: Optional[str] = None
    function_name: Optional[str] = None
    arg_names: List[str] = field(default_factory=list)
    source_files: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _safe_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_function_signature(code_text: str) -> tuple[Optional[str], List[str]]:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return None, []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name, [arg.arg for arg in node.args.args]
    return None, []


def _iter_json_objects(text: str) -> Iterator[Any]:
    """
    Parse either:
      - standard JSON
      - JSONL
      - concatenated JSON objects separated only by whitespace

    This is useful for files such as IndustryOR/dataset.jsonl, whose raw view can
    appear as adjacent JSON objects instead of one object per visible line.
    """
    text = text.strip()
    if not text:
        return

    # Fast path: normal JSON array or object.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for item in obj:
                yield item
        else:
            yield obj
        return
    except json.JSONDecodeError:
        pass

    # JSONL path.
    jsonl_ok = True
    items: List[Any] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            jsonl_ok = False
            break
    if jsonl_ok and items:
        for item in items:
            yield item
        return

    # Concatenated objects path.
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        yield obj
        idx = end


def _guess_answer_type(answer: Any) -> str:
    if answer is None:
        return "none"
    if isinstance(answer, dict):
        return "dict"
    if isinstance(answer, list):
        return "list"
    if isinstance(answer, (int, float)):
        return "scalar"
    if isinstance(answer, str):
        return "string"
    return type(answer).__name__


def _render_instance_values(values: Dict[str, Any], title: str = "Concrete values for this instance") -> str:
    if not values:
        return ""
    lines = [f"- {k}: {v}" for k, v in values.items()]
    return f"\n\n{title}:\n" + "\n".join(lines)


# -----------------------------------------------------------------------------
# Main loader
# -----------------------------------------------------------------------------


class ORBenchmarkLoader:
    """
    Unified loader for the benchmark folders under:

        static/datasets/
          ComplexOR/
          IndustryOR/
          Mamo/
          NL4Opt/
          NLP4LP/
          ReSocratic/

    It normalizes all supported sources into BenchmarkInstance objects so the
    CAFA runner can consume a single interface.
    """

    SUPPORTED_DATASETS = {
        "NL4Opt",
        "ComplexOR",
        "IndustryOR",
        "Mamo",
        "NLP4LP",
        "ReSocratic",
    }

    def __init__(self, benchmarks_root: str | Path):
        self.root = Path(benchmarks_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Benchmark root not found: {self.root}")

    def available_datasets(self) -> List[str]:
        names = [p.name for p in self.root.iterdir() if p.is_dir() and p.name in self.SUPPORTED_DATASETS]
        return sorted(names)

    def iter_instances(self, dataset_names: Optional[Sequence[str]] = None) -> Iterator[BenchmarkInstance]:
        names = list(dataset_names) if dataset_names else self.available_datasets()
        for name in names:
            if name not in self.SUPPORTED_DATASETS:
                raise ValueError(f"Unsupported dataset: {name}")
            loader = getattr(self, f"_iter_{name.lower()}_instances")
            yield from loader(self.root / name)

    # ------------------------------------------------------------------
    # NL4Opt
    # ------------------------------------------------------------------

    def _iter_nl4opt_instances(self, dataset_dir: Path) -> Iterator[BenchmarkInstance]:
        for folder in sorted(dataset_dir.iterdir(), key=lambda p: int(p.name.split("_")[1]) if re.fullmatch(r"prob_\d+", p.name) else 10**9):
            if not folder.is_dir() or not re.fullmatch(r"prob_\d+", folder.name):
                continue

            description_path = folder / "description.txt"
            sample_path = folder / "sample.json"
            code_path = folder / "code_example.py"

            description = _read_text(description_path) if description_path.exists() else ""
            code_example = _read_text(code_path) if code_path.exists() else None
            function_name, arg_names = _parse_function_signature(code_example) if code_example else (None, [])
            samples = _safe_load_json(sample_path) if sample_path.exists() else []

            for sample_idx, sample in enumerate(samples):
                inputs = sample.get("input", {}) if isinstance(sample, dict) else {}
                outputs = sample.get("output", None) if isinstance(sample, dict) else None
                question = description + _render_instance_values(inputs)

                yield BenchmarkInstance(
                    dataset_name="NL4Opt",
                    instance_id=f"{folder.name}::sample_{sample_idx}",
                    question=question.strip(),
                    expected_answer=outputs,
                    answer_type=_guess_answer_type(outputs),
                    code_example=code_example,
                    function_name=function_name,
                    arg_names=arg_names,
                    source_files={
                        "description": str(description_path),
                        "sample": str(sample_path),
                        "code_example": str(code_path),
                    },
                    metadata={
                        "problem_id": folder.name,
                        "sample_index": sample_idx,
                        "input_values": inputs,
                    },
                )

    # ------------------------------------------------------------------
    # ComplexOR
    # ------------------------------------------------------------------

    def _iter_complexor_instances(self, dataset_dir: Path) -> Iterator[BenchmarkInstance]:
        for folder in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            description_path = folder / "description.txt"
            sample_path = folder / "sample.json"
            code_path = folder / "code_example.py"
            gt_model_path = folder / "gt_model.txt"

            if not description_path.exists():
                continue

            description = _read_text(description_path)
            code_example = _read_text(code_path) if code_path.exists() else None
            function_name, arg_names = _parse_function_signature(code_example) if code_example else (None, [])
            gt_model = _read_text(gt_model_path) if gt_model_path.exists() else None
            samples = _safe_load_json(sample_path) if sample_path.exists() else []

            if isinstance(samples, list) and samples:
                for sample_idx, sample in enumerate(samples):
                    inputs = sample.get("input", {}) if isinstance(sample, dict) else {}
                    outputs = sample.get("output", None) if isinstance(sample, dict) else None
                    question = description + _render_instance_values(inputs)
                    yield BenchmarkInstance(
                        dataset_name="ComplexOR",
                        instance_id=f"{folder.name}::sample_{sample_idx}",
                        question=question.strip(),
                        expected_answer=outputs,
                        answer_type=_guess_answer_type(outputs),
                        code_example=code_example,
                        function_name=function_name,
                        arg_names=arg_names,
                        source_files={
                            "description": str(description_path),
                            "sample": str(sample_path),
                            "code_example": str(code_path),
                            "gt_model": str(gt_model_path),
                        },
                        metadata={
                            "problem_name": folder.name,
                            "sample_index": sample_idx,
                            "input_values": inputs,
                            "gt_model": gt_model,
                        },
                    )
            else:
                yield BenchmarkInstance(
                    dataset_name="ComplexOR",
                    instance_id=folder.name,
                    question=description,
                    expected_answer=None,
                    answer_type="none",
                    code_example=code_example,
                    function_name=function_name,
                    arg_names=arg_names,
                    source_files={
                        "description": str(description_path),
                        "sample": str(sample_path),
                        "code_example": str(code_path),
                        "gt_model": str(gt_model_path),
                    },
                    metadata={
                        "problem_name": folder.name,
                        "gt_model": gt_model,
                    },
                )

    # ------------------------------------------------------------------
    # NLP4LP
    # ------------------------------------------------------------------

    def _iter_nlp4lp_instances(self, dataset_dir: Path) -> Iterator[BenchmarkInstance]:
        numeric_dirs = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        for folder in sorted(numeric_dirs, key=lambda p: int(p.name)):
            description_path = folder / "description.txt"
            info_path = folder / "problem_info.json"
            solution_path = folder / "solution.json"
            parameters_path = folder / "parameters.json"
            code_path = folder / "optimus_code.py"

            description = _read_text(description_path) if description_path.exists() else ""
            problem_info = _safe_load_json(info_path) if info_path.exists() else {}
            solution = _safe_load_json(solution_path) if solution_path.exists() else None
            parameters = _safe_load_json(parameters_path) if parameters_path.exists() else None
            code_example = _read_text(code_path) if code_path.exists() else None
            function_name, arg_names = _parse_function_signature(code_example) if code_example else (None, [])

            expected_answer = solution
            if isinstance(solution, dict) and "objective" in solution:
                expected_answer = solution["objective"]

            yield BenchmarkInstance(
                dataset_name="NLP4LP",
                instance_id=folder.name,
                question=description,
                expected_answer=expected_answer,
                answer_type=_guess_answer_type(expected_answer),
                code_example=code_example,
                function_name=function_name,
                arg_names=arg_names,
                source_files={
                    "description": str(description_path),
                    "problem_info": str(info_path),
                    "solution": str(solution_path),
                    "parameters": str(parameters_path),
                    "code_example": str(code_path),
                },
                metadata={
                    "folder_id": folder.name,
                    "problem_info": problem_info,
                    "parameters": parameters,
                    "full_solution": solution,
                },
            )

    # ------------------------------------------------------------------
    # IndustryOR
    # ------------------------------------------------------------------

    def _iter_industryor_instances(self, dataset_dir: Path) -> Iterator[BenchmarkInstance]:
        data_path = dataset_dir / "dataset.jsonl"
        if not data_path.exists():
            return

        text = data_path.read_text(encoding="utf-8")
        for idx, record in enumerate(_iter_json_objects(text)):
            if not isinstance(record, dict):
                continue
            question = record.get("en_question") or record.get("question") or record.get("Question") or ""
            answer = record.get("en_answer") or record.get("answer") or record.get("Answer")
            yield BenchmarkInstance(
                dataset_name="IndustryOR",
                instance_id=f"industryor_{idx}",
                question=str(question).strip(),
                expected_answer=answer,
                answer_type=_guess_answer_type(answer),
                problem_type=record.get("question_type") or record.get("type"),
                difficulty=record.get("difficulty"),
                source_files={"dataset": str(data_path)},
                metadata=record,
            )

    # ------------------------------------------------------------------
    # ReSocratic
    # ------------------------------------------------------------------

    def _iter_resocratic_instances(self, dataset_dir: Path) -> Iterator[BenchmarkInstance]:
        data_dir = dataset_dir / "data"
        if not data_dir.exists():
            return

        for json_path in sorted(data_dir.glob("*.json")):
            try:
                data = _safe_load_json(json_path)
            except Exception:
                continue

            if not isinstance(data, list):
                continue

            for idx, record in enumerate(data):
                if not isinstance(record, dict):
                    continue
                question = record.get("question") or record.get("en_question") or ""
                answer = record.get("results") or record.get("answer")
                record_idx = record.get("index", idx)
                yield BenchmarkInstance(
                    dataset_name="ReSocratic",
                    instance_id=f"{json_path.stem}::{record_idx}",
                    question=str(question).strip(),
                    expected_answer=answer,
                    answer_type=_guess_answer_type(answer),
                    problem_type=record.get("type"),
                    source_files={"dataset": str(json_path)},
                    metadata=record,
                )

    # ------------------------------------------------------------------
    # Mamo
    # ------------------------------------------------------------------

    def _iter_mamo_instances(self, dataset_dir: Path) -> Iterator[BenchmarkInstance]:
        # CAFA is aimed at optimization modeling, so we focus on optimization files.
        opt_dir = dataset_dir / "data" / "optimization"
        if not opt_dir.exists():
            return

        for jsonl_path in sorted(opt_dir.glob("*.jsonl")):
            text = jsonl_path.read_text(encoding="utf-8")
            for idx, record in enumerate(_iter_json_objects(text)):
                if not isinstance(record, dict):
                    continue
                question = record.get("Question") or record.get("question") or ""
                answer = record.get("Answer") or record.get("answer")
                yield BenchmarkInstance(
                    dataset_name="Mamo",
                    instance_id=f"{jsonl_path.stem}::{record.get('id', idx)}",
                    question=str(question).strip(),
                    expected_answer=answer,
                    answer_type=_guess_answer_type(answer),
                    problem_type=record.get("Type") or record.get("type"),
                    source_files={"dataset": str(jsonl_path)},
                    metadata=record,
                )


# -----------------------------------------------------------------------------
# CLI preview
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Inspect OR benchmark datasets through one unified loader")
    parser.add_argument("--root", type=str, required=True, help="Path to static/datasets")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of datasets to load, e.g. NL4Opt NLP4LP IndustryOR",
    )
    parser.add_argument("--show", type=int, default=5, help="How many instances to preview")
    args = parser.parse_args()

    loader = ORBenchmarkLoader(args.root)
    print("Available datasets:", loader.available_datasets())

    instances = list(loader.iter_instances(args.datasets))
    print(f"Loaded {len(instances)} normalized instances")

    for item in instances[: args.show]:
        pprint(
            {
                "dataset": item.dataset_name,
                "instance_id": item.instance_id,
                "problem_type": item.problem_type,
                "difficulty": item.difficulty,
                "question": item.question[:400] + ("..." if len(item.question) > 400 else ""),
                "expected_answer": item.expected_answer,
                "source_files": item.source_files,
            }
        )
        print("-" * 100)
