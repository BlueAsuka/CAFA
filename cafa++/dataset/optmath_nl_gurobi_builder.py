#!/usr/bin/env python3
"""OptMATH-Train -> NL/Gurobi pairs + AST-mutated contrastive negatives.

Quick start:
    pip install datasets huggingface_hub tqdm
    python optmath_nl_gurobi_builder.py --limit 2000 --out optmath_clip
    python optmath_nl_gurobi_builder.py --limit 0 --out optmath_clip_full
    python optmath_nl_gurobi_builder.py --mode download --limit 0
    python optmath_nl_gurobi_builder.py --local OptMATH-Train.jsonl --limit 1000
    python optmath_nl_gurobi_builder.py --demo --out demo_clip

IMPORTANT: The source model is a dataset reference, NOT a formally certified
semantic match. Synthetic mutations are LIKELY negatives, NOT proven invalid.
This script NEVER executes source or generated code.
"""
import argparse
import ast
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from tqdm.auto import tqdm


REPO = "Aurora-Gem/OptMATH-Train"
FILENAME = "OptMATH-Train.jsonl"
FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
MODEL_METHODS = {"addVar", "addVars", "addMVar", "addConstr", "addConstrs",
                 "addLConstr", "addQConstr", "addGenConstrIndicator", "setObjective",
                 "setObjectiveN", "addSOS", "optimize"}
CONSTR_METHODS = {"addConstr", "addConstrs", "addLConstr", "addQConstr"}
VAR_METHODS = {"addVar", "addVars", "addMVar"}


def records_from_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    continue


def load_records(args):
    if args.demo:
        code1 = '''import gurobipy as gp
from gurobipy import GRB
m = gp.Model("production")
x = m.addVar(vtype=GRB.INTEGER, name="x")
y = m.addVar(vtype=GRB.INTEGER, name="y")
m.setObjective(10*x + 15*y, GRB.MAXIMIZE)
m.addConstr(46*x + 13*y <= 1345)
m.addConstr(43*x + 4*y <= 346)
m.optimize()'''
        code2 = '''import gurobipy as gp
from gurobipy import GRB
m = gp.Model("packing")
I = range(3)
w = [3, 4, 2]
x = m.addVars(I, vtype=GRB.BINARY)
m.setObjective(gp.quicksum(x[i] for i in I), GRB.MAXIMIZE)
m.addConstrs((x[i] <= 1 for i in I))
m.addConstr(gp.quicksum(w[i]*x[i] for i in I) <= 5)
m.optimize()'''
        records = []
        for question, code in [
            ("Produce two products with total resource limits 1345 and 346, maximizing profit.", code1),
            ("Select from three items of weights 3, 4 and 2 with capacity 5.", code2),
        ]:
            records.append({"instruction": "Use gurobipy", "input": "# Question:\n" + question,
                            "output": "A proposed model:\n```python\n" + code + "\n```", "answer": 0})
        return records
    if args.local:
        return records_from_jsonl(args.local)
    if args.mode == "download":
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import tqdm as hub_tqdm
        except ImportError as e:
            raise SystemExit("Install dependency: pip install huggingface_hub") from e
        class DownloadProgress(hub_tqdm):
            def __init__(self, *positional, **kwargs):
                kwargs["desc"] = "Downloading dataset"
                kwargs["dynamic_ncols"] = True
                kwargs["disable"] = args.no_progress or kwargs.get("disable", False)
                super().__init__(*positional, **kwargs)

        if not args.no_progress:
            tqdm.write("Preparing dataset download (cached files are reused).")
        source = hf_hub_download(repo_id=args.repo, repo_type="dataset",
                                 filename=FILENAME, local_dir=str(Path(args.out) / "source"),
                                 tqdm_class=DownloadProgress)
        return records_from_jsonl(source)
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise SystemExit("Install dependency: pip install datasets") from e
        # Streaming avoids downloading all 1.77 GB for exploratory subsets.
        if not args.no_progress:
            tqdm.write("Opening dataset stream; records are downloaded as curation proceeds.")
        ds = load_dataset(args.repo, split="train", streaming=True)
        return ds


def curation_records(records, args, counts):
    """Count inspected records, including rejects, without fetching past --limit."""
    try:
        total = len(records)
    except TypeError:
        splits = getattr(getattr(records, "info", None), "splits", None)
        train = splits.get("train") if splits else None
        total = getattr(train, "num_examples", None) or None
    if args.limit:
        total = min(total, args.limit) if total is not None else args.limit
        records = islice(records, args.limit)
    with tqdm(records, total=total, desc="Curating dataset", unit="records",
              dynamic_ncols=True, disable=args.no_progress,
              postfix={"accepted": 0, "skipped": 0, "negatives": 0}) as progress:
        for index, row in enumerate(progress):
            yield index, row
            progress.set_postfix(accepted=counts["accepted_source"],
                                 skipped=counts["seen"] - counts["accepted_source"],
                                 negatives=counts["negative_total"], refresh=False)
        # A requested limit can exceed the available records; finish at the actual count.
        if progress.total is not None and progress.n < progress.total:
            progress.total = progress.n
            if not args.no_progress:
                progress.refresh()


def methods(tree):
    return [n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]


def parse_candidate(src):
    src = src.strip()
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError):
        return None
    calls = methods(tree)
    # Require modeling operations, not merely an optimize() or an example mention.
    if not any(x in calls for x in ("addConstr", "addConstrs", "addLConstr",
                                    "addQConstr", "setObjective", "setObjectiveN")):
        return None
    if not any(x in calls for x in ("addVar", "addVars", "addMVar")):
        return None
    if "gurobi" not in src.lower() and "GRB." not in src and "gp." not in src:
        return None
    score = 4 * sum(x in CONSTR_METHODS for x in calls) + 3 * sum(x in VAR_METHODS for x in calls)
    score += 4 * sum(x in ("setObjective", "setObjectiveN") for x in calls)
    score += min(len(src), 20000) / 10000
    return score, src, tree


def extract_code(output):
    if not isinstance(output, str):
        return None
    choices = []
    for lang, body in FENCE.findall(output):
        tag = lang.strip().lower()
        if tag and tag not in ("python", "py", "python3"):
            continue
        found = parse_candidate(body)
        if found:
            choices.append(found)
    if not choices:
        # Fallback for text with an unfenced code section.
        lines = output.splitlines()
        starts = [i for i, x in enumerate(lines) if re.match(
            r"\s*(import\s+gurobipy\b|from\s+gurobipy\s+import\b)", x)]
        for start in starts[:3]:
            end = len(lines)
            for j in range(start + 1, len(lines)):
                if re.match(r"^\s*(?:#{2,}\s+[A-Za-z]|```|---\s*$)", lines[j]):
                    end = j
                    break
            found = parse_candidate("\n".join(lines[start:end]))
            if found:
                choices.append(found)
    return max(choices, key=lambda x: x[0])[1:] if choices else None


def get_nl(row):
    text = row.get("input", "")
    if not isinstance(text, str):
        return ""
    # Keep the entire question, including inline data and Markdown tables.
    text = text.strip()
    text = re.sub(r"^\s*#{1,3}\s*Question\s*:\s*", "", text, count=1, flags=re.I)
    return text.strip()


def code_canonical(tree):
    try:
        out = ast.unparse(tree)
        ast.parse(out)
        return out
    except (SyntaxError, ValueError, RecursionError, TypeError):
        return None


def nl_variation(nl):
    # Formatting-only variation; NOT an LLM paraphrase or an altered requirement.
    return "Optimization problem:\n" + nl if not nl.startswith("Optimization problem:") else nl


def equivalent_constraint_code(tree):
    """Mathematically equivalent RHS normalization for a numeric linear bound.

    E.g. m.addConstr(2*x + y <= 10) becomes
         m.addConstr((2*x + y) - 10 <= 0).
    Still needs solver validation before being called executable-equivalent.
    """
    cloned = copy.deepcopy(tree)
    for call in ast.walk(cloned):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr in CONSTR_METHODS and call.args):
            continue
        target = call.args[0]
        if isinstance(target, ast.GeneratorExp):
            target = target.elt
        if not isinstance(target, ast.Compare) or len(target.ops) != 1:
            continue
        right = target.comparators[0]
        if (isinstance(target.ops[0], (ast.LtE, ast.GtE, ast.Eq))
                and isinstance(right, ast.Constant)
                and type(right.value) in (int, float)
                and right.value != 0):
            target.left = ast.BinOp(left=target.left, op=ast.Sub(), right=right)
            target.comparators = [ast.Constant(value=0)]
            ast.fix_missing_locations(cloned)
            return code_canonical(cloned)
    return None


def is_grb_attr(node, names):
    return isinstance(node, ast.Attribute) and node.attr in names


def mutation_specs(tree, rng):
    """Collect safe-to-PARSE one-element AST changes, not verified wrong solutions."""
    buckets = defaultdict(list)
    nodes = list(ast.walk(tree))
    ids = {id(n): i for i, n in enumerate(nodes)}
    enclosing_expr = {id(e.value): ids[id(e)] for e in nodes if isinstance(e, ast.Expr)}
    for node in nodes:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        method = node.func.attr
        if method in CONSTR_METHODS and node.args:
            expr = node.args[0]
            if isinstance(expr, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                expr = expr.elt
            for sub in ast.walk(expr):
                if not isinstance(sub, ast.Compare) or len(sub.ops) != 1:
                    continue
                if isinstance(sub.ops[0], (ast.LtE, ast.GtE, ast.Eq)):
                    buckets["inequality_flip"].append((ids[id(sub)], None))
                    rhs = sub.comparators[0]
                    if isinstance(rhs, ast.Constant) and isinstance(rhs.value, (int, float)) and not isinstance(rhs.value, bool):
                        buckets["rhs_perturb"].append((ids[id(rhs)], None))
            # Removing an entire constraint statement remains syntactically valid.
            if id(node) in enclosing_expr:
                buckets["constraint_drop"].append((enclosing_expr[id(node)], None))
        if method in VAR_METHODS:
            for kw in node.keywords:
                if kw.arg == "vtype":
                    v = kw.value
                    if is_grb_attr(v, ("BINARY", "INTEGER", "CONTINUOUS")):
                        buckets["variable_domain"].append((ids[id(v)], None))
                    elif isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value in ("B", "I", "C"):
                        buckets["variable_domain"].append((ids[id(v)], None))
        if method == "setObjective":
            senses = ([node.args[1]] if len(node.args) >= 2 else []) + [
                kw.value for kw in node.keywords if kw.arg == "sense"]
            for sense in senses:
                if is_grb_attr(sense, ("MINIMIZE", "MAXIMIZE")):
                    buckets["objective_flip"].append((ids[id(sense)], None))
    for kind in list(buckets):
        rng.shuffle(buckets[kind])
        buckets[kind] = buckets[kind][:32]
    return buckets


def make_mutation(tree, kind, index):
    obj = copy.deepcopy(tree)
    nodes = list(ast.walk(obj))
    if index >= len(nodes):
        return None
    n = nodes[index]
    if kind == "inequality_flip":
        op = n.ops[0]
        n.ops[0] = ast.GtE() if isinstance(op, ast.LtE) else (
            ast.LtE() if isinstance(op, ast.GtE) else ast.LtE())
    elif kind == "rhs_perturb":
        value = n.value
        # Shift a numeric RHS; 0 and small integer RHS require special handling.
        n.value = value + (1 if isinstance(value, int) and abs(value) < 10
                           else max(1, round(abs(value) * 0.2)) if isinstance(value, int)
                           else max(0.1, abs(value) * 0.2))
    elif kind == "variable_domain":
        if isinstance(n, ast.Attribute):
            n.attr = {"BINARY": "CONTINUOUS", "INTEGER": "CONTINUOUS",
                      "CONTINUOUS": "INTEGER"}[n.attr]
        else:
            n.value = {"B": "C", "I": "C", "C": "I"}[n.value]
    elif kind == "objective_flip":
        n.attr = "MINIMIZE" if n.attr == "MAXIMIZE" else "MAXIMIZE"
    elif kind == "constraint_drop":
        success = False
        for parent in ast.walk(obj):
            for _, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for j, item in enumerate(value):
                        if item is n:
                            value[j] = ast.copy_location(ast.Pass(), n)
                            success = True
                            break
                if success:
                    break
            if success:
                break
        if not success:
            return None
    else:
        return None
    ast.fix_missing_locations(obj)
    return code_canonical(obj)


def negatives(tree, n, rng):
    if n <= 0:
        return []
    original = code_canonical(tree)
    pools = mutation_specs(tree, rng)
    kinds = ["inequality_flip", "rhs_perturb", "constraint_drop",
             "variable_domain", "objective_flip"]
    result, seen = [], {hashlib.sha256(original.encode()).hexdigest()}
    # Round-robin over mutation types to avoid filling every slot with one family.
    while len(result) < n and any(pools.values()):
        for kind in kinds:
            if not pools[kind] or len(result) >= n:
                continue
            index, _ = pools[kind].pop()
            code = make_mutation(tree, kind, index)
            if not code:
                continue
            key = hashlib.sha256(code.encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                result.append({"code": code, "mutation": kind,
                               "label_status": "synthetic_negative_unverified"})
    return result


def split_name(text, seed):
    key = hashlib.sha256((str(seed) + "|" + " ".join(text.split())).encode()).digest()
    r = int.from_bytes(key[:8], "big") / 2**64
    return "train" if r < 0.9 else "val" if r < 0.95 else "test"


def write_jsonl(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO, help="Hugging Face dataset ID")
    ap.add_argument("--local", help="Use an existing local source JSONL instead")
    ap.add_argument("--mode", choices=["stream", "download"], default="stream",
                    help="Stream data or download the entire source file")
    ap.add_argument("--limit", type=int, default=2000,
                    help="Maximum SOURCE records to inspect; 0 = entire dataset")
    ap.add_argument("--out", default=r"D:\data", help="Output directory")
    ap.add_argument("--negatives", type=int, default=5, help="Max mutations per positive")
    ap.add_argument("--augment", type=int, default=1,
                    help="Max additional positive variants per source (0..3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-code-chars", type=int, default=100000)
    ap.add_argument("--demo", action="store_true", help="Offline smoke-test records")
    ap.add_argument("--no-progress", action="store_true", help="Disable download and curation progress bars")
    args = ap.parse_args()
    if args.limit < 0 or args.negatives < 0 or not 0 <= args.augment <= 3:
        ap.error("--limit/--negatives must be >=0 and --augment must be 0..3")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rng = random.Random(args.seed)
    dedupe = set()
    records = load_records(args)

    with ExitStack() as stack:
        pair_files = {s: stack.enter_context(open(out / f"pairs_{s}.jsonl", "w", encoding="utf-8"))
                      for s in ("train", "val", "test")}
        ranking_files = {s: stack.enter_context(open(out / f"ranking_{s}.jsonl", "w", encoding="utf-8"))
                         for s in ("train", "val", "test")}
        for row_idx, row in curation_records(records, args, counts):
            counts["seen"] += 1
            nl = get_nl(row)
            if not nl or len(nl) < 30:
                counts["skip_missing_nl"] += 1
                continue
            extracted = extract_code(row.get("output", ""))
            if not extracted:
                counts["skip_no_valid_gurobi_code"] += 1
                continue
            code, tree = extracted
            if len(code) > args.max_code_chars:
                counts["skip_oversize_code"] += 1
                continue
            key = hashlib.sha256((" ".join(nl.split()) + "\0" +
                                  (code_canonical(tree) or code)).encode()).hexdigest()
            if key in dedupe:
                counts["skip_duplicate"] += 1
                continue
            dedupe.add(key)
            split = split_name(nl, args.seed)
            rid = f"optmath_{row_idx:07d}_{key[:8]}"
            meta = {"id": rid, "split": split, "source_index": row_idx,
                    "answer": row.get("answer"), "instruction": row.get("instruction", ""),
                    "source_repo": "offline_demo" if args.demo else args.repo}
            positive = {**meta, "nl": nl, "code": code, "variant": "source",
                        "label": 1, "label_status": "source_reference_unverified"}
            write_jsonl(pair_files[split], positive)
            counts[f"pairs_{split}"] += 1
            positives = [{"code": code, "variant": "source"}]
            canonical = code_canonical(tree)
            extra = []
            equivalent = equivalent_constraint_code(tree)
            if equivalent and equivalent != code:
                extra.append((nl, equivalent, "algebraic_bound_normalization"))
            if canonical and canonical != code:
                extra.append((nl, canonical, "ast_canonical"))
            if nl_variation(nl) != nl:
                extra.append((nl_variation(nl), code, "nl_header"))
            for aug_nl, aug_code, variant in extra[:args.augment]:
                write_jsonl(pair_files[split], {**meta, "nl": aug_nl, "code": aug_code,
                    "variant": variant, "label": 1,
                    "label_status": "source_reference_augmented_unverified"})
                counts[f"augmented_{split}"] += 1
                if aug_nl == nl:
                    positives.append({"code": aug_code, "variant": variant})
            negs = negatives(tree, args.negatives, rng)
            if negs:
                write_jsonl(ranking_files[split], {**meta, "nl": nl,
                    "positives": positives, "negatives": negs,
                    "verification_note": "Source pairs and mutations are not independently semantically verified."})
                counts[f"ranking_{split}"] += 1
                counts["negative_total"] += len(negs)
                for n in negs:
                    counts["negative_" + n["mutation"]] += 1
            else:
                counts["no_negative_found"] += 1
            counts["accepted_source"] += 1
    report = {"source": "offline_demo" if args.demo else (args.local or args.repo),
              "seed": args.seed, "requested_limit": args.limit,
              "label_policy": "Unverified source reference / synthetic structural negative; no model execution",
              "counts": dict(counts)}
    (out / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Created:", out.resolve())


if __name__ == "__main__":
    main()
