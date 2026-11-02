# CAFA 2.0

A compact MathOpt implementation following CAFA 1.0's client-and-runner structure.
The runtime has two Python modules, with the full prompt in a separate text file.

| File | Responsibility |
| --- | --- |
| `lmstudio_client.py` | Build the prompt, call LM Studio, extract surrounding code fences |
| `cafa.py` | Validate, build, solve, evaluate, and save each instance |
| `prompt.txt` | Output contract and four authored LP/IP/MILP demonstrations |
| `tests/test_cafa.py` | Offline checks, including actual GLOP and GSCIP solves |

The execution path is `generate -> validate -> build -> solve -> evaluate -> save`.
Each failure includes its stage. CAFA 1.0's scripts are retained; the shared benchmark
loader now discovers nested MAMO files and preserves zero-valued answers.

## Install and run

From the repository root, using Python 3.10 or newer:

```powershell
python -m venv .venv-cafa2
.\.venv-cafa2\Scripts\python.exe -m pip install -r cafa_2.0/requirements.txt
.\.venv-cafa2\Scripts\python.exe cafa_2.0/cafa.py --dataset-root benchmarks --datasets NL4Opt --max-instances 5
```

Start LM Studio's OpenAI-compatible server and load your model first. The default
server is `http://localhost:1234/v1`. Set `--model-name` to the loaded model ID;
use `--base-url` for another compatible endpoint. Authentication is available through
`LM_STUDIO_API_KEY` or `--api-key`; the key is excluded from saved run settings.
The minimal dependencies above are sufficient; legacy notebook dependencies are unnecessary.

Useful options:

| Option | Behavior |
| --- | --- |
| `--datasets IndustryOR Mamo` | Select benchmark folders; names are case-sensitive |
| `--start-index 10 --stop-index 20` | Select a slice of the combined loaded instances |
| `--solver auto` | GLOP for LP; GSCIP for models containing integer variables |
| `--solver gurobi` | Use Gurobi through MathOpt; requires a usable Gurobi installation/license |
| `--time-limit 30` | Solver limit in seconds |
| `--execution-timeout 120` | Hard time limit for the entire model worker, including startup and construction |
| `--skip-solve` | Generate and syntax-check code, without executing it or scoring accuracy |
| `--zero-shot` | Keep the instructions and omit all examples |
| `--objective-key custom_score` | Identify the gold objective using an exact dictionary key |
| `--code-file path/to/00001.py` | Replay a model-building function on the selected instances without calling an LLM |

Keep the original dataset and instance slice when replaying saved code. Replay uses
the current CLI solver settings. Every selected instance receives the same function,
with that instance's input parameters passed as `data`.

## Inspect a result

Each run creates a UTC timestamped folder under `output/cafa_2.0`:

```text
run.json                 configuration and Python/library versions
results.jsonl            one completed record per line, flushed after each instance
summary.json             overall and per-dataset counts and objective accuracy
00001.prompt.txt         exact prompt, including the original text/table and input parameters
00001.response.txt       original model response
00001.py                 extracted function; comments, dictionaries and equalities preserved
00001.model.pbtxt        exported MathOpt model before solving
00001.result.json        status, objective, bound, variable values, scoring, artifact paths
```

Numbered files correspond to the order in `results.jsonl`; every record includes the
dataset name and original instance ID. Prompts are saved before HTTP requests.
Model exports are available after successful construction, including solver failures.
Replay creates code/result artifacts and exports the model when construction succeeds.

Generated code has one entry point, `build_model(data)`, and returns a named MathOpt
model. The runner supplies `mathopt`, the documented builtins, and input parameters.
All decision variables are read from the model using their unique names; no extra
variable registry is needed. `input_values` and, for NLP4LP, `parameters` are passed
to the function. Gold answers and reference implementations never enter the prompt.

The four fixed demonstrations cover continuous blending, integer production,
transportation with demand equalities, and facility activation with binary decisions
and continuous shipments. Tables remain in the supplied question format, including
Markdown or LaTeX; this version does not add OCR, a table parser, or example retrieval.

## Evaluation and scope

The supported mathematical scope is LP/IP/MILP with one linear objective. Unsupported
API constructs or explicit `NotImplementedError` responses are recorded as unsupported.
Nonlinear objectives are rejected after model export. Validation checks the output
contract; it does not prove that a formulation matches the question.

`correct=true` requires an optimal termination and a matching scalar gold objective
(`rel_tol=1e-4`, `abs_tol=1e-4`). A feasible incumbent remains visible and is not counted
as proven optimal. Infeasible/unbounded models have no fabricated objective value.
Accuracy measures objective agreement; it does not certify formulation equivalence.

Gold extraction accepts numeric scalars, one-element lists, explicit objective fields,
or an unambiguous objective-like dictionary key. Free-form answers and ambiguous
vectors require manual review or `--objective-key`. Their `correct` field is null.
`--skip-solve` records also have null correctness. These records are excluded from
the `scored` denominator, which is reported separately from `total`. Execution errors
and unsupported generations with a scalar gold objective count as incorrect.
ReSocratic source subsets remain identifiable through the original instance IDs.

Each generated function runs in a fresh process with a timeout and a restricted
modeling interface. This is a process boundary, not an operating-system security sandbox.
There is one generation attempt per instance; HTTP retries do not repair model code.

## Verify without an LLM server

```powershell
.\.venv-cafa2\Scripts\python.exe -B -m unittest discover -s cafa_2.0/tests -v
```

The tests solve all four prompt examples and check integer domains, equality balances,
infeasibility, nonlinear rejection, timeouts, replay, error artifacts, gold selection,
and the nested MAMO loader. They do not measure live LLM generation accuracy.

API reference: [Google MathOpt](https://developers.google.com/optimization/math_opt).
