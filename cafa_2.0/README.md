# CAFA 2.0

A compact MathOpt implementation following CAFA 1.0's client-and-runner structure.
The runtime has two Python modules, with the full prompt in a separate text file.

| File | Responsibility |
| --- | --- |
| `clients.py` | Load provider settings, build the prompt, call OpenAI/DeepSeek/LM Studio, extract surrounding code fences |
| `cafa.py` | Validate, build, solve, evaluate, and save each instance |
| `prompt/mathopt_prompt.txt` | Default MathOpt contract and six authored LP/IP/MILP demonstrations |
| `prompt/gurobi_prompt.txt` | Equivalent demonstrations using Gurobi's `gurobipy` API |
| `tests/test_cafa.py` | Offline checks, including actual GLOP and GSCIP solves |

The execution path is `generate -> validate -> build -> solve -> evaluate -> save`.
Each failure includes its stage. CAFA 1.0's scripts are retained; the shared benchmark
loader now discovers nested MAMO files and preserves zero-valued answers.

The original `prompt.txt` has moved to `prompt/mathopt_prompt.txt`. Use `--prompt
mathopt` (the default) or `--prompt gurobi` to select a template and its matching
code execution API. Both templates request flat code without comments, docstrings,
or functions and leave the model in `m`. The runner supplies `mathopt` for MathOpt
code, or `gp` (`gurobipy`) and `GRB` for native Gurobi code, plus `data` for either.

From `cafa_2.0` in your activated Conda environment:

```powershell
python cafa.py --dataset-root ..\benchmarks --datasets IndustryOR --provider deepseek --prompt mathopt --max-instances 100
python cafa.py --dataset-root ..\benchmarks --datasets IndustryOR --provider deepseek --prompt gurobi --max-instances 100
```

`--prompt gurobi` uses `prompt/gurobi_prompt.txt` and Gurobi as the solver, with
`--solver auto` or `--solver gurobi`. Install `gurobipy` in your active environment
(`python -m pip install gurobipy`) and configure a valid Gurobi license. Initialization
is checked before generation so a missing package or expired license does not waste
LLM calls. `--skip-solve` can generate and validate Gurobi code without a license.
Native Gurobi models are saved as `00001.model.lp`; MathOpt exports remain `.pbtxt`.

For a custom file use `--prompt-file path/to/template.txt` together with the matching
API, for example `--prompt gurobi --prompt-file prompt/gurobi_prompt.txt`. Relative
custom paths resolve from your working directory; built-in templates resolve from
the code directory. The selected file and its hash are recorded in `run.json`.
`--solver gurobi` alone continues to select Gurobi through MathOpt; `--prompt gurobi`
selects native `gurobipy` code. Replay also needs the matching `--prompt` selection.

## Install and run

From the repository root, using Python 3.10 or newer:

```powershell
python -m venv .venv-cafa2
.\.venv-cafa2\Scripts\python.exe -m pip install -r cafa_2.0/requirements.txt
.\.venv-cafa2\Scripts\python.exe cafa_2.0/cafa.py --dataset-root benchmarks --datasets NL4Opt --max-instances 5
```

The default provider is OpenAI. Model IDs and credentials are read from the repository's
`configs/.env`, independently of the working directory. The file uses these names:

```dotenv
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=your-openai-model
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_MODEL=your-deepseek-model
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=your-loaded-local-model
LMSTUDIO_API_KEY=lm-studio
```

Use the exact model IDs available to your account. Your existing `.env` is retained.
Choose a provider explicitly:

```powershell
python cafa_2.0/cafa.py --dataset-root benchmarks --datasets NL4Opt --provider openai --max-instances 5
python cafa_2.0/cafa.py --dataset-root benchmarks --datasets NL4Opt --provider deepseek --max-instances 5
python cafa_2.0/cafa.py --dataset-root benchmarks --datasets NL4Opt --provider lmstudio --max-instances 5
```

Settings precedence is explicit CLI options, `.env`, then process environment.
This prevents an inherited system API key from overriding the key in your configuration file.
`--env-file path/to/.env` selects a different configuration file. Optional
`OPENAI_BASE_URL` and `DEEPSEEK_BASE_URL` override their standard remote endpoints;
`--base-url`, `--model-name`, and `--api-key` override the selected provider's settings.
LM Studio requires its server running with the configured model loaded; its key
defaults to `lm-studio` if unset (`LM_STUDIO_API_KEY` is also accepted).
Remote providers require a configured model and key. Keys are excluded from saved
run settings and configuration representations.

All providers use the OpenAI SDK's Chat Completions interface. Temperature is omitted
by default so models with restricted sampling parameters can use their defaults;
set `--temperature` only when supported by your selected model. `--max-tokens` uses
`max_completion_tokens` for OpenAI and `max_tokens` for DeepSeek/LM Studio.
The minimal dependencies above are sufficient; legacy notebook dependencies are unnecessary.

Useful options:

| Option | Behavior |
| --- | --- |
| `--datasets IndustryOR Mamo` | Select benchmark folders; names are case-sensitive |
| `--start-index 10 --stop-index 20` | Select a slice of the combined loaded instances |
| `--solver auto` | GLOP for LP; GSCIP for models containing integer variables |
| `--prompt mathopt` | Generate MathOpt code using `prompt/mathopt_prompt.txt` (default) |
| `--prompt gurobi` | Generate and execute native Gurobi code using `prompt/gurobi_prompt.txt` |
| `--prompt-file path/to/template.txt` | Override the template while retaining the API selected by `--prompt` |
| `--solver gurobi` | Use Gurobi through MathOpt; requires a usable Gurobi installation/license |
| `--time-limit 30` | Solver limit in seconds |
| `--execution-timeout 120` | Hard time limit for the entire model worker, including startup and construction |
| `--skip-solve` | Generate and syntax-check code, without executing it or scoring accuracy |
| `--zero-shot` | Keep the instructions and omit all examples |
| `--objective-key custom_score` | Identify the gold objective using an exact dictionary key |
| `--provider openai/deepseek/lmstudio` | Select provider-specific model, endpoint, and credentials |
| `--env-file path/to/.env` | Override the default `configs/.env` file |
| `--max-tokens 8192` | Set the output budget, including reasoning tokens where applicable |
| `--repair-attempts 1` | Allow one extra generation using validation/build/solver feedback; adds API calls |
| `--code-file path/to/00001.py` | Replay flat model-building statements on the selected instances without calling an LLM |

Keep the original dataset and instance slice when replaying saved code. Replay uses
the current CLI solver settings. Every selected instance executes the same code,
with that instance's input parameters supplied as `data`. Replay needs no API key or model.

## Inspect a result

Each run creates a UTC timestamped folder under `output/cafa_2.0`:

```text
run.json                 configuration and Python/library versions
prompt_template.txt      exact prompt template used by this run
results.jsonl            one completed record per line, flushed after each instance
summary.json             overall and per-dataset counts and objective accuracy
00001.prompt.txt         exact prompt, including the original text/table and input parameters
00001.response.txt       original model response
00001.py                 extracted statements; dictionaries and equalities preserved
00001.model.pbtxt        exported MathOpt model before solving
00001.result.json        status, objective, bound, variable values, scoring, artifact paths
```

Numbered files correspond to the order in `results.jsonl`; every record includes the
dataset name and original instance ID. Prompts are saved before HTTP requests.
Model exports are available after successful construction, including solver failures.
Replay creates code/result artifacts and exports the model when construction succeeds.

Generated code consists of top-level Python statements and leaves a named model
in `m`. Functions, classes, lambdas, imports, and returns are rejected. For MathOpt,
the runner supplies `mathopt`, the documented builtins, and input parameters as `data`:

```python
m = mathopt.Model(name="production")
x = m.add_integer_variable(lb=0, name="units")
m.add_linear_constraint(2 * x <= data["hours"], name="labor")
m.maximize(3 * x)
```

Loops, dictionaries, and comprehensions remain available for indexed formulations.
The runner executes these statements in one shared namespace so comprehensions
can access model variables and input parameters.
All decision variables are read from the model using their unique names; no extra
variable registry is needed. `input_values` and, for NLP4LP, `parameters` are passed
as `data`. Gold answers and reference implementations never enter the prompt.
Previously saved `build_model(data)` functions must be flattened before replay:
remove the function declaration and `return m`, and dedent the body once.

The six fixed demonstrations cover continuous blending, integer production,
transportation with demand equalities, and facility activation with binary decisions
and continuous shipments, plus inventory balances and disjunctive machine scheduling.
Generated output must contain no comments, docstrings, or Markdown. Loops and
comprehensions remain valid. The modeling checklist covers domains, units, logical
linearization, objective constants, and exact MathOpt call signatures.
Tables remain in the supplied question format, including
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

Each generated script runs in a fresh process with a timeout and a restricted
modeling interface. This is a process boundary, not an operating-system security sandbox.
By default there is one generation attempt per instance. `--repair-attempts 1`
allows one extra request after validation/build/solver failures, unsupported output,
or infeasible/unbounded termination. These statuses can also describe a legitimate
formulation: the repair instructions prohibit inventing data or weakening requirements
just to get an optimum. A repair cannot certify that a model matches its question.
API/authentication errors and worker timeouts do not trigger code repairs. Replay
and `--skip-solve` do not request repairs. HTTP retries are separate from code repairs.

Repairs never use gold answers, correctness labels, or reference code. An optimal
model is not retried because it disagrees with a gold objective. Every repair-enabled
run saves numbered `00001.attempt01.*`, `00001.attempt02.*` snapshots, and the result's
`attempts` array records statuses, objectives, diagnostics, and artifact paths.
The ordinary `00001.*` artifacts describe the final attempt. `summary.json` includes
first-attempt status counts, total generation calls, and `optimal_mismatch` counts.
`run.json` includes the template's SHA-256 hash for prompt comparisons.

From `cafa_2.0` with your Conda environment active, compare the updated prompt first:

```powershell
python cafa.py --dataset-root ..\benchmarks --datasets IndustryOR --provider deepseek --max-instances 100 --output-dir ..\output\industryor_prompt
```

Then repeat with `--repair-attempts 1` and a different output directory to measure
the effect of execution feedback separately. Use `--provider openai` for the
OpenAI model configured in `.env`. Keep dataset slices, solver limits, model IDs,
and sampling settings the same. Extra calls increase cost and run time.
See [the saved-run diagnosis](PERFORMANCE.md) for the observed baseline limitations.

## Verify without an LLM server

```powershell
.\.venv-cafa2\Scripts\python.exe -B -m unittest discover -s cafa_2.0/tests -v
```

The tests solve all six prompt examples and check integer domains, equality balances,
infeasibility, nonlinear rejection, timeouts, replay, error artifacts, gold selection,
and the nested MAMO loader, plus repair budgets, attempt artifacts, gold isolation,
comment-free examples, collection operations, and generated-code line diagnostics.
Provider checks use mocked HTTP responses to verify
configuration precedence, authentication, model IDs, and request parameters.
Native Gurobi result handling, cleanup, and LP artifact checks use a mocked solver;
live Gurobi solves require a valid license. Both templates are syntax checked.
They do not measure live LLM generation accuracy or remote model availability.

API reference: [Google MathOpt](https://developers.google.com/optimization/math_opt).
Client reference: [OpenAI Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).
Gurobi template reference: [gurobipy Model API](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html).
