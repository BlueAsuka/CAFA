"""Offline regression checks, including actual MathOpt LP and MILP solves."""
import json
import io
import subprocess
import sys
import tempfile
import unittest
import tokenize
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import asdict
from importlib.metadata import version
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cafa_2.0"))
sys.path.insert(0, str(ROOT))
import cafa
from benchmark_loader import BenchmarkInstance, ORBenchmarkLoader
from clients import CodeResponse, CAFAClient, ClientConfig, extract_code


def examples(style="mathopt"):
    text = (ROOT / "cafa_2.0" / "prompt" / f"{style}_prompt.txt").read_text(encoding="utf-8")
    result = {}
    for section in text.split("\nEXAMPLE: ")[1:]:
        name = section.splitlines()[0]
        description, code = section.split("\nCODE:\n", 1)
        question, data = description.split("\nQUESTION:\n", 1)[1].split("\nDATA:\n", 1)
        result[name] = (question, json.loads(data), code.strip())
    return result


class MathOptTests(unittest.TestCase):
    def setUp(self):
        self.examples = examples()
        self.config = cafa.RunConfig("unused")

    def test_all_prompt_examples_solve_with_expected_domains(self):
        expected = {"continuous_blend": (48, "glop"), "integer_production": (15, "gscip"),
                    "transportation_table": (7, "glop"), "facility_table_milp": (33, "gscip"),
                    "inventory_balance": (17, "glop"), "machine_scheduling": (5, "gscip")}
        for name, (_, data, code) in self.examples.items():
            with self.subTest(example=name):
                result = cafa.execute_code(code, data, self.config)
                self.assertEqual(result["status"], "optimal", result)
                self.assertAlmostEqual(result["prediction"], expected[name][0])
                self.assertEqual(result["solver"], expected[name][1])
                self.assertTrue(result["variable_values"])
                self.assertIn("variables", result["model_proto"])

    def test_integer_model_is_not_solved_as_a_relaxation(self):
        _, data, code = self.examples["integer_production"]
        result = cafa.execute_code(code, dict(data, hours=8), self.config)
        self.assertEqual(result["status"], "optimal", result)
        self.assertAlmostEqual(result["prediction"], 13)  # LP relaxation is 40/3.
        self.assertTrue(all(abs(v - round(v)) < 1e-6 for v in result["variable_values"].values()))
        self.config.solver = "glop"
        rejected = cafa.execute_code(code, data, self.config)
        self.assertEqual(rejected["status"], "error")
        self.assertIn("cannot solve integer", rejected["error"])

    def test_infeasible_model_keeps_termination_without_objective_access(self):
        code = '''m = mathopt.Model(name="infeasible")
x = m.add_variable(lb=0, ub=1, name="x")
m.add_linear_constraint(x >= 2, name="impossible_demand")
m.minimize(x)'''
        result = cafa.execute_code(code, {}, self.config)
        self.assertEqual(result["status"], "infeasible", result)
        self.assertIsNone(result["prediction"])
        self.assertNotIn("error", result)

    def test_nonlinear_objective_is_explicitly_unsupported(self):
        code = '''m = mathopt.Model(name="quadratic")
x = m.add_variable(lb=0, name="x")
m.minimize(x * x)'''
        result = cafa.execute_code(code, {}, self.config)
        self.assertEqual(result["status"], "unsupported", result)
        self.assertIn("model_proto", result)

    def test_worker_timeout_stops_generated_execution(self):
        self.config.execution_timeout = 2
        result = cafa.execute_code("while True:\n    pass", {}, self.config)
        self.assertEqual(result["status"], "timeout")

    def test_failed_build_keeps_code_prompt_and_result_for_review(self):
        instance = BenchmarkInstance("fixture", "bad_model", "Minimize cost", expected_answer=33)
        code = "m = {}"
        client = Mock()
        client.build_prompt.return_value = "reviewable prompt"
        client.generate_code.return_value = CodeResponse(code, "reviewable prompt", code)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = cafa.run_cafa([instance], client, self.config, folder)[0]
            self.assertEqual(result["stage"], "build")
            self.assertFalse(result["correct"])
            self.assertEqual((folder / result["artifacts"]["code"]).read_text(encoding="utf-8"), code)
            saved = json.loads((folder / result["artifacts"]["result"]).read_text(encoding="utf-8"))
            journal = json.loads((folder / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved, journal)

    def test_replay_preserves_table_model_and_equality_constraints(self):
        question, data, code = self.examples["facility_table_milp"]
        instance = BenchmarkInstance("fixture", "facility", question, expected_answer=33)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "saved.py"
            path.write_text(code, encoding="utf-8")
            self.config.code_file = str(path)
            result = cafa.run_instance(instance, None, self.config, folder, 1)
            self.assertTrue(result["correct"], result)
            values = result["variable_values"]
            self.assertAlmostEqual(values["ship_A_North"] + values["ship_B_North"], 6)
            self.assertAlmostEqual(values["ship_A_South"] + values["ship_B_South"], 5)
            self.assertEqual(values["open_A"], 1)
            self.assertEqual(values["open_B"], 1)
            self.assertTrue((folder / result["artifacts"]["model"]).exists())

    def test_feasible_incumbent_is_not_scored_as_an_optimum(self):
        _, _, code = self.examples["continuous_blend"]
        client = Mock()
        client.build_prompt.return_value = "prompt"
        client.generate_code.return_value = CodeResponse(code, "prompt", code)
        instance = BenchmarkInstance("fixture", "limited", "Minimize cost", expected_answer=48)
        with tempfile.TemporaryDirectory() as tmp, patch.object(cafa, "execute_code", return_value={
                "status": "feasible", "stage": "complete", "prediction": 48}):
            result = cafa.run_instance(instance, client, self.config, Path(tmp), 1)
        self.assertFalse(result["correct"])

    def test_client_request_and_saved_prompt_agree(self):
        import httpx
        from openai import OpenAI
        question, _, code = self.examples["facility_table_milp"]
        requests = []

        def complete(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": f"```python\n{code}\n```"}, "finish_reason": "stop"}]})

        client = CAFAClient(ClientConfig(provider="lmstudio", model_name="fixture", api_key="local"))
        client.client.close()
        client.client = OpenAI(base_url="http://fixture.test/v1", api_key="local",
                               http_client=httpx.Client(transport=httpx.MockTransport(complete)))
        instance = BenchmarkInstance("fixture", "completion", question, expected_answer=33)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = cafa.run_cafa([instance], client, self.config, folder)[0]
            self.assertTrue(result["correct"], result)
            saved_prompt = (folder / result["artifacts"]["prompt"]).read_text(encoding="utf-8")
            self.assertEqual(saved_prompt, requests[0]["messages"][1]["content"])
        client.client.close()

    def test_cli_replay_records_configuration_and_summary(self):
        question, _, code = self.examples["facility_table_milp"]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            dataset = folder / "IndustryOR"
            dataset.mkdir()
            (dataset / "dataset.jsonl").write_text(json.dumps({
                "en_question": question, "en_answer": "33", "question_type": "MIP"}), encoding="utf-8")
            code_file = folder / "model.py"
            code_file.write_text(code, encoding="utf-8")
            process = subprocess.run([sys.executable, str(ROOT / "cafa_2.0/cafa.py"),
                "--dataset-root", str(folder), "--datasets", "IndustryOR", "--max-instances", "1",
                "--output-dir", str(folder / "runs"), "--code-file", str(code_file), "--api-key", "TEST_SECRET"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            run_folder = next((folder / "runs").iterdir())
            settings_text = (run_folder / "run.json").read_text(encoding="utf-8")
            self.assertNotIn("TEST_SECRET", settings_text)
            self.assertEqual(json.loads(settings_text)["runtime"]["ortools"], version("ortools"))
            summary = json.loads((run_folder / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["overall"]["accuracy"], 1)
            self.assertEqual(summary["per_dataset"]["IndustryOR"]["correct"], 1)


class InputAndEvaluationTests(unittest.TestCase):
    def test_prompt_examples_have_no_comments_or_wrappers(self):
        for name, (_, _, code) in examples().items():
            with self.subTest(example=name):
                tokens = tokenize.generate_tokens(io.StringIO(code).readline)
                self.assertFalse(any(t.type == tokenize.COMMENT for t in tokens))
                cafa.validate_code(code)

    def test_gurobi_template_covers_the_same_questions_without_mathopt_calls(self):
        mathopt_examples = examples()
        gurobi_examples = examples("gurobi")
        self.assertEqual(set(mathopt_examples), set(gurobi_examples))
        for name, (question, data, code) in gurobi_examples.items():
            with self.subTest(example=name):
                self.assertEqual((question, data), mathopt_examples[name][:2])
                compile(code, "<gurobi_example>", "exec")
                self.assertNotIn("mathopt", code)
                self.assertIn("gp.Model(", code)
                self.assertIn("m.setObjective(", code)
                self.assertFalse(any(t.type == tokenize.COMMENT for t in
                                     tokenize.generate_tokens(io.StringIO(code).readline)))

    def test_collection_operations_and_ranged_constraints_are_supported(self):
        code = '''m = mathopt.Model(name="collections")
labels = " a,b ".strip().replace(" ", "").split(",")
items = []
items.append(labels[0])
items.extend(labels[1:])
seen = set()
seen.add("a")
bounds = {}
bounds.setdefault("a", 1)
bounds.update({"b": 2})
x = {i: m.add_variable(lb=0, is_integer=True, name=str(i)) for i in items}
m.add_linear_constraint(lb=2, expr=sum(x.values()), ub=3, name="range")
m.minimize(sum(bounds[i] * x[i] for i in items))'''
        result = cafa.execute_code(code, {}, cafa.RunConfig("unused"))
        self.assertEqual(result["status"], "optimal", result)
        self.assertEqual(result["prediction"], 2)

    def test_flat_contract_rejects_wrappers_and_runner_name_overwrites(self):
        for code in ("def build_model(data):\n    return mathopt.Model()",
                     "class Model: pass", "f = lambda: 1", "return m",
                     "data = {}", "mathopt = {}", "__builtins__ = {}", ""):
            with self.subTest(code=code), self.assertRaises(ValueError):
                cafa.validate_code(code)

    def test_missing_model_is_reported_as_build_failure(self):
        result = cafa.solve_request({"code": "x = 1", "data": {}})
        self.assertEqual(result["stage"], "build")
        self.assertIn("MathOpt Model in m", result["error"])

    def test_build_error_reports_generated_line(self):
        result = cafa.solve_request({"code": 'm = mathopt.Model()\nx = data["missing"]', "data": {}})
        self.assertEqual(result["error_line"], 2)

    def test_code_extraction_keeps_equality_braces_and_indentation(self):
        code = examples()["transportation_table"][2]
        self.assertEqual(extract_code(f"```python\n{code}\n```"), code)
        cafa.validate_code(code)
        with self.assertRaises(ValueError):
            cafa.validate_code("import os\n" + code)

    def test_gold_selection_does_not_guess_from_numbers_or_vectors(self):
        self.assertEqual(cafa.extract_scalar_answer({"Number of trips": "2", "Maximum profit": "48"}), 48)
        self.assertEqual(cafa.extract_scalar_answer([0]), 0)
        self.assertIsNone(cafa.extract_scalar_answer([3, 48]))
        self.assertIsNone(cafa.extract_scalar_answer("x=3, objective=48"))
        self.assertIsNone(cafa.extract_scalar_answer({"x": 3, "y": 48}))
        self.assertEqual(cafa.extract_scalar_answer({"custom_score": 48}, "custom_score"), 48)
        self.assertIsNone(cafa.extract_scalar_answer(float("inf")))

    def test_prompt_preserves_table_and_parameters_without_leaking_gold(self):
        instance = BenchmarkInstance("fixture", "one", "| A | 12 |", expected_answer="GOLD_SECRET",
                                     code_example="REFERENCE_SECRET", metadata={"input_values": {"hours": 8}})
        client = CAFAClient(ClientConfig(provider="lmstudio", model_name="fixture", api_key="local"))
        prompt = client.build_prompt(instance)
        self.assertIn(instance.question, prompt)
        self.assertIn('"hours": 8', prompt)
        self.assertNotIn("GOLD_SECRET", prompt)
        self.assertNotIn("REFERENCE_SECRET", prompt)
        self.assertIn("EXAMPLE: facility_table_milp", prompt)
        client.config.zero_shot = True
        self.assertNotIn("EXAMPLE:", client.build_prompt(instance))

    def test_nested_mamo_files_and_zero_answers_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Mamo/data/optimization/Easy_LP/test.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"id": 1, "Question": "Minimize cost", "Answer": 0}), encoding="utf-8")
            rows = list(ORBenchmarkLoader(tmp).iter_instances(["Mamo"]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].expected_answer, 0)

    def test_skipped_generation_is_unscored(self):
        _, _, code = examples()["continuous_blend"]
        client = Mock()
        client.build_prompt.return_value = "prompt"
        client.generate_code.return_value = CodeResponse(code, "prompt", code)
        instance = BenchmarkInstance("fixture", "skip", "Minimize cost", expected_answer=48)
        config = cafa.RunConfig("unused", skip_solve=True)
        with tempfile.TemporaryDirectory() as tmp, patch.object(cafa, "execute_code") as execute:
            result = cafa.run_instance(instance, client, config, Path(tmp), 1)
        execute.assert_not_called()
        self.assertEqual(result["status"], "generated")
        self.assertIsNone(result["correct"])
        self.assertIsNone(cafa.summarize_results([result])["overall"]["accuracy"])


class GurobiAdapterTests(unittest.TestCase):
    def test_native_statuses_bounds_artifacts_and_model_lifetimes(self):
        constants = SimpleNamespace(OPTIMAL=2, INFEASIBLE=3, INF_OR_UNBD=4, UNBOUNDED=5,
                                    TIME_LIMIT=9, INTEGER="I", BINARY="B", CONTINUOUS="C", INFINITY=1e100)
        for status, incumbent, expected in ((2, True, "optimal"), (9, True, "feasible"), (3, False, "infeasible")):
            with self.subTest(status=status):
                disposed = []
                parameters = {}

                class Environment:
                    def __init__(self, **kwargs): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): disposed.append("env")

                class Model:
                    NumQNZs = NumQConstrs = NumGenConstrs = NumSOS = 0
                    NumObj = 1
                    SolCount = int(incumbent)
                    Status = status
                    Runtime = 0.1
                    ObjBound = 6
                    def __init__(self, *args, **kwargs): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): disposed.append("model")
                    def update(self): pass
                    def write(self, path): Path(path).write_text("LP_MODEL", encoding="utf-8")
                    def getVars(self): return [SimpleNamespace(VarName="x", VType="I", X=7)]
                    def getConstrs(self): return [SimpleNamespace(ConstrName="demand")]
                    def setParam(self, key, value): parameters[key] = value
                    def optimize(self): pass
                    @property
                    def ObjVal(self):
                        if not incumbent:
                            raise AssertionError("Do not read an objective without an incumbent")
                        return 7

                module = SimpleNamespace(Env=Environment, Model=Model, GRB=constants, quicksum=sum)
                with patch.dict(sys.modules, {"gurobipy": module}):
                    result = cafa.solve_request(dict(code='m = gp.Model("fixture")', data={},
                                                    solver="auto", modeling_api="gurobi", time_limit=11))
                self.assertEqual(result["status"], expected, result)
                self.assertEqual(result["prediction"], 7 if incumbent else None)
                self.assertEqual(result["model_text"], "LP_MODEL")
                self.assertEqual(result["model_type"], "IP")
                self.assertEqual(parameters["TimeLimit"], 11)
                self.assertEqual(disposed, ["model", "env"])
                self.assertEqual(result["variable_values"], {"x": 7} if incumbent else {})

    def test_gurobi_validation_allows_examples_and_blocks_solver_calls(self):
        for _, _, code in examples("gurobi").values():
            cafa.validate_code(code, "gurobi")
        for code in ('m.optimize()', 'm.write("file")', 'm.setParam("TimeLimit", 0)', 'gp = {}', 'GRB = {}'):
            with self.subTest(code=code), self.assertRaises((ValueError, NotImplementedError)):
                cafa.validate_code(code, "gurobi")

    def test_native_model_export_is_saved_as_lp(self):
        code = 'm = gp.Model("fixture")'
        client = Mock()
        client.build_prompt.return_value = "Gurobi template"
        client.generate_code.return_value = CodeResponse(code, "Gurobi template", code)
        with tempfile.TemporaryDirectory() as tmp, patch.object(cafa, "execute_code", return_value={
                "status": "optimal", "stage": "complete", "prediction": 7, "model_text": "LP_MODEL"}):
            result = cafa.run_instance(BenchmarkInstance("fixture", "native", "Minimize cost", expected_answer=7),
                client, cafa.RunConfig("unused", modeling_api="gurobi"), Path(tmp), 1)
            self.assertTrue(result["correct"])
            self.assertEqual(result["artifacts"]["model"], "00001.model.lp")
            self.assertEqual((Path(tmp) / "00001.model.lp").read_text(encoding="utf-8"), "LP_MODEL")


class RepairTests(unittest.TestCase):
    def test_build_failure_is_repaired_and_all_attempts_are_saved(self):
        code = examples()["inventory_balance"][2]
        instance = BenchmarkInstance("fixture", "repair", "Minimize the specified inventory cost", expected_answer=17)
        client = Mock()
        client.build_prompt.return_value = "initial prompt"
        client.build_repair_prompt.return_value = "repair prompt"
        client.generate_code.side_effect = [CodeResponse("m = {}", "initial prompt", "m = {}"),
                                            CodeResponse(code, "repair prompt", code)]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = cafa.run_instance(instance, client, cafa.RunConfig("unused", repair_attempts=1), folder, 1)
            self.assertTrue(result["correct"], result)
            self.assertNotIn("error", result)
            self.assertEqual(result["generation_attempts"], 2)
            self.assertEqual([a["status"] for a in result["attempts"]], ["error", "optimal"])
            self.assertEqual((folder / result["attempts"][0]["artifacts"]["code"]).read_text(encoding="utf-8"), "m = {}")
            self.assertEqual((folder / result["artifacts"]["code"]).read_text(encoding="utf-8"), code)
            self.assertTrue((folder / result["attempts"][1]["artifacts"]["model"]).exists())
            self.assertEqual(json.loads((folder / result["artifacts"]["result"]).read_text(encoding="utf-8")), result)
        feedback = client.build_repair_prompt.call_args.args[2]
        self.assertNotIn("correct", feedback)
        self.assertNotIn("gold_scalar", feedback)
        summary = cafa.summarize_results([result])["overall"]
        self.assertEqual(summary["first_attempt_statuses"], {"error": 1})
        self.assertEqual(summary["generation_attempts"], 2)

    def test_optimal_mismatch_does_not_trigger_gold_guided_repair(self):
        code = examples()["continuous_blend"][2]
        client = Mock()
        client.build_prompt.return_value = "prompt"
        client.generate_code.return_value = CodeResponse(code, "prompt", code)
        instance = BenchmarkInstance("fixture", "wrong", "Minimize cost", expected_answer=987654321)
        with tempfile.TemporaryDirectory() as tmp, patch.object(cafa, "execute_code", return_value={
                "status": "optimal", "stage": "complete", "prediction": 48}):
            result = cafa.run_instance(instance, client, cafa.RunConfig("unused", repair_attempts=2), Path(tmp), 1)
        self.assertFalse(result["correct"])
        self.assertEqual(client.generate_code.call_count, 1)
        client.build_repair_prompt.assert_not_called()
        self.assertEqual(cafa.summarize_results([result])["overall"]["optimal_mismatch"], 1)

    def test_repair_budget_is_bounded_and_never_applies_to_auth_failures(self):
        instance = BenchmarkInstance("fixture", "failure", "Minimize cost")
        for api_failure in (False, True):
            with self.subTest(api_failure=api_failure), tempfile.TemporaryDirectory() as tmp:
                client = Mock()
                client.build_prompt.return_value = "prompt"
                client.build_repair_prompt.return_value = "repair prompt"
                if api_failure:
                    client.generate_code.side_effect = RuntimeError("API authentication failed")
                else:
                    client.generate_code.return_value = CodeResponse("m = {}", "prompt", "m = {}")
                with patch.object(cafa, "execute_code", return_value={"status": "error", "stage": "build", "error": "Bad model"}):
                    result = cafa.run_instance(instance, client, cafa.RunConfig("unused", repair_attempts=1), Path(tmp), 1)
                self.assertEqual(client.generate_code.call_count, 1 if api_failure else 2)
                self.assertEqual(result["status"], "error")


class ClientTests(unittest.TestCase):
    def test_prompt_selection_and_custom_template(self):
        instance = BenchmarkInstance("fixture", "prompt", "Minimize cost")
        with tempfile.TemporaryDirectory() as tmp, patch("clients.OpenAI") as sdk:
            custom = Path(tmp) / "custom.txt"
            custom.write_text("CUSTOM GUROBI TEMPLATE\nEXAMPLES:\nOMIT_ME", encoding="utf-8")
            for style in ("mathopt", "gurobi"):
                config = ClientConfig(model_name="fixture", api_key="fixture", prompt_style=style)
                client = CAFAClient(config)
                prompt = client.build_prompt(instance)
                self.assertIn("gurobipy" if style == "gurobi" else "MathOpt", prompt)
                sdk.return_value.chat.completions.create.return_value = Mock(choices=[
                    Mock(finish_reason="stop", message=Mock(content="m = {}"))])
                client.generate_code(instance)
                system = sdk.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"]
                self.assertIn("gurobipy" if style == "gurobi" else "MathOpt", system)
            client = CAFAClient(ClientConfig(model_name="fixture", api_key="fixture", prompt_style="gurobi",
                                            prompt_file=str(custom), zero_shot=True))
            self.assertIn("CUSTOM GUROBI TEMPLATE", client.build_prompt(instance))
            self.assertNotIn("OMIT_ME", client.build_prompt(instance))

    def test_cli_prompt_selects_matching_runtime(self):
        for style in ("mathopt", "gurobi"):
            with self.subTest(style=style), patch.object(sys, "argv", ["cafa", "--dataset-root", "unused", "--prompt", style]):
                run, client = cafa.parse_args()
                self.assertEqual(run.modeling_api, style)
                self.assertEqual(client.prompt_style, style)
                self.assertEqual(client.prompt_path.name, f"{style}_prompt.txt")
        with patch.object(sys, "argv", ["cafa", "--dataset-root", "unused", "--prompt", "gurobi", "--solver", "glop"]), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                cafa.parse_args()

    def test_repair_prompt_filters_gold_reference_and_unrelated_fields(self):
        instance = BenchmarkInstance("fixture", "repair", "Minimize cost", expected_answer="GOLD_SECRET",
                                     code_example="REFERENCE_SECRET")
        with patch("clients.OpenAI"):
            client = CAFAClient(ClientConfig(model_name="fixture", api_key="fixture"))
        prompt = client.build_repair_prompt(instance, "m = {}", dict(stage="build", status="error",
            error="Model missing", error_line=1, expected_answer="GOLD_SECRET", model_proto="UNRELATED_PROTO"))
        self.assertIn("Model missing", prompt)
        self.assertIn('"error_line": 1', prompt)
        for secret in ("GOLD_SECRET", "REFERENCE_SECRET", "UNRELATED_PROTO"):
            self.assertNotIn(secret, prompt)

    def test_provider_configuration_and_override_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {}, clear=True):
            env = Path(tmp) / ".env"
            env.write_text('OPENAI_MODEL="openai-fixture"\nOPENAI_API_KEY=file-secret\n'
                           'DEEPSEEK_MODEL=deepseek-fixture\nDEEPSEEK_API_KEY=deepseek-secret\n'
                           'LMSTUDIO_MODEL=local-fixture\nLMSTUDIO_BASE_URL=http://fixture.test/v1\n',
                           encoding="utf-8")
            expected = {"openai": ("openai-fixture", "https://api.openai.com/v1", "file-secret"),
                        "deepseek": ("deepseek-fixture", "https://api.deepseek.com", "deepseek-secret"),
                        "lmstudio": ("local-fixture", "http://fixture.test/v1", "lm-studio")}
            for provider, values in expected.items():
                config = ClientConfig(provider=provider, env_file=str(env))
                self.assertEqual((config.model_name, config.base_url, config.api_key), values)
                self.assertNotIn(values[2], repr(config))
            with patch.dict("os.environ", {"OPENAI_MODEL": "environment-model", "OPENAI_API_KEY": "environment-secret"}):
                config = ClientConfig(env_file=str(env))
                self.assertEqual(config.model_name, "openai-fixture")
                self.assertEqual(config.api_key, "file-secret")
                explicit = ClientConfig(env_file=str(env), model_name="cli-model", api_key="cli-secret",
                                        base_url="http://override.test/v1")
                self.assertEqual(explicit.model_name, "cli-model")
                self.assertEqual(explicit.api_key, "cli-secret")
                self.assertEqual(explicit.base_url, "http://override.test/v1")
            import os
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_missing_remote_credentials_fail_before_request(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_MODEL"):
                CAFAClient(ClientConfig(env_file="nonexistent.env"))
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                CAFAClient(ClientConfig(provider="deepseek", model_name="fixture", env_file="nonexistent.env"))

    def test_cli_selects_provider_and_env_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {}, clear=True):
            env = Path(tmp) / ".env"
            env.write_text("DEEPSEEK_MODEL=fixture-model\nDEEPSEEK_API_KEY=fixture-secret\n", encoding="utf-8")
            with patch.object(sys, "argv", ["cafa", "--dataset-root", "unused", "--provider", "deepseek",
                                           "--env-file", str(env), "--max-tokens", "4096"]):
                _, config = cafa.parse_args()
            self.assertEqual(config.provider, "deepseek")
            self.assertEqual(config.model_name, "fixture-model")
            self.assertEqual(config.max_tokens, 4096)
            self.assertNotIn("fixture-secret", json.dumps({k: v for k, v in asdict(config).items() if k != "api_key"}))

    def test_provider_http_requests_use_selected_model_auth_and_token_parameter(self):
        import httpx
        from openai import OpenAI
        instance = BenchmarkInstance("fixture", "request", "Minimize cost")
        for provider in ("openai", "deepseek", "lmstudio"):
            with self.subTest(provider=provider):
                requests = []

                def complete(request):
                    requests.append(request)
                    return httpx.Response(200, json={"choices": [{"message": {
                        "role": "assistant", "content": 'm = mathopt.Model(name="fixture")'}, "finish_reason": "stop"}]})

                config = ClientConfig(provider=provider, model_name="selected-model", api_key="fixture-secret",
                                      base_url="https://fixture.test/v1", max_tokens=2048)
                client = CAFAClient(config)
                client.client.close()
                client.client = OpenAI(base_url=config.base_url, api_key=config.api_key,
                    http_client=httpx.Client(transport=httpx.MockTransport(complete)))
                try:
                    response = client.generate_code(instance)
                    request = requests[0]
                    body = json.loads(request.content)
                    self.assertEqual(request.headers["authorization"], "Bearer fixture-secret")
                    self.assertEqual(request.url.path, "/v1/chat/completions")
                    self.assertEqual(body["model"], "selected-model")
                    self.assertEqual(body["messages"][1]["content"], response.prompt)
                    self.assertNotIn("temperature", body)
                    token_key = "max_completion_tokens" if provider == "openai" else "max_tokens"
                    self.assertEqual(body[token_key], 2048)
                    self.assertNotIn("max_tokens" if provider == "openai" else "max_completion_tokens", body)
                    cafa.validate_code(response.code)
                finally:
                    client.client.close()

    def test_explicit_temperature_and_invalid_completions(self):
        config = ClientConfig(model_name="fixture", api_key="fixture", temperature=0)
        with patch("clients.OpenAI") as sdk:
            client = CAFAClient(config)
            create = sdk.return_value.chat.completions.create
            for choices, message in (([], "no completion choices"),
                    ([Mock(finish_reason="length")], "truncated"),
                    ([Mock(finish_reason="stop", message=Mock(content=""))], "no modeling code")):
                create.return_value = Mock(choices=choices)
                with self.assertRaisesRegex(ValueError, message):
                    client.generate_code(BenchmarkInstance("fixture", "bad", "Minimize cost"))
            self.assertEqual(create.call_args.kwargs["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
