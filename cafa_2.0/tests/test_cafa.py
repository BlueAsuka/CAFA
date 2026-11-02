"""Offline regression checks, including actual MathOpt LP and MILP solves."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cafa_2.0"))
sys.path.insert(0, str(ROOT))
import cafa
from benchmark_loader import BenchmarkInstance, ORBenchmarkLoader
from lmstudio_client import CodeResponse, LMStudioCAFAClient, LMStudioConfig, extract_code


def examples():
    text = (ROOT / "cafa_2.0" / "prompt.txt").read_text(encoding="utf-8")
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
                    "transportation_table": (7, "glop"), "facility_table_milp": (33, "gscip")}
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
        code = '''def build_model(data):
    m = mathopt.Model(name="infeasible")
    x = m.add_variable(lb=0, ub=1, name="x")
    m.add_linear_constraint(x >= 2, name="impossible_demand")
    m.minimize(x)
    return m'''
        result = cafa.execute_code(code, {}, self.config)
        self.assertEqual(result["status"], "infeasible", result)
        self.assertIsNone(result["prediction"])
        self.assertNotIn("error", result)

    def test_nonlinear_objective_is_explicitly_unsupported(self):
        code = '''def build_model(data):
    m = mathopt.Model(name="quadratic")
    x = m.add_variable(lb=0, name="x")
    m.minimize(x * x)
    return m'''
        result = cafa.execute_code(code, {}, self.config)
        self.assertEqual(result["status"], "unsupported", result)
        self.assertIn("model_proto", result)

    def test_worker_timeout_stops_generated_execution(self):
        self.config.execution_timeout = 2
        result = cafa.execute_code("def build_model(data):\n    while True:\n        pass", {}, self.config)
        self.assertEqual(result["status"], "timeout")

    def test_failed_build_keeps_code_prompt_and_result_for_review(self):
        instance = BenchmarkInstance("fixture", "bad_model", "Minimize cost", expected_answer=33)
        code = "def build_model(data):\n    return {}"
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

        client = LMStudioCAFAClient(LMStudioConfig())
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
            self.assertEqual(json.loads(settings_text)["runtime"]["ortools"], "9.15.6755")
            summary = json.loads((run_folder / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["overall"]["accuracy"], 1)
            self.assertEqual(summary["per_dataset"]["IndustryOR"]["correct"], 1)


class InputAndEvaluationTests(unittest.TestCase):
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
        client = LMStudioCAFAClient(LMStudioConfig())
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


if __name__ == "__main__":
    unittest.main()
