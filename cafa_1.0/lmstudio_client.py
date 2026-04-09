from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from benchmark_loader import BenchmarkInstance


PROMPT_TEMPLATE = """You are an expert in optimization problems and domain specific language generation.

Your task is to convert the textual optimization problem into lines of Gurobi Python code.

Rules:
1. Use the variable name `m` as the Gurobi model.
2. Use `gp.GRB.INTEGER` or `gp.GRB.CONTINUOUS` for decision variables.
3. Add only variable declarations, the objective, and constraints.
4. Do not include imports, model creation, comments, markdown fences, helper functions, or explanations.
5. Output code only.

Examples:
QUESTION:
A car manufacturer makes two types of car oils: Oil Max and Oil Max Pro. A container of Oil Max contains 46 grams of substance A, 43 grams of substance B and 56 grams of substance C. A container of Oil Max Pro contains 13 grams of substance A, 4 grams of substance B and 45 grams of substance C. The car manufacturer has 1345 grams of substance A, 346 grams of substance B, 1643 grams of substance C. In addition, the profit per container of Oil Max is $10 and the profit per container of Oil Max Pro is $15. How many containers of each oil should the car manufacturer make to maximize profit?
CODE:
x = m.addVar(name=\"Oil Max\", vtype=gp.GRB.INTEGER)
y = m.addVar(name=\"Oil Max Pro\", vtype=gp.GRB.INTEGER)
m.setObjective(10 * x + 15 * y, gp.GRB.MAXIMIZE)
m.addConstr(46 * x + 13 * y <= 1345)
m.addConstr(43 * x + 4 * y <= 346)
m.addConstr(56 * x + 45 * y <= 1643)

QUESTION:
Ben is growing apples and pears on his orchard. He has 50 acres available on which he must grow a minimum of 5 acres of apples and a minimum of 10 acres of pears to meet demands. The profit per apple is $2 and the profit per pear is $4. He prefers to grow more pears than apples but limitations in his workforce allow him to grow at most twice the amount of pears as apples. How many of each fruit should Ben grow in order to maximize his profit? What is that profit?
CODE:
x = m.addVar(name=\"apples\", vtype=gp.GRB.INTEGER)
y = m.addVar(name=\"pears\", vtype=gp.GRB.INTEGER)
m.setObjective(2 * x + 4 * y, gp.GRB.MAXIMIZE)
m.addConstr(x + y <= 50)
m.addConstr(x >= 5)
m.addConstr(y >= 10)
m.addConstr(y <= 2 * x)

Now solve the following instance.
QUESTION:
{question}
"""


@dataclass
class LMStudioConfig:
    model_name: str = "lmstudio-community/qwen/qwen3-4b-2507"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    temperature: float = 0.0
    timeout: int = 120
    max_retries: int = 3
    system_message: str = "Return only the requested code."


class CodeResponse(BaseModel):
    code: str
    prompt: str
    raw_text: str


class LMStudioCAFAClient:
    """Small LM Studio client for CAFA-style code generation across benchmark datasets."""

    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def build_prompt(self, instance: BenchmarkInstance) -> str:
        return PROMPT_TEMPLATE.format(question=instance.question)

    def generate_code(self, instance: BenchmarkInstance) -> CodeResponse:
        prompt = self.build_prompt(instance)
        raw_text = self._chat(prompt)
        code = self._extract_code(raw_text)
        if not code.strip():
            raise RuntimeError(f"Empty code returned for {instance.instance_id}")
        return CodeResponse(code=code, prompt=prompt, raw_text=raw_text)

    def _chat(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "system", "content": self.config.system_message},
                        {"role": "user", "content": prompt},
                    ],
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logging.warning("LM Studio attempt %d failed: %s", attempt, exc)
        raise RuntimeError(f"LM Studio request failed after {self.config.max_retries} attempts: {last_error}")

    @staticmethod
    def _extract_code(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text
