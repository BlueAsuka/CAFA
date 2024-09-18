import re
import json
import loguru
import random
import numpy as np

from typing import List, Dict, Tuple


cfg = json.load(open('../configs./configs.json', 'r'))


def read_txt_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        loguru.logger.debug(f"Reading file: {file_path}")
        lines = f.readlines()
        loguru.logger.debug(f"File read successfully: {file_path}")
        return lines
    
def read_json_file(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        loguru.logger.debug(f"Reading file: {file_path}")
        data = json.load(f)
        loguru.logger.debug(f"File read successfully: {file_path}")
        return data
    
def get_nl4opt_qas(data: List[str], count: bool=True) -> Tuple[List[str], List[str]]:
    questions, answers = [], []
    for d in data:
        qa_dict = json.loads(d)
        questions.append(qa_dict['en_question'])
        answer = qa_dict['en_answer'] if qa_dict['en_answer'] != 'No Best Solution' else np.inf
        answers.append(answer)
        
    if count:
        loguru.logger.info(f"Number of questions: {len(questions)}")
        loguru.logger.info(f"Number of answers: {len(answers)}")
    return questions, answers

def get_demo_and_test_samples(data: List, count: bool=True) -> Tuple[List[str], List[str]]:
    rng = random.Random(42)
    rng.shuffle(data)
    
    demo_samples = data[:cfg["DEMO_SAMPLES"]]
    test_samples = data[cfg["DEMO_SAMPLES"]:]
    
    if count:
        loguru.logger.info(f"Number of demo samples: {len(demo_samples)}")
        loguru.logger.info(f"Number of test samples: {len(test_samples)}")
    
    return demo_samples, test_samples

def save_test_questions_and_answers(questions: List[str], answers: List[str], file_path: str) -> None:
    with open(file_path, 'w') as f:
        for q, a in zip(questions, answers):
            f.write(f"{q} @ {a}\n")
            
prefix = """
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()
m = gp.Model(env=env)
"""
                
suffix = """
m.optimize()
"""

def complement_code(code: str) -> float:
    return prefix + code + suffix

def clean_code(code: str) -> str:
    cleand_code = []
    for line in code.split('\n'):
        line = line.strip()
        if line.startswith('m.addConstr') and not re.findall(r'<=|>=', line):
            line = re.sub(r'<', r'<=', line)
            line = re.sub(r'>', r'>=', line)
        cleand_code.append(line)
    return '\n'.join(cleand_code)

def execute_code(code: str) -> float:
    ex_locals = {}
    exec(code, None, ex_locals)
    
    try:
        return ex_locals["m"].objVal
    except Exception as e:
        # print(e)
        return np.inf

def get_pred_answers(codes: List[str]) -> List[str]:
    pred_answers = []
    for i, code_str in enumerate(codes):
        try:
            cleaned_code = clean_code(code_str)
            code = complement_code(cleaned_code)
            ans = execute_code(code)
            loguru.logger.info(f"question {i} obtain answer")
            pred_answers.append(ans)
        except Exception as e:
            loguru.logger.error(f"Error for question {i}: {e}")
            pred_answers.append("Error")
    return pred_answers

def mark(pred, real, error: float) -> List[bool]:    
    correct = []
    for p, r in zip(pred, real):
        if p == 'Error':
            continue
        if (float(p) == np.inf and float(r) == np.inf) or (abs(float(p) - float(r)) / float(r) < error):
            correct.append(True)
        else:
            correct.append(False)
    return correct