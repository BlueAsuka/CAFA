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
