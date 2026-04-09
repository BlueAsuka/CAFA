# CAFA

## News
- [2026.04/09] Tidy up the CAFA code from the notebook with runnable scripts in the `cafa_1.0` folder. Also, add the benckmarks from the [LLM4OR](https://llm4or.github.io/LLM4OR/) repo to allow running the CAFA on more datasets.
- [2025.10.13] Use Qwen3-4B-2507-Q8_0 as the LLM model with CAFA achieving 69.79% accuracy on the NLP4Opt dataset with 99.63% successfully rate. The model size is only 4.28 GB which can be run on many cosumer GPU with over 6 GB VRAM. This shows a promising direction to further develop CAFA for effective LP optimization on the edge efficiently.

## Introduction
The code for the Neurips 2024 MATH-AI workshop paper

[Code as Auto-Formulation can boost large language model in solving linear problem](https://openreview.net/forum?id=xC2xtBLmri)

This paper proposed a compact prompting for large language models to solve linear programming problems. Instead of using a multi-agent framework with a complex workflow, we guide the LLM to autoformulate the optimization problem in natural lamguage with Python code interfacing to the optimization solvers such as [Gurobi](https://pypi.org/project/gurobipy/) and [CPLEX](https://developers.google.com/optimization/install?_gl=1*avvr0*_up*MQ..*_ga*MTQ4NzU5MzgwNC4xNzYwMzUzNjE3*_ga_SM8HXJ53K2*czE3NjAzNTM2MTYkbzEkZzAkdDE3NjAzNTM2MTYkajYwJGwwJGgw).

The results show that our method can solve the linear programming problems with a comparable accuracy to many multi-agent frameworks such as [Chain-of-Experts](https://openreview.net/pdf?id=HobyL1B9CZ) and [OptiMUS](https://arxiv.org/abs/2310.06116). Interestingly, this CAFA can enable opensource small language models like Llama3.1-8B-Q8 and DeepSeek-Coder-16B-Q4 achieve a significant performance boost in solving linear programming problems compared to that without CAFA.

The following tables shows the accuracy of different methods with various LLMs on the NL4Opt dataset. (The results of CoE and the OptiMUS are from the original papers)

| Model |standard | Chain-of-Experts | OptiMUS | CAFA |
| --- | --- | --- | --- | --- |
| GPT-4 | 47.3% | 64.2% | 78.7% | 70.1% |
| GPT-3.5-Turbo | 42.4% | 58.9% | __ | 59.0% |
| Llama3.1-8B-Q8 | 5.2% | __ | __ | 34.1% |
| DeepSeek-Coder-16B-Q4 | 5.5% | __ | __ | 60.1%|

Thus, CAFA is a prompting stradgy for improving LLMs in different sizes to solve LP problems effectively.

## Run the code

It suggestes to use Python>=3.10 to run the code in a virtual python environment.

### Reproduce the experiment results in the CAFA paper

`
conda create --name cafa python=3.10
`

`
pip install -r requirements.txt
`

Then, this paper used the [LMStudio](https://lmstudio.ai/) as the LM inference engine, and use the [Llama-index](https://www.llamaindex.ai/) as the interconnect components. After downloading the model weights, loaded into the disk, and start running the server, it is ready to run the code.

The main experiment are run with the Jupyter notebooks in the `notebooks` folder, the files starts with the prefix `e2e_code` are the notebooks for the paper of the CAFA method, Others are the baselines for comparison. 

### Run CAFA on the LLM4OR benchmarks

First download the LLM4OR benchmarks from by running the following command and save the benchmarks in the `benchmarks` folder:

`
python download_benckmarks.py
`

Then, run the LMStudio with a specific LLMs, and run the following command to run the CAFA method on the LLM4OR benchmarks:

`
python cafa_1.0/cafa.py --dataset-root benchmarks 
`

Or run a specific benchmark with the following command:

`
python cafa_1.0/cafa.py --dataset-root benchmarks --datasets NL4Opt
`

The datasets names can be selected from the benchmarks folder.

## Citation

If you found the idea useful in your work, please consider citing our paper:
```
@inproceedings{
deng2024cafa,
title={{CAFA}: Coding as Auto-Formulation Can Boost Large Language Models in Solving Linear Programming Problem},
author={haoxuan deng and Bohao Zheng and Yirui Jiang and Trung Hieu Tran},
booktitle={The 4th Workshop on Mathematical Reasoning and AI at NeurIPS'24},
year={2024},
url={https://openreview.net/forum?id=xC2xtBLmri}
}
```
