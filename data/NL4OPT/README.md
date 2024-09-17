---
license: cc-by-nc-4.0
language:
- en
pretty_name: NL4OPT
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: "NL4OPT_with_optimal_solution.json"
---
## Overview
This dataset is a conversion of the NL4OPT test set. 
The official NL4OPT provides only mathematical models as targets, complicating the verification of execution accuracy due to the absence of optimal solutions for the optimization modeling task. 
To address this issue, we have converted these mathematical models into programs using GPT-4, calculated and checked the optimal solutions, and used these as ground truth. 
Note that a small percentage of examples (15%) were discarded due to failed conversions.

## Citation

```latex
@article{tang2024orlm,
  title={ORLM: Training Large Language Models for Optimization Modeling},
  author={Tang, Zhengyang and Huang, Chenyu and Zheng, Xin and Hu, Shixi and Wang, Zizhuo and Ge, Dongdong and Wang, Benyou},
  journal={arXiv preprint arXiv:2405.17743},
  year={2024}
}
```

```latex
@inproceedings{nl4opt,
  title={NL4Opt competition: Formulating optimization problems based on their natural language descriptions},
  author={Ramamonjison, Rindranirina and Yu, Timothy and Li, Raymond and Li, Haley and Carenini, Giuseppe and Ghaddar, Bissan and He, Shiqi and Mostajabdaveh, Mahdi and Banitalebi-Dehkordi, Amin and Zhou, Zirui and others},
  booktitle={NeurIPS 2022 Competition Track},
  pages={189--203},
  year={2023},
  organization={PMLR}
}
```