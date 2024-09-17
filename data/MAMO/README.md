---
license: cc-by-nc-4.0
language:
- en
pretty_name: MAMO
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: easy_lp
    path: "MAMO_EasyLP.json"
  - split: complex_lp
    path: "MAMO_ComplexLP.json"
---
## Overview
This dataset is a direct copy of the [MAMO Optimization Data](https://github.com/FreedomIntelligence/Mamo), with its EasyLP and ComplexLP components duplicated but with adapted field names.

## Citation

```latex
@misc{huang2024mamo,
      title={Mamo: a Mathematical Modeling Benchmark with Solvers}, 
      author={Xuhan Huang and Qingning Shen and Yan Hu and Anningzhe Gao and Benyou Wang},
      year={2024},
      eprint={2405.13144},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```