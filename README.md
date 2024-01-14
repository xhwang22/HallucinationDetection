# Hallucination Detection for Generative Large Language Models by Bayesian Sequential Estimation

## Intro

This is an implementation of paper: 

<a href="https://aclanthology.org/2023.emnlp-main.949.pdf">Hallucination Detection for Generative Large Language Models by Bayesian Sequential Estimation</a>

## Citation
If you use this code in your own work, please cite our paper:
```
@inproceedings{wang2023hallucination,
  title={Hallucination Detection for Generative Large Language Models by Bayesian Sequential Estimation},
  author={Wang, Xiaohua and Yan, Yuliang and Huang, Longtao and Zheng, Xiaoqing and Huang, Xuan-Jing},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={15361--15371},
  year={2023}
}
```

## Installation
Install HallucinationDetection:
```
git clone https://github.com/xhwang22/HallucinationDetection.git
cd HallucinationDetection
```
Install other dependencies:
```
# Install conda and pip dependencies
conda env create -f conda_env.yml
conda activate HD
```

## Reproduce
Reproduce the experiment by running the bash file:
```
bash run.sh
```
