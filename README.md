# Experiments for the paper "An Active Learning Reliability Method for Systems with Partially Defined Performance Functions"

[![pre-commit](https://github.com/fiveai/hGP_experiments/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/fiveai/hGP_experiments/actions/workflows/pre-commit.yml)
[![Run tests](https://github.com/fiveai/hGP_experiments/actions/workflows/test.yml/badge.svg)](https://github.com/fiveai/hGP_experiments/actions/workflows/test.yml)

[![arXiv](https://img.shields.io/badge/arXiv-2210.02168-b31b1b.svg)](https://arxiv.org/abs/2210.02168)

This repository presents the AK-MCS algorithm with a Hierarchical Gaussian Processes on some benchmark problems
alongside baseline methods.

## Getting started

Simply install the requirements and run the script to reproduce our published results:

`pip install requirements.txt`

`python run_all.py`

Results will be output in directory `tmp` with 5 repeats by default.
The experiments will by default run in parallel with `n_jobs=15` for the repeats in order to save compute time.
If your computer has fewer gpus then you can set the command line options:
`run_all.py [-h] [--n-repeats N_REPEATS] [--save-dir SAVE_DIR] [--n-jobs N_JOBS]`.

We used `python3.9` but the script should also work in other `python3.8`.

If you use the package in your research please consider citing our paper:
```
Sadeghi, J., Mueller, R., & Redford, J. (2022). An Active Learning Reliability Method for Systems with Partially Defined Performance Functions. doi:10.48550/ARXIV.2210.02168
```
