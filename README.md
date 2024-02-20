# HardFocusTraining

Codes for [Preprint] "Take the Bull by the Horns: Hard Sample-Reweighted Continual Training Improves LLM Generalization" by Xuxi Chen, Zhendong Wang, Daouda Sow, Junjie Yang, Tianlong Chen, Yingbin Liang, Mingyuan Zhou, Zhangyang Wang

## Overview

## Requirements

We provide the environment requirement in `requirements.txt`. Please run the following:
```bash
pip install -r requirements.txt
git submodule update --init
cd lm-evaluation-harness
pip install -e .
cd ../
```

## Experiments

### Continual Pre-training (OPT-125m)
#### Training

```bash
accelerate launch batch_training.py --data_path </path/to/dataset> --model_path </path/to/models> --batch_size 8 --frac 1 --save_dir opt_125m_baseline_sgd_bs8_n100_dro_2e-6 --num_batch 100 --model facebook/opt-125m --save_freq 100 --mode dro
```

#### Evaluation

```bash
CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
    --model_args pretrained=facebook/opt-125m \
    --tasks piqa,hellaswag,winogrande,arc_challenge,boolq \
    --num_fewshot 0 \
    --batch_size 32 \
    --checkpoint_path opt_125m_baseline_sgd_bs8_n100_dro_2e-6/model_99.pt > instance_350m_2e-6.txt
```
