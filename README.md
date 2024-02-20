# HardFocusTraining

Codes for [Preprint] "Take the Bull by the Horns: Hard Sample-Reweighted Continual Training Improves LLM Generalization" by Xuxi Chen, Zhendong Wang, Daouda Sow, Junjie Yang, Tianlong Chen, Yingbin Liang, Mingyuan Zhou, Zhangyang Wang

## Requirements

We provide the environment requirement in `requirements.txt`. Please run the following commands to install necessary dependencies:

```bash
pip install -r requirements.txt
git submodule update --init
cd lm-evaluation-harness
pip install -e .
cd ../
```

## Experiments

### Training

```bash
accelerate launch batch_training.py --data_path </path/to/dataset> --model_path </path/to/models> --batch_size 8 --frac 1 --save_dir opt_125m_baseline_sgd_bs8_n100_dro_2e-6 --num_batch 100 --model facebook/opt-125m --save_freq 100 --mode dro --lr 2e-6

accelerate launch batch_training.py --data_path </path/to/dataset> --model_path </path/to/models> --batch_size 8 --frac 1 --save_dir opt_350m_baseline_sgd_bs8_n100_dro_2e-6 --num_batch 100 --model facebook/opt-350m --save_freq 100 --mode dro --lr 2e-6

accelerate launch batch_training.py --data_path </path/to/dataset> --model_path </path/to/dataset> --batch_size 4 --frac 1 --save_dir sheared_1.3b_baseline_sgd_bs4_n100_dro_2e-6 --mode dro --num_batch 100 --model princeton-nlp/Sheared-LLaMA-1.3B --save_freq 10 --lr 2e-6 
```

### Evaluation


```bash
CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
    --model_args pretrained=facebook/opt-125m \
    --tasks piqa,hellaswag,winogrande,arc_challenge,boolq,mmlu \
    --num_fewshot 0 \
    --batch_size 32 \
    --checkpoint_path <path>/model_99.pt

CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
    --model_args pretrained=facebook/opt-125m \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 32 \
    --checkpoint_path <path>/model_99.pt
```

## Todos

- [ ] Upload codes related to instruction-tuning experiment. 
