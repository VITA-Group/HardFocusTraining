import transformers
from datasets import load_dataset
import torch
import heapq
raw_datasets = load_dataset("c4", "en", cache_dir='/mnt/Data/xuxi/datasets', split="train", streaming=True)
for item in raw_datasets:
    print(item)
    break

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/xc4863/GAP/checkpoints/OPT-125M-HARD-DESCENT-CC_1', truncation_side='left', use_fast=False)

def tokenize_function(examples):
    data = tokenizer(examples["text"], padding=True, truncation=True, max_length=2048)
    return data
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["url", "timestamp"])
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/home/xc4863/GAP/checkpoints/OPT-125M-HARD-DESCENT-CC_1").cuda()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=1)

from tqdm import tqdm
losses = []
h = []
capacity = 1000
import pandas as pd
losses = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(train_dataloader)):
        text = batch['text'][0]
        del batch['text']
        batch = {k: torch.tensor(v).cuda().unsqueeze(0) for k, v in batch.items()}
        batch['labels'] = batch['input_ids']
        outputs = model(input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids'])
        loss = outputs.loss
        losses.append(loss.item())

        if i > 10000:
            break
import matplotlib.pyplot as plt
import pickle
with open('post_training_loss_1.txt', "w") as f:
    f.write(','.join(list(map(str, losses))))
plt.figure(figsize=(6, 6))
plt.hist(losses)
plt.savefig("post_training_1.png")