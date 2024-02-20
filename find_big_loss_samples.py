import transformers
from datasets import load_dataset
import torch
import heapq
raw_datasets = load_dataset("c4", "en", cache_dir='/mnt/Data/xuxi/datasets', split="train", streaming=True)
for item in raw_datasets:
    print(item)
    break

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

def tokenize_function(examples):
    data = tokenizer(examples["text"], padding=True, truncation=True, max_length=2048)
    return data
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["url", "timestamp"])
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").cuda()
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets, shuffle=False, batch_size=1)

from tqdm import tqdm
losses = []
h = []
capacity = 1000
import pandas as pd
with torch.no_grad():
    for i, batch in enumerate(tqdm(train_dataloader)):
        text = batch['text'][0]
        del batch['text']
        batch = {k: torch.tensor(v).cuda().unsqueeze(0) for k, v in batch.items()}
        batch['labels'] = batch['input_ids']
        outputs = model(**batch)
        loss = outputs.loss
        if len(h) < capacity:
            heapq.heappush(h, (-loss, f"\"{text}\""))
        else:
            # Equivalent to a push, then a pop, but faster
            spilled_value = heapq.heappushpop(h, (-loss, f"\"{text}\""))
    
        if i % capacity == 0 and i > 0:
            df = pd.DataFrame({"doc_id": range(capacity), "text": [hh[1] for hh in h], "corpus": ['CC'] * capacity})
            df.to_csv("cc.csv", index=False)