import os
from datasets import load_dataset
from accelerate import Accelerator, DeepSpeedPlugin
import torch

from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import argparse
import logging

from accelerate.logging import get_logger

logger = get_logger(__name__)
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def g_func(losses, l):
    return torch.exp(losses/l)

def h_func(losses, delta=1., l_min=0.75, l_max=1.8):
    return 2. * delta * losses / max(l_max - l_min, 1e-6) - delta * (l_max + l_min) / max(l_max - l_min, 1e-6)

def f_func(losses, delta=1.):
    return 1 - losses**2 / delta**2


def main(args):
    set_seed(args.seed)
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=1)
    accelerator = Accelerator(mixed_precision='bf16', deepspeed_plugin=deepspeed_plugin,
        log_with="all")
    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    raw_dataset = load_dataset("c4", "en", cache_dir=args.data_path, split="train", streaming=True)
    raw_dataset = raw_dataset.shuffle(buffer_size=10_000, seed=42)
    for item in raw_dataset:
        print(item)
        break
    hf_token = args.hf_token
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m",
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              cache_dir=args.model_path,
                                              token=hf_token,
                                              padding_side="left")
    tokenizer.pad_token_id = 0

    def tokenize_function(examples):
        data = tokenizer(examples["text"], padding=True, truncation=True, max_length=512)
        data = {k: torch.tensor(v) for k, v in data.items()}
        return data
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["url", "timestamp", "text"])
    # tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    bs = args.batch_size
    lr = args.lr
    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", 
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                 cache_dir=args.model_path, 
                                                 token=hf_token)
    vocab_size = model.config.vocab_size
    train_dataloader = torch.utils.data.DataLoader(tokenized_dataset, shuffle=False, batch_size=bs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer
    )
    # save_dir = f'/v-zhendwang/datasets/C4/filterd_llama2_7b_num{capacity}'
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    num_batch = args.num_batch
    frac = args.frac
    buffers = []
    all_weights = []
    for i, batch in enumerate(tqdm(train_dataloader, mininterval=60)):
        # text = batch['text'][0]
        # del batch['text']
        if i > num_batch:
            break
        labels = batch['input_ids']
        batch['labels'] = labels
        outputs = model(**batch)
        if accelerator.is_main_process:
            logger.info(f"Step: {i}, Official Loss: {outputs['loss']}")
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size), shift_labels.view(-1), reduction='none')
        loss = loss.reshape(bs, -1)
        mask = shift_labels.reshape(bs, -1) != -100
        num_non_zeros = mask.sum(1)
        loss = (loss * mask).sum(1) / num_non_zeros
        gathered_loss = loss

        if args.mode == 'rank':
            capacity = int(frac * gathered_loss.shape[0])
            sorted_loss, _ = torch.sort(gathered_loss, descending=True)
            if args.patience > 1:
                buffers.append(gathered_loss)
                if len(buffers) == args.patience:
                    collected_loss = torch.cat(buffers)
                    if accelerator.is_main_process:
                        logger.info(f"Step: {i}, Loss: {collected_loss.mean()}")
                    sorted_loss, _ = torch.sort(collected_loss, descending=True)
                    gathered_loss = sorted_loss[args.portion*capacity:(args.portion + 1)*capacity].mean()
                    optimizer.zero_grad()
                    accelerator.backward(gathered_loss)
                    optimizer.step()
                    buffers = []
            else:
                gathered_loss = sorted_loss[args.portion*capacity:(args.portion + 1)*capacity]
                gathered_loss = gathered_loss.mean()
                # print(gathered_loss)
                if accelerator.is_main_process:
                    logger.info(f"Step: {i}, Loss: {gathered_loss}")
                optimizer.zero_grad()
                accelerator.backward(gathered_loss)
                optimizer.step()
        elif args.mode == 'dro':
            kl_reg = args.kl_reg
            if args.patience > 1:
                buffers.append(gathered_loss)
                if len(buffers) == args.patience:
                    collected_loss = torch.cat(buffers)
                    if accelerator.is_main_process:
                        logger.info(f"Step: {i}, Loss: {collected_loss.mean()}")
                    g_losses = g_func(collected_loss.detach() - collected_loss.max().detach(), l=kl_reg)
                    weights = g_losses / g_losses.sum()
                    all_weights.append(weights)
                    collected_loss = torch.sum(weights.detach() * collected_loss)
                    # pdb.set_trace()
                    optimizer.zero_grad()
                    accelerator.backward(collected_loss)
                    optimizer.step()
                    buffers = []
            else:
                g_losses = g_func(gathered_loss.detach() - gathered_loss.max().detach(), l=kl_reg)
                weights = g_losses / g_losses.sum()
                all_weights.append(weights)
                gathered_loss = torch.sum(weights.detach() * gathered_loss)
                # print(gathered_loss)
                if accelerator.is_main_process:
                    logger.info(f"Step: {i}, Loss: {gathered_loss.mean()}")
                optimizer.zero_grad()
                accelerator.backward(gathered_loss)
                optimizer.step()
        if (i + 1) % args.save_freq == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(
                    unwrapped_model.state_dict(),
                    os.path.join(save_dir, f"model_{i}.pt"))

    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(
        unwrapped_model.state_dict(),
        os.path.join(save_dir, "model_final.pt"))

    # get local rank
    local_rank = accelerator.local_process_index
    torch.save(all_weights, os.path.join(save_dir, f"weights_{local_rank}.pt"))
    # torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default='naive')
    parser.add_argument("--num_batch", type=int, default=100)
    parser.add_argument("--frac", type=float, default=0.125)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl_reg", type=float, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--portion", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--hf_token", type=str, required=True)
    args = parser.parse_args()
    main(args)