import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import bitsandbytes as bnb
import random
from tqdm import tqdm
import time

from data.dataset import MLSuperbDataset
from models.downstream import DownstreamModel
from models.utils import collate_fn
from evaluation.metrics import calculate_cer


def train_and_evaluate_monolingual(lang, data_pair, upstream_model, feature_extractor, datasets, train_split="10min_train", device="cuda", lora_config=None, quantize=False):
    torch.manual_seed(42)
    random.seed(42)

    train_data = datasets[data_pair]['splits'][train_split]
    val_data = datasets[data_pair]['splits']['10min_dev']

    train_dataset = MLSuperbDataset(train_data, feature_extractor)
    val_dataset = MLSuperbDataset(val_data, feature_extractor)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn(batch))

    num_val_samples = min(100, len(val_dataset))
    val_sample_indices = list(range(len(val_dataset)))
    if len(val_dataset) > num_val_samples:
        val_sample_indices = random.sample(val_sample_indices, num_val_samples)
    val_subset = torch.utils.data.Subset(val_dataset, val_sample_indices)
    val_loader = DataLoader(val_subset, batch_size=8, collate_fn=lambda batch: collate_fn(batch))

    chars = set()
    for entry in train_data + val_data:
        chars.update(entry['text'])
    vocab = sorted(list(chars))

    idx_to_char = {0: '<blank>'}
    char_to_idx = {'<blank>': 0}

    for i, char in enumerate(vocab, start=1):
        idx_to_char[i] = char
        char_to_idx[char] = i

    model = DownstreamModel(
        upstream_model,
        num_chars=len(idx_to_char),
        task="asr",
        lora_config=lora_config
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {total_params:,} parameters")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-6) if quantize else optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)

    num_iterations = 15000
    iteration = 0
    running_loss = 0
    update_interval = 100

    pbar = tqdm(total=num_iterations, desc=f"Training {lang}", unit="iter", dynamic_ncols=True)

    model.train()
    while iteration < num_iterations:
        for batch in train_loader:
            if iteration >= num_iterations:
                break

            # Keep inputs in float16 for efficiency
            input_values = batch['input_values'].to(device, dtype=torch.float16)
            input_lengths = batch['input_lengths'].to(device)
            texts = batch['text']

            target_indices = [[char_to_idx[c] for c in text] for text in texts]
            target_lengths = torch.tensor([len(indices) for indices in target_indices]).to(device)
            targets = torch.cat([torch.tensor(indices) for indices in target_indices]).to(device)

            with torch.autocast('cuda', enabled=False):
                logits = model(input_values)

                logits_float = logits.float()

                time_reduction_factor = input_values.size(1) // logits_float.size(1)
                input_lengths_for_ctc = torch.div(input_lengths, time_reduction_factor, rounding_mode='floor')
                input_lengths_for_ctc = torch.clamp(input_lengths_for_ctc, min=1, max=logits_float.size(1))

                loss = criterion(logits_float.transpose(0, 1), targets, input_lengths_for_ctc, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iteration += 1

            if iteration % update_interval == 0:
                avg_loss = running_loss / update_interval
                pbar.update(update_interval)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                running_loss = 0

    pbar.close()

    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        inference_times = []
        throughputs = []
        memory_usages = []
        for batch in val_loader:
            torch.cuda.reset_peak_memory_stats(device)

            input_values = batch['input_values'].to(device, dtype=torch.float16)
            texts = batch['text']
            batch_size = input_values.shape[0]

            start_time = time.time()
            logits = model(input_values)
            inference_time = time.time() - start_time

            memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

            inference_times.append(inference_time)
            throughputs.append(batch_size / inference_time)
            memory_usages.append(memory_usage)

            predictions = torch.argmax(logits, dim=-1)

            for pred, text in zip(predictions, texts):
                pred_indices = pred.cpu().numpy()

                merged_indices = []
                prev_idx = -1

                for idx in pred_indices:
                    if idx != prev_idx and idx != 0:
                        merged_indices.append(idx)
                    prev_idx = idx

                pred_text = ''.join([idx_to_char[idx] for idx in merged_indices if idx in idx_to_char])
                all_predictions.append(pred_text)
                all_references.append(text)

    cers = [calculate_cer(ref, pred) for ref, pred in zip(all_references, all_predictions)]
    avg_cer = sum(cers) / len(cers)
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_throughput = sum(throughputs) / len(throughputs)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    results = {}
    results["cer"] = avg_cer
    results["avg_inference_time"] = avg_inference_time
    results["avg_throughput"] = avg_throughput
    results["avg_memory_usage"] = avg_memory_usage

    print(f"\n===== Evaluation Results =====")
    print(f"Language: {lang} | Dataset: {data_pair}")
    print(f"Total Evaluated Samples: {len(all_references)}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average inference time: {avg_inference_time:.4f} seconds per batch")
    print(f"Average throughput: {avg_throughput:.2f} samples/second")
    print(f"Average peak memory usage: {avg_memory_usage:.2f} MB")
    print("===============================\n")

    return model, results, {'idx_to_char': idx_to_char, 'char_to_idx': char_to_idx}
