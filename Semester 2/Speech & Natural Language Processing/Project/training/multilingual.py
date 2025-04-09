import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import bitsandbytes as bnb
import random
from tqdm import tqdm
import time

from data.dataset import MLSuperbDataset
from data.preprocess import prepare_multilingual_data
from models.downstream import DownstreamModel
from models.utils import collate_fn
from evaluation.metrics import calculate_cer


def train_and_evaluate_multilingual(upstream_model, feature_extractor, datasets, task="asr", train_split="10min_train", device="cuda", lora_config=None, quantize=False):
    torch.manual_seed(42)
    random.seed(42)

    train_data = prepare_multilingual_data(datasets, train_split)
    val_data = prepare_multilingual_data(datasets, "10min_dev")

    train_dataset = MLSuperbDataset(train_data, feature_extractor, multilingual=True)
    val_dataset = MLSuperbDataset(val_data, feature_extractor, multilingual=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn(batch, multilingual=True))

    num_val_samples = min(100, len(val_dataset))
    val_sample_indices = list(range(len(val_dataset)))
    if len(val_dataset) > num_val_samples:
        val_sample_indices = random.sample(val_sample_indices, num_val_samples)
    val_subset = torch.utils.data.Subset(val_dataset, val_sample_indices)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn(batch, multilingual=True))

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
        num_languages=train_dataset.num_languages,
        task=task,
        lora_config=lora_config
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {total_params:,} parameters")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-6) if quantize else optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)

    num_iterations = 30000
    iteration = 0

    running_loss = 0
    update_interval = 100

    pbar = tqdm(total=num_iterations, desc=f"Training {task}", unit="iter", dynamic_ncols=True)

    model.train()
    while iteration < num_iterations:
        for batch in train_loader:
            if iteration >= num_iterations:
                break  # Stop when we reach the desired iterations

            input_values = batch['input_values'].to(device, dtype=torch.float16)
            input_lengths = batch['input_lengths'].to(device)
            texts = batch['text']
            lang_ids = batch['lang_ids'].to(device)

            with torch.autocast('cuda', enabled=False):
                outputs = model(input_values)

            batch_loss = 0

            if task in ["asr", "asr+lid"]:
                asr_logits = outputs["asr_logits"] if task == "asr+lid" else outputs
                asr_logits = asr_logits.float()

                target_indices = [[char_to_idx[c] for c in text] for text in texts]
                target_lengths = torch.tensor([len(indices) for indices in target_indices]).to(device)

                targets = torch.cat([torch.tensor(indices) for indices in target_indices]).to(device)

                time_reduction_factor = input_values.size(1) // asr_logits.size(1)
                input_lengths_for_ctc = torch.div(input_lengths, time_reduction_factor, rounding_mode='floor')
                input_lengths_for_ctc = torch.clamp(input_lengths_for_ctc, min=1, max=asr_logits.size(1))

                asr_loss = ctc_loss(asr_logits.transpose(0, 1), targets, input_lengths_for_ctc, target_lengths)
                batch_loss += asr_loss

            if task in ["lid", "asr+lid"]:
                lid_logits = outputs["lid_logits"].float()
                lid_loss = ce_loss(lid_logits, lang_ids)
                batch_loss += lid_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
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
    all_langs = []

    correct_count = 0
    total_count = 0

    with torch.no_grad():
        inference_times = []
        throughputs = []
        memory_usages = []
        for batch in val_loader:
            torch.cuda.reset_peak_memory_stats(device)

            input_values = batch['input_values'].to(device, dtype=torch.float16)
            texts = batch['text']
            languages = batch['languages']
            lang_ids = batch['lang_ids'].to(device)
            batch_size = input_values.shape[0]

            start_time = time.time()
            outputs = model(input_values)
            inference_time = time.time() - start_time

            memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

            inference_times.append(inference_time)
            throughputs.append(batch_size / inference_time)
            memory_usages.append(memory_usage)

            if task in ["asr", "asr+lid"]:
                asr_logits = outputs["asr_logits"] if task == "asr+lid" else outputs
                predictions = torch.argmax(asr_logits, dim=-1)

                for pred, ref, lang in zip(predictions, texts, languages):
                    pred_indices = pred.cpu().numpy()

                    merged_indices = []
                    prev_idx = -1

                    for idx in pred_indices:
                        if idx != prev_idx and idx != 0:
                            merged_indices.append(idx)
                        prev_idx = idx

                    pred_text = ''.join([idx_to_char[idx] for idx in merged_indices if idx in idx_to_char])

                    all_predictions.append(pred_text)
                    all_references.append(ref)
                    all_langs.append(lang)

            if task in ["lid", "asr+lid"]:
                lid_logits = outputs["lid_logits"]
                lid_predictions = torch.argmax(lid_logits, dim=1)

                correct = (lid_predictions == lang_ids).sum().item()
                correct_count += correct
                total_count += len(lang_ids)

    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_throughput = sum(throughputs) / len(throughputs)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    results = {}

    if task in ["asr", "asr+lid"]:
        cers = [calculate_cer(ref, pred) for ref, pred in zip(all_references, all_predictions)]
        avg_cer = sum(cers) / len(cers) if cers else 0
        results["cer"] = avg_cer

        lang_cers = {}
        for pred, ref, lang in zip(all_predictions, all_references, all_langs):
            if lang not in lang_cers:
                lang_cers[lang] = []
            lang_cers[lang].append(calculate_cer(ref, pred))

        results["lang_cer"] = {lang: sum(cers)/len(cers) if cers else 0 for lang, cers in lang_cers.items()}

        print(f"\n===== ASR Results =====")
        print(f"Overall CER: {avg_cer:.4f}")
        print(f"Per-language CER:")
        for lang, cer in sorted(results["lang_cer"].items()):
            print(f"  {lang}: {cer:.4f}")

    if task in ["lid", "asr+lid"]:
        accuracy = correct_count / total_count if total_count > 0 else 0
        results["lid_accuracy"] = accuracy

        print(f"\n===== LID Results =====")
        print(f"Accuracy: {accuracy:.4f}")

    results["avg_inference_time"] = avg_inference_time
    results["avg_throughput"] = avg_throughput
    results["avg_memory_usage"] = avg_memory_usage

    print(f"Average inference time: {avg_inference_time:.4f} seconds per batch")
    print(f"Average throughput: {avg_throughput:.2f} samples/second")
    print(f"Average peak memory usage: {avg_memory_usage:.2f} MB")

    return model, results, {'idx_to_char': idx_to_char, 'char_to_idx': char_to_idx}
