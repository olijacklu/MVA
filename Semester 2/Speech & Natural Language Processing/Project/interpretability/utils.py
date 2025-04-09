import os
import json
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random
import editdistance
import gc
from peft import get_peft_model, LoraConfig
import time

base_dir = '/content/drive/MyDrive/MVA/NLP/AlgorithmsSpeechNLP'
monolingual_train_pairs = {
    'eng1': 'eng_mls',
    # 'eng2': 'eng_nchlt',
    # 'eng3': 'eng_voxpopuli',
    'fra1': 'fra_mls',
    # 'fra2': 'fra_voxpopuli',
    'deu1': 'deu_swc',
    # 'deu2': 'deu_voxpopuli',
    'rus': 'rus_M-AILABS',
    'swa': 'swa_ALFFA',
    'swe': 'swe_NST',
    'xty': 'xty_mexico-el'
}

def preprocess_ml_superb_data(data_dir=f"/content/drive/MyDrive/MVA/NLP/AlgorithmsSpeechNLP/seventh_version"):
    datasets = {}

    sources = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for source in tqdm(sources):
        source_dir = os.path.join(data_dir, source)

        languages = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

        for lang in languages:
            lang_dir = os.path.join(source_dir, lang)

            key = f"{lang}_{source}"

            transcript_files = {
                '10min_train': os.path.join(lang_dir, 'transcript_10min_train.txt'),
                '10min_dev': os.path.join(lang_dir, 'transcript_10min_dev.txt'),
                '10min_test': os.path.join(lang_dir, 'transcript_10min_test.txt'),
                '1h_train': os.path.join(lang_dir, 'transcript_1h_train.txt'),
            }

            path = os.path.join('seventh_version', source, lang)

            datasets[key] = {
                'source': source,
                'language': lang,
                'path': path,
                'splits': {}
            }

            for split_name, file_path in transcript_files.items():
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    transcripts = []
                    for line in lines:
                        parts = line.strip().split(maxsplit=2)
                        if len(parts) >= 2:
                            if len(parts) == 3:
                                wav_id, _, text = parts
                            else:
                                wav_id, text = parts

                            wav_path = os.path.join(path, 'wav', f"{wav_id}.wav")
                            if os.path.exists(os.path.join("/content/drive/MyDrive/MVA/NLP/AlgorithmsSpeechNLP", wav_path)):
                                transcripts.append({
                                    'wav_id': wav_id,
                                    'text': text,
                                    'wav_path': wav_path
                                })

                    datasets[key]['splits'][split_name] = transcripts

    return datasets



def prepare_multilingual_data(datasets, split="10min_train", num_samples=None):
    all_data = {}
    monolingual_train_pairs = {
            'eng1': 'eng_mls',
            # 'eng2': 'eng_nchlt',
            # 'eng3': 'eng_voxpopuli',
            'fra1': 'fra_mls',
            # 'fra2': 'fra_voxpopuli',
            'deu1': 'deu_swc',
            # 'deu2': 'deu_voxpopuli',
            'rus': 'rus_M-AILABS',
            'swa': 'swa_ALFFA',
            'swe': 'swe_NST',
            'xty': 'xty_mexico-el'
        }

    for lang_pair, data in datasets.items():
        if lang_pair in monolingual_train_pairs.values():
            all_data[lang_pair] = []
            splits = data.get('splits', {})
            if split in splits and splits[split]:
                for entry in splits[split]:
                    if data['language'] not in ['jpn', 'cmn']:
                        entry_with_lang = entry.copy()
                        entry_with_lang['language'] = data['language']
                        if num_samples is not None:
                            if len(all_data[lang_pair]) < num_samples:
                                all_data[lang_pair].append(entry_with_lang)
                            else:
                                break
                        else:
                            all_data[lang_pair].append(entry_with_lang)

    return all_data

def load_model(model_name, device="cuda"):
    if "hubert" in model_name.lower():
        model = HubertModel.from_pretrained(model_name)
    else:  # For both wav2vec2 and xlsr models
        model = Wav2Vec2Model.from_pretrained(model_name)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model.to(device)
    return model, feature_extractor

class ASRDataset(Dataset):
    def __init__(self, data_entries, feature_extractor, multilingual=False):
        self.data_entries = data_entries
        self.feature_extractor = feature_extractor
        self.multilingual = multilingual

        if multilingual:
            languages = sorted(list(set([entry.get('language', 'unknown') for entry in data_entries])))
            self.lang_to_idx = {lang: i for i, lang in enumerate(languages)}
            self.idx_to_lang = {i: lang for i, lang in enumerate(languages)}
            self.num_languages = len(languages)

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        waveform, sample_rate = torchaudio.load(os.path.join("/content/drive/MyDrive/MVA/NLP/AlgorithmsSpeechNLP", entry['wav_path']))

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze()

        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.squeeze()

        item = {
            'input_values': input_values,
            'text': entry['text'],
            'wav_id': entry['wav_id']
        }

        if self.multilingual:
            item['language'] = entry.get('language', 'unknown')
            item['lang_id'] = self.lang_to_idx.get(item['language'], 0)

        return item

class MLSUPERBDownstreamModel(nn.Module):
    def __init__(self, upstream_model, num_chars, num_languages=None, task="asr", lora_config=None):
        super().__init__()
        self.upstream_model = upstream_model
        self.task = task

        for param in self.upstream_model.parameters():
            param.requires_grad = False

        self.layer_weights = nn.Parameter(torch.ones(upstream_model.config.num_hidden_layers) / upstream_model.config.num_hidden_layers)

        self.conv_downsample = nn.Conv1d(
            upstream_model.config.hidden_size,
            upstream_model.config.hidden_size,
            kernel_size=2,
            stride=2
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=upstream_model.config.hidden_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        if task in ["asr", "asr+lid"]:
            self.asr_classifier = nn.Linear(upstream_model.config.hidden_size, num_chars)

        if task in ["lid", "asr+lid"] and num_languages is not None:
            self.lid_classifier = nn.Linear(upstream_model.config.hidden_size, num_languages)

        if lora_config:
            for i in range(len(self.transformer.layers)):
                print(self.transformer.layers[i])
                self.transformer.layers[i] = get_peft_model(self.transformer.layers[i], lora_config)

    def forward(self, input_values, input_lengths=None):
        with torch.no_grad():
            outputs = self.upstream_model(
                input_values,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states[1:]
        weighted_sum = torch.zeros_like(hidden_states[0])

        for i, h in enumerate(hidden_states):
            weighted_sum += self.layer_weights[i] * h

        x = weighted_sum.transpose(1, 2)
        x = self.conv_downsample(x)
        x = x.transpose(1, 2)

        x = self.transformer(x)

        result = {}

        if self.task in ["asr", "asr+lid"]:
            result["asr_logits"] = self.asr_classifier(x)

        if self.task in ["lid", "asr+lid"]:
            pooled = torch.mean(x, dim=1)
            result["lid_logits"] = self.lid_classifier(pooled)

        if self.task == "asr":
            return result["asr_logits"]

        return result

def collate_fn(batch, multilingual=False):
    batch.sort(key=lambda x: x['input_values'].size(0), reverse=True)

    input_values = [item['input_values'] for item in batch]
    texts = [item['text'] for item in batch]

    input_lengths = torch.tensor([x.size(0) for x in input_values])

    max_length = input_lengths[0].item()
    padded_inputs = torch.zeros(len(batch), max_length)

    for i, (input_value, length) in enumerate(zip(input_values, input_lengths)):
        padded_inputs[i, :length] = input_value

    result = {
        'input_values': padded_inputs,
        'input_lengths': input_lengths,
        'text': texts
    }

    if multilingual:
        languages = [item['language'] for item in batch]
        lang_ids = torch.tensor([item['lang_id'] for item in batch])
        result['languages'] = languages
        result['lang_ids'] = lang_ids

    return result

def data_loaders_and_vocab_multilingual(datasets, feature_extractor, train_split="10min_train", model_type="multilingual", task= "lid", lang=None, num_samples=10):
    NUM_WORKERS = os.cpu_count()
    # TRAIN AND VALIDATION
    random.seed(42)

    train_data = prepare_multilingual_data(datasets, train_split)
    train_data = [sample for language in train_data.keys() for sample in train_data[language]]
    val_data = prepare_multilingual_data(datasets, "10min_dev")
    val_data = [sample for language in val_data.keys() for sample in val_data[language]]

    train_dataset = ASRDataset(train_data, feature_extractor, multilingual=True)
    val_dataset = ASRDataset(val_data, feature_extractor, multilingual=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn(batch, multilingual=True))

    # Determine the number of validation samples
    num_val_samples = min(100, len(val_dataset))
    val_sample_indices = list(range(len(val_dataset)))

    if len(val_dataset) > num_val_samples:
        val_sample_indices = random.sample(val_sample_indices, num_val_samples)

    val_subset = torch.utils.data.Subset(val_dataset, val_sample_indices)
    val_loader = DataLoader(val_subset, batch_size=8, collate_fn=lambda batch: collate_fn(batch, multilingual=True))

    chars = set()
    for entry in train_data + val_data:
        chars.update(entry['text'])
    vocab = sorted(list(chars))

    idx_to_char = {0: '<blank>'}
    char_to_idx = {'<blank>': 0}

    for i, char in enumerate(vocab, start=1):
        idx_to_char[i] = char
        char_to_idx[char] = i

    char_mappings = {'idx_to_char': idx_to_char, 'char_to_idx': char_to_idx}
    
    # TEST
    idx_to_char = char_mappings['idx_to_char']
    char_to_idx = char_mappings['char_to_idx']

    if lang:
        test_samples = prepare_multilingual_data(datasets, '10min_test', num_samples=num_samples)[lang]
        #test_samples = [sample for sample in prepare_multilingual_data(datasets, '10min_test')
        #                if sample['language'] == lang_id][:num_samples]
    else:
        test_samples = prepare_multilingual_data(datasets, '10min_test', num_samples=num_samples)
        test_samples = [sample for language in test_samples.keys() for sample in test_samples[language]]

    test_dataset = ASRDataset(test_samples, feature_extractor, multilingual=(task != "asr"))
    test_loader = DataLoader(test_dataset, batch_size=8,
                            collate_fn=lambda batch: collate_fn(batch, multilingual=(task != "asr")),
                            num_workers=NUM_WORKERS, pin_memory=True)
    language_mapping = {'idx_to_lang': test_dataset.idx_to_lang, 'lang_to_idx': test_dataset.lang_to_idx}
    return train_loader, val_loader, test_loader, char_mappings, language_mapping

# For monolingual model
def load_monolingual_model(lang, upstream_model, vocab_size, model_path, device="cuda"):
    model = MLSUPERBDownstreamModel(upstream_model, num_chars=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model
    
# For multilingual model
def load_multilingual_model(task, upstream_model, vocab_size, num_languages, model_path, device="cuda"):
    model = MLSUPERBDownstreamModel(
        upstream_model,
        num_chars=vocab_size,
        num_languages=num_languages,
        task=task
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model