import os
import torch
import torchaudio
from torch.utils.data import Dataset

class MLSuperbDataset(Dataset):
    def __init__(self, base_dir, data_entries, feature_extractor, multilingual=False):
        self.base_dir = base_dir
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
        waveform, sample_rate = torchaudio.load(os.path.join(self.base_dir, entry['wav_path']))

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