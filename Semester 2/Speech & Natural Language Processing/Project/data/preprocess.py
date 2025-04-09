import os
from tqdm import tqdm

from config.config import TRAIN_PAIRS


def preprocess_data(base_dir):
    datasets = {}

    data_dir = f"{base_dir}/seventh_version"

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
                            if os.path.exists(os.path.join(base_dir, wav_path)):
                                transcripts.append({
                                    'wav_id': wav_id,
                                    'text': text,
                                    'wav_path': wav_path
                                })

                    datasets[key]['splits'][split_name] = transcripts

    return datasets

def prepare_multilingual_data(datasets, split="10min_train"):
    all_data = []

    for lang_pair, data in datasets.items():
        if lang_pair in TRAIN_PAIRS.values():
            splits = data.get('splits', {})
            if split in splits and splits[split]:
                for entry in splits[split]:
                    if data['language'] not in ['jpn', 'cmn']:
                        entry_with_lang = entry.copy()
                        entry_with_lang['language'] = data['language']
                        all_data.append(entry_with_lang)

    return all_data
