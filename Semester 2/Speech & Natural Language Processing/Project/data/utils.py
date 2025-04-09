from .dataset import MLSuperbDataset, ASRDataset
from .preprocess import prepare_multilingual_data
from ../models.utils import collate_fn
from torch.utils.data import DataLoader
import torch
import random
import os

def data_loaders_and_vocab(datasets, feature_extractor, train_split="10min_train", model_type="multilingual", data_pair=None, task= "lid", lang_id=None, num_samples=10):
    NUM_WORKERS = os.cpu_count()
    # TRAIN AND VALIDATION
    random.seed(42)

    train_data = None
    val_data = None
    test_data = None

    if model_type == "multilingual":
        train_data = prepare_multilingual_data(datasets, train_split)
        val_data = prepare_multilingual_data(datasets, "10min_dev")
    elif model_type == "monolingual":
        train_data = datasets[data_pair]['splits'][train_split]
        val_data = datasets[data_pair]['splits']['10min_dev']

    train_dataset = ASRDataset(train_data, feature_extractor, multilingual=(model_type == "multilingual"))
    val_dataset = ASRDataset(val_data, feature_extractor, multilingual=(model_type == "multilingual"))

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

    if model_type == "monolingual":
        test_samples = datasets[data_pair]['splits']['10min_test'][:num_samples]
        test_dataset = MLSuperbDataset(test_samples, feature_extractor)
        test_loader = DataLoader(test_dataset, batch_size=len(test_samples), collate_fn=collate_fn, num_workers=4, pin_memory=True)
     elif model_type == "multilingual":
        test_samples = prepare_multilingual_data(datasets, '10min_test')
        if lang_id:
            test_samples = test_samples[lang_id]
        test_samples = [sample for lang in test_samples for sample in lang[:num_samples]]

    test_dataset = ASRDataset(test_samples, feature_extractor, multilingual=(model_type == "multilingual"))
    test_loader = DataLoader(test_dataset, batch_size=8,
                            collate_fn=lambda batch: collate_fn(batch, multilingual=(model_type == "multilingual")),
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, test_loader, char_mappings, train_dataset.num_languages