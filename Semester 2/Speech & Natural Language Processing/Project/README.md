# Parameter-Efficient Fine-Tuning and Layer Analysis of Self-Supervised Speech Models: A Multilingual Study with ML-SUPERB

This repository contains code for fine-tuning and evaluating pre-trained audio models (like HuBERT and XLSR) on the ML-SUPERB dataset for monolingual and multilingual speech recognition tasks.

## Features

- Low-rank adaptation (LoRA) fine-tuning for efficient model specialization
- Support for multiple tasks:
  - Automatic Speech Recognition (ASR)
  - Language Identification (LID)
  - Joint ASR and LID
- Monolingual and multilingual model training
- Comprehensive evaluation metrics including Character Error Rate (CER)
- Layer weight analysis to understand representation learning across languages

## Project Structure

```
ML-SUPERB-Project/
├── ml_superb.ipynb          # Main notebook for running experiments
├── memory_usage_exp.ipynb   # Notebook for analysis of memory usage using wandb
├── README.md
├── requirements.txt
├── config/                  # Configuration parameters
├── data/                    # Data processing tools
├── evaluation/              # Evaluation metrics
├── interpretability/        # Interpretability tools
├── models/                  # Model definitions
└── training/                # Training methods
```

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/olijacklu/ML-SUPERB-Project.git
cd ML-SUPERB-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your data directory (see next section for further details)

## Data Preparation

The ML-SUPERB dataset includes recordings from multiple languages and sources. To preprocess the data:

1. Download the ML-SUPERB dataset used in the original paper from the following link:
https://drive.google.com/file/d/1QYjl-7vflle__3AfuosAC5VJGiBDvEqz/view
2. Store the downloaded data in a folder where you also wish to save any models or results obtained from the experiments
3. Update the `base_dir` in the notebook `ml_superb.ipynb` to point to your data directory
4. Run the preprocessing function in `ml_superb.ipynb` to create a JSON lookup file for the data

## Usage

The main entry point for this codebase is the `ml_superb.ipynb` notebook, which demonstrates:

1. Loading and preprocessing data
2. Training monolingual ASR models
3. Training multilingual ASR, LID, and joint models
4. Evaluating model performance
5. Visualizing layer weights

You can also use the individual modules in your own scripts:

```python
import json
from models.utils import load_model
from training.monolingual import train_and_evaluate_monolingual
from evaluation.test import test_model

# Load preprocessed data lookup file
with open(f'{base_dir}/ml_superb_dataset.json', 'r') as f:
    datasets = json.load(f)

print(f"Loaded {len(datasets)} language-source pairs")

# Load a pretrained model
upstream_model, feature_extractor = load_model("facebook/hubert-base-ls960")

# Train a monolingual model
trained_model, results, char_mappings = train_and_evaluate_monolingual(
    lang="eng1",
    data_pair="eng_mls",
    upstream_model=model,
    feature_extractor=feature_extractor,
    datasets=datasets
)

# Evaluate the model
test_model(trained_model, feature_extractor, datasets, char_mappings)
```

## Model Training

### Monolingual ASR

Train separate models for individual languages:

```python
for lang, data_pair in monolingual_train_pairs.items():
    model, results, char_mappings = train_and_evaluate_monolingual(
        lang=lang,
        data_pair=data_pair,
        upstream_model=upstream_model,
        feature_extractor=feature_extractor,
        datasets=datasets
    )
```

### Multilingual ASR

Train a single model on multiple languages:

```python
model, results, char_mappings = train_and_evaluate_multilingual(
    upstream_model=upstream_model,
    feature_extractor=feature_extractor,
    datasets=datasets,
    task="asr"
)
```

### Language Identification (LID)

Train a model to identify the language being spoken:

```python
model, results, char_mappings = train_and_evaluate_multilingual(
    upstream_model=upstream_model,
    feature_extractor=feature_extractor,
    datasets=datasets,
    task="lid"
)
```

### Joint ASR+LID

Train a model that performs both ASR and LID simultaneously:

```python
model, results, char_mappings = train_and_evaluate_multilingual(
    upstream_model=upstream_model,
    feature_extractor=feature_extractor,
    datasets=datasets,
    task="asr+lid"
)
```

## Evaluation

Evaluate trained models using Character Error Rate (CER) for ASR and accuracy for LID:

```python
test_model(
    model=model,
    feature_extractor=feature_extractor,
    datasets=datasets,
    char_mappings=char_mappings,
    model_type="multilingual",
    task="asr+lid"
)
```

## Results

After training, you can analyze the model's layer weights to understand which layers are most important for different languages or tasks:

```python
analyze_layer_weights(
    models_by_language,
    title="Layer Weight Analysis",
    save_path="layer_weights.png"
)
```

## References

1. Jiatong Shi, Dan Berrebbi, et al. Ml-superb: Multilingual speech universal performance benchmark. In INTERSPEECH, 2023.
2. Jiatong Shi, Shih-Heng Wang, et al. Ml-superb 2.0: Benchmarking multilingual speech models across modeling constraints, languages, and datasets, 2024.
