# LoRA-Adapted RoBERTa for Portuguese News Classification & SuperGLUE Tasks

## Group Members
- **Paulo SILVA**
- **Oliver JACK**

## Overview
This repository contains two Jupyter notebooks for fine-tuning RoBERTa using Low-Rank Adaptation (LoRA) on different datasets. One focuses on classifying Portuguese news articles, while the other fine-tunes and evaluates the model on SuperGLUE benchmark tasks.

## 1. LoRA-Adapted RoBERTa for Portuguese News Classification

### Background
This project fine-tunes a pre-trained English RoBERTa model to classify news articles in Portuguese using LoRA. LoRA enables efficient adaptation of large language models to new languages and tasks with minimal computational resources.

This implementation is based on the tutorial [Lightweight RoBERTa Sequence Classification Fine-tuning with LoRA](https://achimoraites.medium.com/lightweight-roberta-sequence-classification-fine-tuning-with-lora-using-the-hugging-face-peft-8dd9edf99d19), which demonstrates LoRA-based fine-tuning for sequence classification in English. We've adapted this approach to cross-lingual transfer learning from English to Portuguese.

### Dataset
We used the [LIACC/Emakhuwa-Portuguese-News-MT](https://huggingface.co/datasets/LIACC/Emakhuwa-Portuguese-News-MT) dataset from Hugging Face, which contains Portuguese news articles with their corresponding categories.

#### Dataset Preprocessing Steps
The original dataset consisted of three splits: *train (17.4k samples), validation (964 samples), and test (993 samples)*. However, a detailed analysis revealed several issues:

1. *Label Mismatch Across Splits*: Some categories in the validation and test sets were missing in the training set, making evaluation unreliable.
2. *Overlapping Categories*: Certain labels were highly ambiguous, even for native speakers. For example, the sentence:  
   - "Castro voltou a exigir que os Estados Unidos eliminem o embargo contra Cuba em vigor há 53 anos."  
     ("Castro once again demanded that the United States eliminate the embargo against Cuba that has been in place for 53 years.")  
   - This could be categorized as *"política" (politics)* or *"economia" (economy)* but was actually labeled as *"mundo" (world)*.
3. *Uneven Split Proportions*: The original dataset's test and validation sizes were very small relative to the training set.

#### Solution & Final Dataset
To ensure a more balanced and meaningful dataset, we performed the following preprocessing steps:

- *Filtered the dataset to retain only five core categories*: `'cultura', 'desporto', 'economia', 'mundo', 'saude'`.
- *Merged all original splits* into a single dataset (19.3K samples → 14.3K after filtering).
- *Created a new shuffled split*:  
  - *Train*: 11,478 samples (80%)  
  - *Validation*: 1,435 samples (10%)  
  - *Test*: 1,435 samples (10%)  
- *Applied tokenization* using a RoBERTa tokenizer to preprocess the text data before model training.

This restructuring allows for more reliable model evaluation and ensures that all labels are present in every split.

### Results
Overall, we managed to obtain strong results in terms of accuracy, while also correctly passing each of our dummy tests.
For the setup r=8 and 1 epoch, we managed to obtain:
Base model performance: 0.166
LoRA Fine-tuned model performance: 0.654
When moving to r=64 and 20 epochs, this improved to:
LoRA Fine-tuned model performance: 0.734

## 2. LoRA Roberta Finetuning & Evaluation on SuperGLUE Tasks

### Overview
This notebook fine-tunes and evaluates a LoRA-modified RoBERTa model on SuperGLUE benchmark tasks, supporting multiple tasks such as `boolq`, `cb`, `copa`, `multirc`, `rte`, `wic`, `wsc`, and `wsc.fixed`.

### Dependencies
Ensure you have the following libraries installed before running the notebooks:

pip install transformers datasets peft accelerate bitsandbytes evaluate torch tqdm numpy

### Usage
To run SuperGLUE experiments, change the TASK variable in the notebook:

TASK = "rte"

### Running the Notebooks
Open a terminal and navigate to the project directory.

Start Jupyter Notebook:

jupyter notebook

Open either lora_pt.ipynb or lora_roberta_superglue.ipynb and execute the cells sequentially.

### Results
The table below presents the performance of LoRA across a subset of SuperGLUE benchmark tasks: BoolQ, CB, COPA, MultiRC, RTE, WiC, WSC, and WSC.fixed. Each column represents a different task, and the values indicate the performance scores for each approach in terms of accuracy (in percentage).

| BoolQ | CB   | COPA | MultiRC | RTE  | WiC  | WSC  | WSC.fixed |
|-------|------|------|---------|------|------|------|-----------|
| 77.4  | 75.7 | 70.6 | 70.0    | 60.6 | 69.5 | 64.3 | 64.3      |

## Hardware Requirements
For optimal performance, it is recommended to run these notebooks on a GPU-enabled environment.

## Acknowledgments
These implementations utilize Hugging Face's transformers, datasets, peft, and accelerate libraries to optimize training efficiency using LoRA.
