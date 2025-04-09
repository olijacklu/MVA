import os
import numpy as np
import pandas as pd
import re
import random
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

'''
Before reading the dataset, download it from the following URL:
https://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes/20_newsgroup/
'''

def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


def get_folder_paths(combination):
    newspaper_folder_path = "Datasets/20_newsgroups/"

    if combination == "A2":
        folder_paths = ["alt.atheism", "comp.graphics"]
    elif combination == "B2":
        folder_paths = ["talk.politics.mideast", "talk.politics.misc"]
    elif combination == "A5":
        folder_paths = ["comp.graphics", "rec.motorcycles", "rec.sport.baseball", "sci.space", "talk.politics.mideast"]
    elif combination == "B5":
        folder_paths = ["comp.graphics", "comp.os.ms-windows.misc", "rec.autos", "sci.electronics", "talk.politics.misc"]
    
    folder_paths = [newspaper_folder_path + paper for paper in folder_paths]
    return folder_paths


def get_papers_dataframe(folder_paths, n_sample):
    df = pd.DataFrame()
    for folder_path in folder_paths:
        all_files = os.listdir(folder_path)
        sampled_files = random.sample(all_files, n_sample)
        data = []
        for file_name in sampled_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                data.append(content)
        
        df_ = pd.DataFrame(data, columns=['content'])
        df = pd.concat([df, df_], ignore_index=True)
    df["content"] = df["content"].apply(clean_text)
    return df


def vectorize_papers(df, model = "tf-idf"):
    if model == "tf-idf":
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df['content'])
        feature_names = vectorizer.get_feature_names_out()
        df_vectorized = pd.DataFrame(X.toarray(), columns=feature_names)
    return df_vectorized


def generate_labels(df_vectorized, combination, n_sample):
    df_vectorized['label'] = [0]*n_sample + [1]*n_sample if combination in ["A2", "B2"] else [0]*n_sample + [1]*n_sample + [2]*n_sample + [3]*n_sample + [4]*n_sample
    return df_vectorized


def shuffle_and_split(df_vectorized):
    df_vectorized = shuffle(df_vectorized, random_state=42)
    labels = df_vectorized['label'].tolist()
    df_vectorized = df_vectorized.drop(columns=['label'])
    return df_vectorized.values.T, labels


def get_data_papers(combination, n_sample = 100):
    assert combination in {"A2", "B2", "A5", "B5"}, f"Invalid value for combination: '{combination}'. Must be 'A2', 'B2, 'A5', or 'B5'."
    
    folder_paths = get_folder_paths(combination)
        
    df = get_papers_dataframe(folder_paths, n_sample)

    df_vectorized = vectorize_papers(df, model = "tf-idf")

    df_vectorized = generate_labels(df_vectorized, combination, n_sample)
    
    X, labels = shuffle_and_split(df_vectorized)
    
    return X, labels
