import pandas as pd

def read_csv():
    gene_folder_path = "Datasets/TCGA-PANCAN-HiSeq-801x20531/"
    data_df = pd.read_csv(gene_folder_path + "data.csv")
    labels_df = pd.read_csv(gene_folder_path + "labels.csv")
    data_df = data_df.drop(data_df.columns[0], axis=1)
    labels_df = labels_df.drop(labels_df.columns[0], axis=1)
    return data_df, labels_df


def get_int_labels(labels_df, unique_labels):
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    return labels_df.iloc[:, 0].map(label_to_int)

def get_data_gene():

    data_df, labels_df = read_csv()

    unique_labels = labels_df.iloc[:, 0].unique()

    K = len(unique_labels)

    labels_df['labels'] = get_int_labels(labels_df, unique_labels)

    X = data_df.values.T
    true_labels = labels_df['labels'].values

    return X, true_labels, K
