# utils/create_multilabel.py

import pandas as pd

def create_multilabel(df):
    label_cols = ['fuel', 'machine', 'service', 'price', 'part', 'others']
    sentiment_values = ['negative', 'neutral', 'positive']

    multilabel_targets = {}

    for label in label_cols:
        for sentiment in sentiment_values:
            multilabel_targets[f"{label}_{sentiment}"] = (df[label] == sentiment).astype(int)

    multilabel_df = pd.DataFrame(multilabel_targets)

    return multilabel_df