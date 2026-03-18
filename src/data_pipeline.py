import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def split_data(df):
    return train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])