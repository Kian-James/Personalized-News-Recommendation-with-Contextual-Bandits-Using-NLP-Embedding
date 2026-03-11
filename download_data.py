# download_data.py
import os
import shutil

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

dataset_source = "path_to_your_existing_dataset/news_dataset.csv"
raw_dataset_path = "data/raw/news_dataset.csv"

try:
    shutil.copy(dataset_source, raw_dataset_path)
    print(f"Dataset copied to {raw_dataset_path}")
except FileNotFoundError:
    print("Dataset not found. Please check the source path.")

# Optional: placeholder for preprocessing
processed_dataset_path = "data/processed/news_dataset_clean.csv"
if not os.path.exists(processed_dataset_path):
    print("Processing dataset...")
    # Add your preprocessing code here, e.g.,
    # import pandas as pd
    # df = pd.read_csv(raw_dataset_path)
    # df.to_csv(processed_dataset_path, index=False)
    print(f"Processed dataset will be saved to {processed_dataset_path}")
else:
    print(f"Processed dataset already exists at {processed_dataset_path}")
