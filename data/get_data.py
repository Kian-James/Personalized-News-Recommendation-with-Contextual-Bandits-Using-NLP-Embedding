import os
import pandas as pd
import urllib.request

DATA_DIR = "data"
URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    raw_path = os.path.join(DATA_DIR, "raw.csv")
    out_path = os.path.join(DATA_DIR, "news.csv")

    print("Downloading...")
    urllib.request.urlretrieve(URL, raw_path)

    df = pd.read_csv(raw_path, header=None)
    df.columns = ["label", "title", "desc"]

    df["text"] = df["title"] + " " + df["desc"]
    df["label"] = df["label"] - 1

    df = df[["text", "label"]]
    df.to_csv(out_path, index=False)

    os.remove(raw_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()