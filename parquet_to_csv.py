import pandas as pd

df = pd.read_parquet("data/parquet/train-00000-of-00001.parquet")

label_map = {0: "neg", 1: "neu", 2: "pos"}
df["label"] = df["label"].map(label_map)

# krata mono text label
df = df[["text", "label"]]

df.to_csv("data/raw/tweets.csv", index=False, encoding="utf-8")

print("Saved data/raw/tweets.csv with shape:", df.shape)
