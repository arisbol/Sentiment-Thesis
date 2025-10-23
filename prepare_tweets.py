
import os, re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/tweets.csv"   
OUT_DIR  = "data"                        

def load_df(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
         return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")

def normalize_labels(df):
    df = df[["label", "text"]].dropna().copy()

    ser = df["label"]

    mapped = None
    try:
        vals = set(ser.astype(int).unique().tolist())
        if vals.issubset({0, 1, 2}) and vals == {0, 1, 2}:
            m = {0: "neg", 1: "neu", 2: "pos"}
            mapped = ser.astype(int).map(m)
        elif vals.issubset({-1, 0, 1}) and vals == {-1, 0, 1}:
            m = {-1: "neg", 0: "neu", 1: "pos"}
            mapped = ser.astype(int).map(m)
    except Exception:
        pass

    if mapped is None:
        #Metatrepei oles tis times se string,ola ta grammata se peza,aferei kena stin arxi kai sto telos
        mapped = ser.astype(str).str.lower().str.strip()
        mapped = mapped.replace({
            "positive": "pos", "negative": "neg", "neutral": "neu",
            "θετικό": "pos", "αρνητικό": "neg", "ουδέτερο": "neu",
        })

    df["label"] = mapped

    keep = {"neg", "neu", "pos"}
    df = df[df["label"].isin(keep)].copy()
    return df

def clean_text(df):
    url_re = re.compile(r"https?://\S+")
    mention_re = re.compile(r"@\w+")
    def clean(t):
        t = str(t)
        t = url_re.sub("", t)
        t = mention_re.sub("", t)
        t = t.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        return t.strip()
    df["text"] = df["text"].apply(clean)
    df = df[df["text"].str.len() >= 3]
    df = df.drop_duplicates(subset=["text"])
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_df(RAW_PATH)
    assert {"label","text"}.issubset(df.columns), "Το CSV πρέπει να έχει στήλες: label,text"
    df = normalize_labels(df)
    df = clean_text(df)
    train, tmp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val, test  = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp["label"])
    print("Counts:")
    
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(name, d["label"].value_counts().to_dict())

    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    print("Wrote data/train.csv, data/val.csv, data/test.csv")

if __name__ == "__main__":
    main()
