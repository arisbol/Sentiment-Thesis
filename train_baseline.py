import json, os, argparse, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

def load_csv(path):
    df = pd.read_csv(path)
    assert {"text","label"}.issubset(df.columns), "CSV must have columns: text,label"
    return df

def encode_labels(y):
    classes = sorted(pd.Series(y).unique().tolist())
    cls2id = {c:i for i,c in enumerate(classes)}
    y_ids = np.array([cls2id[c] for c in y])
    return y_ids, cls2id

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    train_df = load_csv(args.train)
    val_df   = load_csv(args.val)
    test_df  = load_csv(args.test)

    if args.merge_train_val:
        train_df = pd.concat([train_df, val_df], ignore_index=True)


    y_train_ids, cls2id = encode_labels(train_df["label"])
    id2cls = {i:c for c,i in cls2id.items()}
    label_names = [id2cls[i] for i in range(len(id2cls))]   #neg ,neu,pos

    X_train = train_df["text"].astype(str)
    X_test  = test_df["text"].astype(str)
    y_test_ids = np.array([cls2id[c] for c in test_df["label"]])

    classes = np.unique(y_train_ids)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_ids)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1,2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            n_jobs=-1
        ))
    ])

    # train
    pipe.fit(X_train, y_train_ids)

    # predict στο test
    y_pred_ids = pipe.predict(X_test)

    macro_f1 = f1_score(y_test_ids, y_pred_ids, average="macro")
    acc = accuracy_score(y_test_ids, y_pred_ids)
    report = classification_report(
        y_test_ids, y_pred_ids,
        target_names=label_names,
        output_dict=True
    )
    cm = confusion_matrix(y_test_ids, y_pred_ids, labels=list(range(len(label_names)))).tolist()


    out = {
        "labels": label_names,
        "test_macro_f1": float(macro_f1),
        "test_accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm
    }
    with open(os.path.join(args.out_dir, "metrics_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    joblib.dump(pipe, os.path.join(args.out_dir, "model.joblib"))
    print(f"[OK] Macro-F1: {macro_f1:.4f} | Acc: {acc:.4f} | Saved to {args.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out_dir", default="models/baseline")
    p.add_argument("--merge_train_val", action="store_true")
    args = p.parse_args()
    main(args)
