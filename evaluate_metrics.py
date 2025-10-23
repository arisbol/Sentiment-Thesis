import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_df_from_report(labels, report):
    rows = []
    for lab in labels:
        vals = report.get(lab, {})
        rows.append({
            "label": lab,
            "precision": float(vals.get("precision", 0.0)),
            "recall": float(vals.get("recall", 0.0)),
            "f1-score": float(vals.get("f1-score", vals.get("f1", 0.0))),
            "support": int(vals.get("support", 0))
        })
    df = pd.DataFrame(rows).set_index("label")
    for avg in ("macro avg", "weighted avg", "micro avg"):
        if avg in report:
            vals = report[avg]
            df.loc[avg] = {
                "precision": float(vals.get("precision", 0.0)),
                "recall": float(vals.get("recall", 0.0)),
                "f1-score": float(vals.get("f1-score", vals.get("f1", 0.0))),
                "support": int(vals.get("support", 0)) if vals.get("support") is not None else df["support"].sum()
            }
    return df

def plot_confusion_matrix(cm, labels, outpath):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm_arr, interpolation='nearest')
    ax.set_title("Confusion matrix")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(int(cm_arr[i, j]), 'd'), ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)

def plot_per_class_prf(df, outpath):
    label_rows = [r for r in df.index if r not in ("macro avg", "weighted avg", "micro avg")]
    df_plot = df.loc[label_rows, ["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.6*len(df_plot))))
    x = np.arange(len(df_plot))
    width = 0.25
    ax.bar(x - width, df_plot["precision"], width, label="precision")
    ax.bar(x, df_plot["recall"], width, label="recall")
    ax.bar(x + width, df_plot["f1-score"], width, label="f1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)

def main(args):
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path.resolve()}")

    metrics = load_metrics(metrics_path)
    labels = metrics.get("labels") or list(metrics.get("classification_report", {}).keys())
    report = metrics.get("classification_report", {})
    cm = metrics.get("confusion_matrix", None)

    df = build_df_from_report(labels, report)

    outdir = metrics_path.parent
    (outdir / "per_class_metrics.csv").write_text(df.to_csv(index=True, encoding="utf-8"))
    print("Saved per_class_metrics.csv ->", (outdir / "per_class_metrics.csv").resolve())

    if cm is not None:
        cm_out = outdir / "confusion_matrix.png"
        plot_confusion_matrix(cm, labels, cm_out)
        print("Saved confusion_matrix.png ->", cm_out.resolve())
    else:
        print("No confusion_matrix in metrics JSON; skipping confusion matrix plot.")

    prf_out = outdir / "per_class_prf.png"
    plot_per_class_prf(df, prf_out)
    print("Saved per_class_prf.png ->", prf_out.resolve())

    print("\nSummary:")
    print("Test macro F1:", metrics.get("test_macro_f1", metrics.get("test_macroF1", "N/A")))
    print(df)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to metrics JSON (e.g. models/custom_rnn/metrics_custom_rnn.json)")
    parsed = ap.parse_args()
    main(parsed)
