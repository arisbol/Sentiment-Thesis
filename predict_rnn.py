# src/predict_rnn.py
import os, argparse, json, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv

# ---------- Tokenization / Vocab ----------
TOKEN_RE = re.compile(r"[A-Za-zΑ-Ωα-ωΆ-Ώά-ώ]+", flags=re.UNICODE)
PAD, UNK = "<pad>", "<unk>"

def tokenize(text: str):
    t = str(text).lower()
    toks = TOKEN_RE.findall(t)
    return toks if toks else t.split()

class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad_idx = self.stoi.get(PAD, 0)
        self.unk_idx = self.stoi.get(UNK, 1)
    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_idx) for t in tokens]

def pad_batch(seqs, pad_idx, max_len):
    xs = [s[:max_len] for s in seqs]
    maxL = min(max_len, max((len(s) for s in xs), default=0))
    out = [s + [pad_idx]*(maxL - len(s)) for s in xs]
    return torch.tensor(out, dtype=torch.long)

# ---------- Model ----------
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)
    def forward(self, H):
        a = torch.softmax(self.w(H).squeeze(-1), dim=-1)  # [B, T]
        z = (H * a.unsqueeze(-1)).sum(dim=1)              # [B, d]
        return z

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes, pad_idx=0, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2*hidden)
        self.fc   = nn.Sequential(nn.Dropout(dropout), nn.Linear(2*hidden, num_classes))
    def forward(self, x):
        x = self.emb(x)                # [B,T,E]
        H, _ = self.lstm(x)            # [B,T,2H]
        z = self.pool(H)               # [B,2H]
        return self.fc(z)              # [B,C]

# ---------- Inference ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="π.χ. models/custom_rnn (θα διαβάσει checkpoint-best/model.pt)")
    ap.add_argument("--input_csv", required=True, help="CSV με στήλη text")
    ap.add_argument("--output_csv", required=True, help="πού να γράψω τις προβλέψεις")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--proba", action="store_true", help="γράψε και πιθανότητες ανά κλάση")
    ap.add_argument("--threshold", type=float, default=0.6,)
    ap.add_argument("--uncertain_label", default="uncertain",)

    args = ap.parse_args()

    import csv

    df = pd.read_csv(
        args.input_csv,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
        skip_blank_lines=True,
        quoting=csv.QUOTE_MINIMAL,
        keep_default_na=False,   # να ΜΗΝ μετατρέπει "None", "NA" σε NaN αυτόματα
        )
    df.columns = [c.strip() for c in df.columns]
    assert args.text_col in df.columns, f"CSV must contain column '{args.text_col}'"

    texts = (
        df[args.text_col]
        .astype(str)
        .str.strip()
        .tolist()
    )
    # πέτα άδειες, "nan", "None"
    texts = [t for t in texts if t and t.lower() not in {"nan", "none"}]


    # load checkpoint
    ckpt_path = os.path.join(args.model_dir, "checkpoint-best", "model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    labels = cfg["labels"]                       # ['neg','neu','pos']
    id2label = {i: lab for i, lab in enumerate(labels)}

    # rebuild model & vocab
    vocab = Vocab(cfg["vocab"])
    model = BiLSTMClassifier(
        vocab_size=len(vocab.itos),
        emb_dim=cfg["emb_dim"],
        hidden=cfg["hidden"],
        num_classes=len(labels),
        pad_idx=cfg["pad_idx"],
        dropout=0.0
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # tokenize → encode → pad
    seqs = [vocab.encode(tokenize(t)) for t in texts]
    xb = pad_batch(seqs, vocab.pad_idx, max_len=cfg.get("max_len", 128))

    # predict (πάντα υπολογίζουμε softmax για να εφαρμόσουμε threshold)
    with torch.no_grad():
        logits = model(xb)
        proba_t = torch.softmax(logits, dim=-1)       # [B, C]
        pred_ids = torch.argmax(proba_t, dim=-1).cpu().numpy()
        max_proba = torch.max(proba_t, dim=-1).values.cpu().numpy()

    # εφάρμοσε threshold → αν καμία πιθανότητα δεν >= threshold, βάλε uncertain
    pred_labels = [id2label[int(i)] for i in pred_ids]
    for i in range(len(pred_ids)):
        if float(max_proba[i]) < float(args.threshold):
            pred_ids[i] = -1
            pred_labels[i] = args.uncertain_label

    # build output
    out = df.copy()
    out["pred_id"] = pred_ids
    out["pred_label"] = pred_labels


    proba = proba_t.cpu().numpy()
    if args.proba and proba is not None:
        for cid, cname in id2label.items():
            out[f"prob_{cname}"] = proba[:, int(cid)]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"[OK] Saved predictions → {os.path.abspath(args.output_csv)}")

if __name__ == "__main__":
    main()
