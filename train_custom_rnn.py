import os, re, json, argparse, math, random
from collections import Counter
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import evaluate

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

TOKEN_RE = re.compile(r"[A-Za-zΑ-Ωα-ωΆ-Ώά-ώ]+", flags=re.UNICODE)

def tokenize(text: str):
    t = str(text).lower()
    toks = TOKEN_RE.findall(t)
    return toks if toks else t.split()

PAD, UNK = "<pad>", "<unk>"

class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad_token = PAD
        self.unk_token = UNK
        self.pad_idx = self.stoi[PAD]
        self.unk_idx = self.stoi[UNK]

    def __len__(self):
        return len(self.itos)
    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_idx) for t in tokens]
    def decode(self, ids):
        return [self.itos[i] if 0 <= i < len(self.itos) else UNK for i in ids]

    
class TextClsDataset(Dataset):
    def __init__(self, df, str2id, vocab):
        self.vocab = vocab
        self.str2id = str2id
        self.X = [vocab.encode(tokenize(t)) for t in df["text"].tolist()]
        self.y = [str2id[s] for s in df["label"].tolist()]

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

@dataclass
class CollateConfig:
    max_len: int
    pad_idx: int

def collate_batch(samples, cfg: CollateConfig):
    xs, ys = zip(*samples)
    xs = [x[:cfg.max_len] for x in xs]
    lens = [len(x) for x in xs]
    maxL = min(cfg.max_len, max(lens)) if lens else cfg.max_len
    batch = []
    for x in xs:
        pad = [cfg.pad_idx] * (maxL - len(x))
        batch.append(x + pad)
    return torch.tensor(batch, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)
    def forward(self, H):              # H: [B, T, d]
        a = torch.softmax(self.w(H).squeeze(-1), dim=-1)  # [B, T]
        z = (H * a.unsqueeze(-1)).sum(dim=1)              # [B, d]
        return z, a

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes, pad_idx=0, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2*hidden)
        self.fc   = nn.Sequential(nn.Dropout(dropout), nn.Linear(2*hidden, num_classes))
    def forward(self, x):
        x = self.emb(x)                # [B, T, E]
        H, _ = self.lstm(x)            # [B, T, 2H]
        z, a = self.pool(H)            # [B, 2H]
        logits = self.fc(z)            # [B, C]
        return logits



def eval_epoch(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=-1)
            all_y.extend(yb.cpu().numpy().tolist())
            all_p.extend(pred.cpu().numpy().tolist())
    macro_f1 = f1_score(all_y, all_p, average="macro")
    return macro_f1, np.array(all_y), np.array(all_p)

def train_model(model, train_loader, val_loader, device, epochs, lr, class_weights=None, patience=3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)  # weights ανισορροπίας

    best_f1, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()
        val_f1, _, _ = eval_epoch(model, val_loader, device)
        print(f"[{ep:02d}/{epochs}] loss={running/len(train_loader):.4f} | val_macroF1={val_f1:.4f}")

        if val_f1 > best_f1 + 1e-4:
            best_f1, best_state, wait = val_f1, {k:v.cpu() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep}.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def read_csv(p):
    df = pd.read_csv(p)
    assert {"text","label"}.issubset(df.columns), "CSV must have columns: text,label"
    return df

def build_vocab_from_df(dfs, min_freq=1, max_size=None):
   #ftiaxnei leksilogio me tin stili text
    counter = Counter()
    for df in dfs:
        for t in df["text"]:
            toks = tokenize(t)                 
            counter.update(toks)

    words = [w for (w, c) in counter.most_common() if c >= min_freq]

    itos = [PAD, UNK] + words

    if max_size is not None:
        itos = itos[:max_size]

    print("Unique tokens total:", len(counter), "| after min_freq:", len(words))

    return Vocab(itos)


def main(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_df, val_df, test_df = read_csv(args.train), read_csv(args.val), read_csv(args.test)
    labels = sorted(pd.concat([train_df["label"], val_df["label"], test_df["label"]]).unique().tolist())
    str2id = {s:i for i,s in enumerate(labels)}
    id2str = {i:s for s,i in str2id.items()}

    vocab = build_vocab_from_df([train_df, val_df], 
                            min_freq=args.min_freq, 
                            max_size=args.vocab_size)

    print("Vocab size:", len(vocab.itos))

    train_ds = TextClsDataset(train_df, str2id, vocab)
    val_ds   = TextClsDataset(val_df,   str2id, vocab)
    test_ds  = TextClsDataset(test_df,  str2id, vocab)

    coll_cfg = CollateConfig(max_len=args.max_len, pad_idx=vocab.pad_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, coll_cfg))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_batch(b, coll_cfg))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_batch(b, coll_cfg))

    cw_tensor = None
    if args.use_class_weights:
        y = np.array([str2id[s] for s in train_df["label"].tolist()])
        cls = np.unique(y)
        w = compute_class_weight("balanced", classes=cls, y=y)
        cw_tensor = torch.tensor(w, dtype=torch.float)
        print("Class weights:", w)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        vocab_size=len(vocab.itos),
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        num_classes=len(labels),
        pad_idx=vocab.pad_idx,
        dropout=args.dropout
    )

    model, best_val_f1 = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, class_weights=cw_tensor, patience=args.patience
    )

    #test
    macro_f1, y_true, y_pred = eval_epoch(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=[id2str[i] for i in range(len(labels))], output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"[OK] Test Macro-F1: {macro_f1:.4f}")

    ckpt_dir = os.path.join(args.out_dir, "checkpoint-best")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab": vocab.itos,
                "pad_idx": vocab.pad_idx,
                "emb_dim": args.emb_dim,
                "hidden": args.hidden,
                "max_len": args.max_len,
                "labels": labels
            }
        },
        os.path.join(ckpt_dir, "model.pt")
    )
    metrics = {
        "labels": labels,
        "best_val_macro_f1": float(best_val_f1),
        "test_macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": cm
    }
    with open(os.path.join(args.out_dir, "metrics_custom_rnn.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--val",   default="data/val.csv")
    ap.add_argument("--test",  default="data/test.csv")
    ap.add_argument("--emb_dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--vocab_size", type=int, default=1000000000)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="models/custom_rnn")
    args = ap.parse_args()
    main(args)

