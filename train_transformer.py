import os, json, argparse, numpy as np, pandas as pd
import evaluate, torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, set_seed
)
from sklearn.utils.class_weight import compute_class_weight


def load_csvs(train_p, val_p, test_p):
    def read(p):
        df = pd.read_csv(p)
        assert {"text","label"}.issubset(df.columns), "CSV must have columns: text,label"
        return df
    return read(train_p), read(val_p), read(test_p)

def make_label_features(df_list):
    labels = sorted(pd.concat([df["label"] for df in df_list]).unique().tolist())
    str2id = {s:i for i,s in enumerate(labels)}
    for df in df_list:
        df["labels"] = df["label"].map(str2id)
    return labels, str2id

def compute_class_weights(y):
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}

def main(args):
    set_seed(args.seed)

    train_df, val_df, test_df = load_csvs(args.train, args.val, args.test)
    labels, str2id = make_label_features([train_df, val_df, test_df])
    id2str = {v: k for k, v in str2id.items()}

    train_df["text"] = train_df["text"].astype(str)
    val_df["text"]   = val_df["text"].astype(str)
    test_df["text"]  = test_df["text"].astype(str)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False),
        "validation": Dataset.from_pandas(val_df[["text", "labels"]], preserve_index=False),
        "test": Dataset.from_pandas(test_df[["text", "labels"]], preserve_index=False),
    })
    print(">>> before tokenize | train columns:", ds["train"].column_names, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    def tok(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_len)
    #h tok pernaei batches keimenon anti gia ena ena, tvra katha apotelesma anti gia mono text tha exei kai input_ids kai attention mask
    ds = ds.map(tok, batched=True)
    for split in ["train", "validation", "test"]:
            cols_to_drop = [c for c in ["text", "__index_level_0__"] if c in ds[split].column_names]
            if cols_to_drop:
                ds[split] = ds[split].remove_columns(cols_to_drop)

    print(">>> after tokenize | train columns:", ds["train"].column_names, flush=True)
    print(">>> sample keys:", ds["train"][0].keys(), flush=True)   # πρέπει να δεις input_ids, attention_mask, labels


    class_weights = None
    if args.use_class_weights:
        y = np.array(train_df["labels"])
        cw = compute_class_weights(y)
        class_weights = torch.tensor([cw[i] for i in range(len(labels))], dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_ckpt, num_labels=len(labels), id2label=id2str, label2id=str2id
    )

    from copy import deepcopy
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            inputs = deepcopy(inputs)
            labels = inputs.pop("labels")
            #bazw sto montelo ola ta alla inputs (attentionmask, input ids)
            outputs = model(**inputs)
            logits = outputs.get("logits", None)
            if logits is None:
                return super().compute_loss(model, inputs, return_outputs, **kwargs)
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(logits.device) if class_weights is not None else None
        )
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss



    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric_f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = metric_f1.compute(predictions=preds, references=labels_ids, average="macro")["f1"]
        return {"macro_f1": macro_f1}

    os.makedirs(args.out_dir, exist_ok=True)

    common_args = dict(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        


    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **common_args)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **common_args)
        

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
)

    #train evaluate save
    print(">>> trainer created, starting training...", flush=True)
    trainer.train()
    print(">>> evaluating on test...", flush=True)

    pred_out = trainer.predict(ds["test"])
    logits = pred_out.predictions
    y_true = pred_out.label_ids
    y_pred = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=[id2str[i] for i in range(len(labels))],
        output_dict=True
        )
    cm = confusion_matrix(y_true, y_pred).tolist()

    out = {
        "labels": labels,
        "test_macro_f1": float(macro_f1),
        "test_accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm
        }

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.abspath(os.path.join(args.out_dir, "metrics_transformer.json"))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Metrics -> {metrics_path}", flush=True)

    trainer.save_model(os.path.join(args.out_dir, "checkpoint-best"))
    print(f"[OK] Test Macro-F1: {macro_f1:.4f} | Acc: {acc:.4f}", flush=True)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--val",   default="data/val.csv")
    ap.add_argument("--test",  default="data/test.csv")
    ap.add_argument("--model_ckpt", default="distilbert-base-multilingual-cased")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="models/transformer/distilmbert")

    args = ap.parse_args()
    main(args)



