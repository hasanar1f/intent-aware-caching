import argparse
import collections
import gzip
import json
import os
from dataclasses import fields

import torch

from iac.classifier import (
    IntentClassifier,
    TrainConfig,
    classification_metrics,
    label_ids,
    load_labeled_jsonl,
    stratified_split,
    train_classifier,
)

DATA = "data/classifier/labeled_conversations_disjoint.jsonl"
CLASSIFIER = "checkpoints/intent_e5_large_instruct_30cls.pt"
DTYPES = ("autocast-bf16", "bf16", "fp32")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_SOURCES = {
    "wildchat": ("allenai/WildChat-1M", "conversation_hash", "conversation"),
    "lmsys": ("lmsys/lmsys-chat-1m", "conversation_id", "conversation"),
    "chatbot-arena": ("lmsys/chatbot_arena_conversations", "question_id", "conversation_a"),
}


def hf_user_turns(source: str, wanted: set[str]) -> dict[str, list[str]]:
    from datasets import load_dataset

    name, id_column, conv_column = HF_SOURCES[source]
    out = {}
    for row in load_dataset(name, split="train").select_columns([id_column, conv_column]):
        conv = row[conv_column] or []
        key = row[id_column]
        if source == "wildchat":
            key = f"{key}:{conv[0].get('turn_identifier')}" if conv else None
        if key in wanted:
            out[key] = [m["content"] for m in conv if m.get("role") == "user"]
    return out


def sharegpt_user_turns(path: str, wanted: set[str]) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        r["id"]: [m["value"] for m in r.get("conversations", []) if m.get("from") == "human"]
        for r in data
        if r.get("id") in wanted
    }


def rebuild(args):
    with gzip.open(args.labels, "rt", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    wanted = collections.defaultdict(set)
    for r in labels:
        wanted[r["source"]].add(r["id"])
    sources = set(args.sources.split(","))
    if "sharegpt" in sources and not args.sharegpt_json:
        raise SystemExit("--sharegpt-json is required for ShareGPT rows (or drop sharegpt from --sources)")
    turns: dict[str, list[str]] = {}
    for source in HF_SOURCES:
        if source in sources:
            turns |= hf_user_turns(source, wanted[source])
    if "sharegpt" in sources:
        turns |= sharegpt_user_turns(args.sharegpt_json, wanted["sharegpt"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    found = collections.Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for r in labels:
            if r["source"] in sources and r["id"] in turns:
                row = {
                    "id": r["id"],
                    "source": r["source"],
                    "label": r["label"],
                    "conversation": turns[r["id"]][: r["turns"]],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                found[r["source"]] += 1
    print({s: f"{found[s]}/{len(wanted[s])}" for s in sorted(wanted) if s in sources})


def train(args):
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    result = train_classifier(args.data, cfg, cache_dir=args.cache_dir or None, device=args.device, progress=True)
    result.checkpoint.save(args.out)
    summary = {"out": args.out, "best_epoch": result.best_epoch, "val": result.val, "test": result.test}
    print(json.dumps(summary, indent=1))
    print(result.report)
    if args.metrics_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, "w") as f:
            extra = {"metadata": result.checkpoint.metadata, "history": result.history, "report": result.report}
            json.dump({**summary, **extra}, f, indent=1)


def evaluate(args):
    clf = IntentClassifier.from_checkpoint(args.classifier, device=args.device, embedder_dtype=args.embedder_dtype)
    rows, labels = load_labeled_jsonl(args.data, num_classes=len(clf.labels))
    cfg = clf.ckpt.metadata.get("config", {})
    splits = stratified_split(rows, labels, cfg.get("val_ratio", 0.1), cfg.get("test_ratio", 0.1), cfg.get("seed", 7))
    split = splits[["train", "val", "test"].index(args.split)]
    y_pred = []
    for i in range(0, len(split), args.batch_size):
        y_pred += clf.predict_ids([r["conversation"] for r in split[i : i + args.batch_size]])
    metrics = classification_metrics(label_ids(split, clf.labels), y_pred)
    print(
        json.dumps(
            {"classifier": args.classifier, "embedder_dtype": clf.embedder_dtype, "split": args.split, **metrics},
            indent=1,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("rebuild", help=rebuild.__doc__, description=rebuild.__doc__)
    s.add_argument("--labels", default="data/classifier/labels.jsonl.gz")
    s.add_argument("--sharegpt-json", help="ShareGPT_V3_unfiltered_cleaned_split.json (for ShareGPT rows)")
    s.add_argument("--sources", default="wildchat,lmsys,chatbot-arena,sharegpt")
    s.add_argument("--out", default=DATA)
    s.set_defaults(fn=rebuild)

    s = sub.add_parser("train", help=train.__doc__, description=train.__doc__)
    s.add_argument("--data", default=DATA, help="labeled JSONL")
    s.add_argument("--out", required=True, help="output checkpoint")
    for f in fields(TrainConfig):
        s.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    s.add_argument("--cache-dir", default="data/cache/classifier_embeddings", help="embedding cache ('' disables it)")
    s.add_argument("--device", default=DEVICE)
    s.add_argument("--metrics-out", help="also write the metrics, per-epoch history and per-class report as JSON")
    s.add_argument("--overwrite", action="store_true", help="replace an existing --out checkpoint")
    s.set_defaults(fn=train)

    s = sub.add_parser("eval", help=evaluate.__doc__, description=evaluate.__doc__)
    s.add_argument("--classifier", default=CLASSIFIER, help="intent head checkpoint")
    s.add_argument("--data", default=DATA, help="its training data")
    s.add_argument("--split", default="test", choices=["train", "val", "test"])
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--embedder-dtype", default="autocast-bf16", choices=DTYPES)
    s.add_argument("--device", default=DEVICE)
    s.set_defaults(fn=evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
