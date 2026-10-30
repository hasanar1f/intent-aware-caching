import argparse
import json
import os

from iac.classifier import BATCH_SIZE, IntentClassifier, conversation_intents_fn, label_split, turn_records
from iac.data import DEFAULT_SESSION_TIMEOUT_S, SyntheticDelay, load_conversations
from iac.stats import ReuseStats


class LazyClassifier:
    def __init__(self, ckpt: str, device: str, embedder_dtype: str):
        self.args = (ckpt, device, embedder_dtype)
        self.model = None

    def predict(self, batch_turns):
        if self.model is None:
            ckpt, device, dtype = self.args
            self.model = IntentClassifier.from_checkpoint(ckpt, device=device, embedder_dtype=dtype)
        return self.model.predict(batch_turns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--classifier", default="checkpoints/intent_e5_large_instruct_30cls.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-timeout", type=float, default=DEFAULT_SESSION_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embedder-dtype", default="bf16", choices=["autocast-bf16", "bf16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    convs = load_conversations(args.split)
    classifier = LazyClassifier(args.classifier, args.device, args.embedder_dtype)
    labels = label_split(convs, args.split, args.classifier, classifier, args.embedder_dtype, args.batch_size)
    records = turn_records(convs, conversation_intents_fn(labels), args.session_timeout, SyntheticDelay(), args.seed)
    meta = {"split": os.path.relpath(args.split), "conversations": len(convs), "requests": len(records)}
    stats = ReuseStats.learn(records, session_timeout=args.session_timeout, metadata=meta)
    stats.save(args.out)
    print(json.dumps({"out": args.out, **meta, "states": len(stats)}, indent=1))


if __name__ == "__main__":
    main()
