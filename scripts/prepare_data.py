import argparse
import json
import os

from iac.data import (
    DEFAULT_RATE,
    DEFAULT_SESSION_TIMEOUT_S,
    HISTORY,
    REPLAY,
    SyntheticDelay,
    build_trace,
    load_conversations,
    prepare_splits,
    read_manifest,
    trace_summary,
    write_splits,
)


def split(args):
    m = write_splits(prepare_splits(args.history, args.replay), args.out)
    for s in m["splits"].values():
        del s["conv_ids"], s["dropped_ids"]
    print(json.dumps(m, indent=1))


def trace(args):
    split_name = os.path.basename(args.split).split(".")[0]
    provenance = {"split": os.path.abspath(args.split), "split_name": split_name}
    manifest = read_manifest(args.split)
    if manifest:
        provenance |= {
            "dataset": manifest["dataset"],
            "source": manifest["source"],
            "split_positions": manifest["splits"].get(split_name, {}).get("positions"),
        }
    t = build_trace(
        load_conversations(args.split),
        rate=args.rate,
        seed=args.seed,
        session_timeout=args.session_timeout,
        synthetic_delay=SyntheticDelay(),
        provenance=provenance,
    )
    t.save(args.out)
    counts = ("split_conversations", "synthetic_conversations", "session_truncated")
    print(json.dumps({"out": args.out, **{k: t.meta[k] for k in counts}, **trace_summary(t)}, indent=1))


def positions(text: str) -> tuple[int, int]:
    lo, hi = (int(x) for x in text.split(":"))
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("split")
    s.add_argument("--out", default="data/splits/wildchat")
    s.add_argument("--history", type=positions, default=HISTORY)
    s.add_argument("--replay", type=positions, default=REPLAY)
    s.set_defaults(fn=split)

    t = sub.add_parser("trace")
    t.add_argument("--split", default="data/splits/wildchat/replay.jsonl.gz")
    t.add_argument("--out", default="traces/wildchat_r3")
    t.add_argument("--rate", type=float, default=DEFAULT_RATE)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--session-timeout", type=float, default=DEFAULT_SESSION_TIMEOUT_S)
    t.set_defaults(fn=trace)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
