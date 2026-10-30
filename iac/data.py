from __future__ import annotations

import csv
import glob
import gzip
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

MAX_CLASSIFIER_TURNS = 4
HISTORY = (0, 40_000)
REPLAY = (40_000, 50_000)
WILDCHAT_ROW_LIMIT = 100_000
DEFAULT_SESSION_TIMEOUT_S = 1800.0
DEFAULT_RATE = 3.0
MAX_UNIX_TIMESTAMP = 2_000_000_000.0

USER_ROLES = frozenset({"user", "human"})
ASSISTANT_ROLES = frozenset({"assistant", "llm", "gpt", "chatgpt", "bing", "bard"})

WILDCHAT_REPO = "allenai/WildChat-1M"

Message = tuple[Optional[str], Optional[str], Optional[float]]


@dataclass
class Turn:
    user: str
    assistant: str
    timestamp: Optional[float] = None


@dataclass
class Conversation:
    conv_id: str
    source: str
    turns: list[Turn]
    language: Optional[str] = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> Conversation:
        return cls(d["conv_id"], d["source"], [Turn(**t) for t in d["turns"]], d.get("language"))

    def messages(self, turn: int) -> list[dict]:
        msgs = []
        for t in self.turns[: turn - 1]:
            msgs += [{"role": "user", "content": t.user}, {"role": "assistant", "content": t.assistant}]
        return msgs + [{"role": "user", "content": self.turns[turn - 1].user}]

    def classifier_turns(self, turn: int) -> list[str]:
        return [t.user for t in self.turns[: min(turn, MAX_CLASSIFIER_TURNS)]]


def _open(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", compresslevel=6)
    return open(path, mode, encoding="utf-8")


def save_conversations(convs: Iterable[Conversation], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with _open(path, "w") as f:
        for c in convs:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")


def iter_conversations(path: str) -> Iterator[Conversation]:
    with _open(path, "r") as f:
        for line in f:
            if line.strip():
                yield Conversation.from_json(json.loads(line))


def load_conversations(path: str) -> list[Conversation]:
    return list(iter_conversations(path))


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class TraceRequest:
    req_id: int
    conv_id: str
    turn: int
    arrival: float
    next_arrival: float = math.inf


@dataclass
class Trace:
    meta: dict
    conversations: dict[str, Conversation]
    requests: list[TraceRequest] = field(default_factory=list)
    path: Optional[str] = None

    def conversation(self, req: TraceRequest) -> Conversation:
        return self.conversations[req.conv_id]

    def messages(self, req: TraceRequest) -> list[dict]:
        return self.conversations[req.conv_id].messages(req.turn)

    def classifier_turns(self, req: TraceRequest) -> list[str]:
        return self.conversations[req.conv_id].classifier_turns(req.turn)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=1)
        save_conversations(self.conversations.values(), os.path.join(path, "conversations.jsonl.gz"))
        with open(os.path.join(path, "requests.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["req_id", "conv_id", "turn", "arrival", "next_arrival"])
            for r in self.requests:
                w.writerow([r.req_id, r.conv_id, r.turn, repr(float(r.arrival)), repr(float(r.next_arrival))])
        self.path = path

    @classmethod
    def load(cls, path: str) -> Trace:
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        convs = {c.conv_id: c for c in iter_conversations(os.path.join(path, "conversations.jsonl.gz"))}
        with open(os.path.join(path, "requests.csv"), newline="") as f:
            requests = [
                TraceRequest(
                    int(r["req_id"]), r["conv_id"], int(r["turn"]), float(r["arrival"]), float(r["next_arrival"])
                )
                for r in csv.DictReader(f)
            ]
        return cls(meta, convs, requests, path)


@dataclass
class Candidate:
    source_row: int
    conv_id: str
    conversation: Optional[Conversation]
    drop_reason: Optional[str] = None


@dataclass
class Split:
    name: str
    positions: tuple[int, int]
    conversations: list[Conversation] = field(default_factory=list)
    dropped: dict[str, list[str]] = field(default_factory=dict)
    source_rows: Optional[tuple[int, int]] = None


@dataclass
class DatasetSplits:
    dataset: str
    source: dict
    splits: dict[str, Split]


def _role(role: Optional[str]) -> Optional[str]:
    role = (role or "").lower()
    if role in USER_ROLES:
        return "user"
    if role in ASSISTANT_ROLES:
        return "assistant"
    return None


def build_turns(messages: Iterable[Message], timestamp_role: Optional[str]) -> list[Turn]:
    turns: list[Turn] = []
    user: list[str] = []
    assistant: list[str] = []
    ts: Optional[float] = None
    for raw_role, text, t in messages:
        role = _role(raw_role)
        if role is None or (role == "assistant" and not user):
            continue
        if role == "user" and assistant:
            turns.append(Turn("\n\n".join(user), "\n\n".join(assistant), ts))
            user, assistant, ts = [], [], None
        (user if role == "user" else assistant).append(text or "")
        if role == timestamp_role and ts is None:
            ts = t
    if user and assistant:
        turns.append(Turn("\n\n".join(user), "\n\n".join(assistant), ts))
    return turns


def arrival_times(raw: Sequence[Optional[float]]) -> Optional[list[float]]:
    if all(x is None for x in raw):
        return None
    filled = list(raw)
    for i in range(1, len(filled)):
        if filled[i] is None:
            filled[i] = filled[i - 1]
    for i in range(len(filled) - 2, -1, -1):
        if filled[i] is None:
            filled[i] = filled[i + 1]
    out = [float(filled[0])]
    for prev, cur in zip(filled, filled[1:]):
        out.append(out[-1] + max(0.0, float(cur) - float(prev)))
    return out


def make_candidate(
    source_row: int,
    conv_id: str,
    source: str,
    messages: Iterable[Message],
    timestamp_role: Optional[str],
    language: Optional[str] = None,
) -> Candidate:
    turns = build_turns(messages, timestamp_role)
    if not turns:
        return Candidate(source_row, conv_id, None, "no_complete_turn")
    if timestamp_role is not None:
        times = arrival_times([t.timestamp for t in turns])
        if times is None:
            return Candidate(source_row, conv_id, None, "no_timestamp")
        for turn, ts in zip(turns, times):
            turn.timestamp = ts
    return Candidate(source_row, conv_id, Conversation(conv_id, source, turns, language))


def timestamp_gaps(conv: Conversation) -> list[float]:
    return [b.timestamp - a.timestamp for a, b in zip(conv.turns, conv.turns[1:])]


def session_turns(gaps: Sequence[float], session_timeout: Optional[float]) -> int:
    if session_timeout is not None:
        for i, gap in enumerate(gaps):
            if gap > session_timeout:
                return i + 1
    return len(gaps) + 1


def _unix(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        value = ts.timestamp()
    else:
        s = str(ts).strip()
        try:
            value = float(s[3:] if s.startswith("ts:") else s)
        except ValueError:
            return None
    return value if 0.0 < value <= MAX_UNIX_TIMESTAMP else None


def _hub_snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, repo_type="dataset", local_files_only=True)


def _iter_parquet_rows(repo_id: str, columns: list[str]) -> Iterator[dict]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(_hub_snapshot(repo_id), "data", "train-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no cached parquet shards for {repo_id}")
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
            yield from batch.to_pylist()


def wildchat_candidates(limit: int) -> Iterator[Candidate]:
    columns = ["conversation_hash", "conversation", "turn", "timestamp", "hashed_ip", "language"]
    seen: set[str] = set()
    n = 0
    for row, r in enumerate(_iter_parquet_rows(WILDCHAT_REPO, columns)):
        if row >= WILDCHAT_ROW_LIMIT or n >= limit:
            return
        conv_id = str(r["conversation_hash"])
        if conv_id in seen:
            continue
        seen.add(conv_id)
        if any(r[k] is None for k in ("conversation", "turn", "timestamp", "hashed_ip")):
            continue
        n += 1
        messages = [(m["role"], m["content"], _unix(m["timestamp"])) for m in r["conversation"]]
        yield make_candidate(row, conv_id, "wildchat", messages, "assistant", r["language"])


def select_splits(stream: Iterable[Candidate], ranges: Mapping[str, tuple[int, int]]) -> dict[str, Split]:
    splits = {name: Split(name, tuple(r)) for name, r in ranges.items()}
    end = max(r[1] for r in ranges.values())
    for position, cand in enumerate(stream):
        if position >= end:
            break
        for split in splits.values():
            if not split.positions[0] <= position < split.positions[1]:
                continue
            lo, hi = split.source_rows or (cand.source_row, cand.source_row + 1)
            split.source_rows = (min(lo, cand.source_row), max(hi, cand.source_row + 1))
            if cand.conversation is None:
                split.dropped.setdefault(cand.drop_reason, []).append(cand.conv_id)
            else:
                split.conversations.append(cand.conversation)
    return splits


def prepare_splits(history: tuple[int, int] = HISTORY, replay: tuple[int, int] = REPLAY) -> DatasetSplits:
    info = {"repo": WILDCHAT_REPO, "revision": os.path.basename(_hub_snapshot(WILDCHAT_REPO))}
    stream = wildchat_candidates(max(history[1], replay[1]))
    return DatasetSplits("wildchat", info, select_splits(stream, {"history": history, "replay": replay}))


def conversation_stats(convs: Sequence[Conversation], session_timeout: float = DEFAULT_SESSION_TIMEOUT_S) -> dict:
    turns = np.array([len(c.turns) for c in convs], dtype=np.int64)
    if not len(turns):
        return {"conversations": 0, "requests": 0}
    hist = Counter(int(min(t, 11)) for t in turns)
    out: dict = {
        "conversations": int(len(turns)),
        "requests": int(turns.sum()),
        "mean_turns": round(float(turns.mean()), 4),
        "max_turns": int(turns.max()),
        "multi_turn_share": round(float((turns > 1).mean()), 4),
        "turn_histogram": {(str(k) if k <= 10 else ">10"): hist[k] for k in sorted(hist)},
        "gaps_s": None,
    }
    timed = [c for c in convs if c.turns[0].timestamp is not None]
    gaps = np.array([g for c in timed for g in timestamp_gaps(c)], dtype=np.float64)
    if len(gaps):
        out["gaps_s"] = {
            "n": int(len(gaps)),
            "mean": round(float(gaps.mean()), 3),
            **{f"p{q}": round(float(np.percentile(gaps, q)), 3) for q in (10, 25, 50, 75, 90, 99)},
            "zero_share": round(float((gaps == 0).mean()), 4),
            "over_session_timeout_share": round(float((gaps > session_timeout).mean()), 4),
        }
        out["session_timeout_s"] = session_timeout
        out["requests_within_session"] = sum(session_turns(timestamp_gaps(c), session_timeout) for c in timed)
    return out


def write_splits(ds: DatasetSplits, out_dir: str) -> dict:
    splits = {}
    for name, s in ds.splits.items():
        save_conversations(s.conversations, os.path.join(out_dir, f"{name}.jsonl.gz"))
        splits[name] = {
            "positions": list(s.positions),
            "source_rows": list(s.source_rows) if s.source_rows else None,
            "candidates": sum(len(v) for v in s.dropped.values()) + len(s.conversations),
            "dropped": {reason: len(ids) for reason, ids in s.dropped.items()},
            **conversation_stats(s.conversations),
            "dropped_ids": s.dropped,
            "conv_ids": [c.conv_id for c in s.conversations],
        }
    m = {"dataset": ds.dataset, "source": ds.source, "splits": splits}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
    return m


def read_manifest(split_path: str) -> Optional[dict]:
    path = os.path.join(os.path.dirname(os.path.abspath(split_path)), "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@dataclass(frozen=True)
class SyntheticDelay:
    mu: float = 4.6
    sigma: float = 1.0


def has_timestamps(conv: Conversation) -> bool:
    return all(t.timestamp is not None for t in conv.turns)


def inter_turn_gaps(conv: Conversation, rng: np.random.Generator, synthetic_delay: SyntheticDelay) -> np.ndarray:
    if has_timestamps(conv):
        return np.maximum(np.asarray(timestamp_gaps(conv), dtype=np.float64), 0.0)
    return rng.lognormal(synthetic_delay.mu, synthetic_delay.sigma, size=len(conv.turns) - 1)


def conversation_starts(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(rng.exponential(1.0 / rate, size=n - 1))])


def build_trace(
    conversations: Iterable[Conversation],
    rate: float = DEFAULT_RATE,
    seed: int = 0,
    session_timeout: Optional[float] = DEFAULT_SESSION_TIMEOUT_S,
    synthetic_delay: SyntheticDelay = SyntheticDelay(),
    provenance: Optional[dict] = None,
) -> Trace:
    convs = list(conversations)
    if len({c.conv_id for c in convs}) != len(convs):
        raise ValueError("conversation ids must be unique")
    rng = np.random.default_rng(seed)
    starts = conversation_starts(len(convs), rate, rng)

    kept: dict[str, Conversation] = {}
    events: list[tuple[float, int, int, str]] = []
    counts: Counter = Counter()
    for order, (conv, start) in enumerate(zip(convs, starts)):
        counts["synthetic_conversations"] += not has_timestamps(conv)
        gaps = inter_turn_gaps(conv, rng, synthetic_delay)
        n = session_turns(gaps, session_timeout)
        counts["session_truncated"] += n < len(conv.turns)
        arrivals = start + np.concatenate([[0.0], np.cumsum(gaps[: n - 1])])
        kept[conv.conv_id] = Conversation(conv.conv_id, conv.source, conv.turns[:n], conv.language)
        events.extend((float(a), order, t, conv.conv_id) for t, a in enumerate(arrivals, start=1))
    events.sort()

    requests = [TraceRequest(i, conv_id, turn, arrival) for i, (arrival, _, turn, conv_id) in enumerate(events)]
    by_turn = {(r.conv_id, r.turn): r for r in requests}
    for r in requests:
        nxt = by_turn.get((r.conv_id, r.turn + 1))
        r.next_arrival = nxt.arrival if nxt else math.inf

    meta = {
        "rate": rate,
        "seed": seed,
        "session_timeout": session_timeout,
        "synthetic_delay": asdict(synthetic_delay),
        "split_conversations": len(convs),
        "conversations": len(kept),
        "requests": len(requests),
        "duration": requests[-1].arrival if requests else 0.0,
        **{k: counts[k] for k in ("synthetic_conversations", "session_truncated")},
        **(provenance or {}),
    }
    return Trace(meta, kept, requests)


def trace_summary(trace: Trace) -> dict:
    turns = Counter(r.turn for r in trace.requests)
    conv_turns = Counter(min(len(c.turns), 11) for c in trace.conversations.values())
    n = len(trace.requests)
    return {
        "conversations": len(trace.conversations),
        "requests": n,
        "duration_s": round(trace.requests[-1].arrival, 3) if n else 0.0,
        "multi_turn_request_share": round(sum(v for t, v in turns.items() if t > 1) / n, 4) if n else 0.0,
        "requests_per_turn": {str(t): turns[t] for t in sorted(turns) if t <= 10}
        | {">10": sum(v for t, v in turns.items() if t > 10)},
        "conversation_turns": {(str(k) if k <= 10 else ">10"): conv_turns[k] for k in sorted(conv_turns)},
        "mean_turns": round(n / len(trace.conversations), 4) if trace.conversations else 0.0,
    }
