from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iac.data import (
    DEFAULT_SESSION_TIMEOUT_S,
    MAX_CLASSIFIER_TURNS,
    Conversation,
    SyntheticDelay,
    file_md5,
    inter_turn_gaps,
    session_turns,
)
from iac.intents import LABELS_30, labels_for
from iac.stats import TurnRecord

logger = logging.getLogger(__name__)

QUERY_PREFIX = "query: "
DEFAULT_EMBEDDER = "intfloat/multilingual-e5-large-instruct"
BATCH_SIZE = 64
IntentsFn = Callable[[Conversation, int], str]


def special_token_wrapper(tokenizer) -> tuple[list[int], list[int]]:
    plain = tokenizer("a", add_special_tokens=False)["input_ids"]
    wrapped = tokenizer("a")["input_ids"]
    for start in range(len(wrapped) - len(plain) + 1):
        if wrapped[start : start + len(plain)] == plain:
            return wrapped[:start], wrapped[start + len(plain) :]
    raise ValueError("cannot infer the tokenizer's special-token layout")


class InputBuilder:
    def __init__(self, tokenizer, max_length: int = 512, max_turns: int = MAX_CLASSIFIER_TURNS):
        self.tokenizer = tokenizer
        self.max_length = min(max_length, tokenizer.model_max_length)
        self.max_turns = max_turns
        self.prefix_ids = tokenizer.encode(QUERY_PREFIX, add_special_tokens=False)
        self.sep_ids = tokenizer.encode("\n", add_special_tokens=False)
        self.special_head, self.special_tail = special_token_wrapper(tokenizer)
        self.pad_id = tokenizer.pad_token_id

    @classmethod
    def from_pretrained(cls, embedder: str, max_length: int, max_turns: int) -> InputBuilder:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(embedder, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(tokenizer, max_length, max_turns)

    def build(self, batch_turns: Sequence[Sequence[str]]) -> list[list[int]]:
        selected = [[(t or "").strip() for t in list(turns)[: self.max_turns]] or [""] for turns in batch_turns]
        pieces = [f"[T{i}] {t}" for turns in selected for i, t in enumerate(turns, start=1)]
        piece_ids = iter(self.tokenizer(pieces, add_special_tokens=False)["input_ids"])
        usable = max(16, self.max_length - 2 - len(self.prefix_ids))
        out = []
        for turns in selected:
            per_turn = max(16, usable // len(turns))
            head_k = per_turn // 2
            ids = list(self.prefix_ids)
            for _ in turns:
                full = next(piece_ids)
                if len(full) >= per_turn:
                    full = full[:head_k] + full[len(full) - (per_turn - head_k) :]
                ids += self.sep_ids + full
            out.append(self.special_head + ids[: self.max_length - 2] + self.special_tail)
        return out

    def pad(self, batch_ids: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(ids) for ids in batch_ids)
        input_ids = torch.full((len(batch_ids), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_ids), width), dtype=torch.long)
        for row, ids in enumerate(batch_ids):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return input_ids, attention_mask


def encode(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, autocast: bool) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        z = F.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6), p=2, dim=-1)
    return z.float()


def embedding_precision(device: torch.device) -> str:
    return "autocast-bf16" if device.type == "cuda" else "fp32"


class IntentHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden1: int = 2048, hidden2: int = 1024, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Checkpoint:
    FORMAT = ""

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.state_dict.items()}
        torch.save({"format": self.FORMAT, **vars(self), "state_dict": state}, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path, map_location="cpu")
        if ckpt.pop("format", None) != cls.FORMAT:
            raise ValueError(f"not an {cls.FORMAT} checkpoint: {path}")
        return cls(**ckpt)


@dataclass
class HeadCheckpoint(Checkpoint):
    FORMAT = "iac-intent-head/v1"

    embedder: str
    labels: list[str]
    state_dict: dict
    emb_dim: int
    hidden1: int = 2048
    hidden2: int = 1024
    dropout: float = 0.25
    max_length: int = 512
    max_turns: int = MAX_CLASSIFIER_TURNS
    metadata: dict = field(default_factory=dict)


def _replay_ms(graph, reps: int = 3) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


class IntentClassifier:
    def __init__(self, ckpt: HeadCheckpoint, device: str = "cuda", embedder_dtype: str = "autocast-bf16"):
        from transformers import AutoModel

        self.ckpt = ckpt
        self.labels = list(ckpt.labels)
        self.device = torch.device(device)
        self.embedder_dtype = embedder_dtype if self.device.type == "cuda" else "fp32"
        self.inputs = InputBuilder.from_pretrained(ckpt.embedder, ckpt.max_length, ckpt.max_turns)
        load_dtype = torch.bfloat16 if self.embedder_dtype == "bf16" else torch.float32
        self.embedder = AutoModel.from_pretrained(ckpt.embedder, dtype=load_dtype).to(self.device).eval()
        self.embedder.requires_grad_(False)
        self.head = IntentHead(ckpt.emb_dim, len(ckpt.labels), ckpt.hidden1, ckpt.hidden2, ckpt.dropout)
        self.head.load_state_dict(ckpt.state_dict, strict=True)
        self.head = self.head.to(self.device).eval()
        self._graphs: dict[tuple[int, int], tuple] = {}

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda", **kwargs) -> IntentClassifier:
        return cls(HeadCheckpoint.load(path), device=device, **kwargs)

    def _embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return encode(self.embedder, input_ids, attention_mask, self.embedder_dtype == "autocast-bf16")

    @torch.inference_mode()
    def embed_ids(self, batch_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        input_ids, attention_mask = self.inputs.pad(batch_ids)
        return self._embed(input_ids.to(self.device), attention_mask.to(self.device))

    @torch.inference_mode()
    def capture_cuda_graphs(
        self,
        batch_sizes: Sequence[int] = (1, 2, 4, 8),
        lengths: Sequence[int] = (32, 64, 128, 256, 512),
        max_tokens: int = 4096,
    ) -> bool:
        if self.device.type != "cuda":
            return False
        max_len = self.inputs.max_length
        lengths = sorted({min(n, max_len) for n in lengths} | {max_len})
        batch_sizes = sorted(set(batch_sizes) | {1})
        shapes = [(b, s) for s in lengths for b in batch_sizes if b == 1 or b * s <= max_tokens]
        shapes.sort(key=lambda bs: -bs[0] * bs[1])
        pool = torch.cuda.graph_pool_handle()
        warmup = torch.cuda.Stream(self.device)
        try:
            for b, s in shapes:
                ids = torch.full((b, s), self.inputs.pad_id, dtype=torch.long, device=self.device)
                mask = torch.zeros((b, s), dtype=torch.long, device=self.device)
                mask[:, 0] = 1
                warmup.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(warmup):
                    self.head(self._embed(ids, mask))
                torch.cuda.current_stream(self.device).wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    logits = self.head(self._embed(ids, mask))
                self._graphs[(b, s)] = (graph, ids, mask, logits)
        except torch.cuda.OutOfMemoryError:
            self._graphs.clear()
            torch.cuda.empty_cache()
            logger.warning("Too little free GPU memory for the intent classifier's CUDA graphs; running it eagerly")
            return False
        torch.cuda.empty_cache()
        self._graph_batches = batch_sizes
        self._graph_ms = {key: _replay_ms(g[0]) for key, g in self._graphs.items()}
        self._length_bucket = [next(s for s in lengths if s >= n) for n in range(max_len + 1)]
        self._batch_bucket = [0] + [next(b for b in batch_sizes if b >= k) for k in range(1, batch_sizes[-1] + 1)]
        return True

    def _plan(self, lengths: list[int]) -> list[tuple[int, int, tuple[int, int]]]:
        n, largest = len(lengths), self._graph_batches[-1]
        best = [0.0] + [math.inf] * n
        choice: list[tuple[int, tuple[int, int]]] = [(0, (0, 0))] * (n + 1)
        for j in range(1, n + 1):
            s = self._length_bucket[lengths[j - 1]]
            for i in range(max(0, j - largest), j):
                key = (self._batch_bucket[j - i], s)
                ms = self._graph_ms.get(key)
                if ms is not None and best[i] + ms < best[j]:
                    best[j], choice[j] = best[i] + ms, (i, key)
        groups = []
        while n > 0:
            i, key = choice[n]
            groups.append((i, n, key))
            n = i
        return groups[::-1]

    def _logits_graphed(self, batch_ids: list[list[int]]) -> torch.Tensor:
        order = sorted(range(len(batch_ids)), key=lambda k: len(batch_ids[k]))
        rows = [batch_ids[k] for k in order]
        out = torch.empty((len(rows), len(self.labels)), device=self.device)
        for i, j, key in self._plan([len(r) for r in rows]):
            graph, ids, mask, logits = self._graphs[key]
            input_ids, attention_mask = self.inputs.pad(rows[i:j])
            ids.fill_(self.inputs.pad_id)
            mask.zero_()
            mask[:, 0] = 1
            ids[: j - i, : input_ids.shape[1]].copy_(input_ids)
            mask[: j - i, : input_ids.shape[1]].copy_(attention_mask)
            graph.replay()
            out[i:j] = logits[: j - i]
        result = torch.empty_like(out)
        result[torch.tensor(order, device=self.device)] = out
        return result

    @torch.inference_mode()
    def logits_ids(self, batch_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        if self._graphs:
            return self._logits_graphed([list(t) for t in batch_ids])
        return self.head(self.embed_ids(batch_ids))

    @torch.inference_mode()
    def predict_ids(self, batch_turns: Sequence[Sequence[str]]) -> list[int]:
        if not batch_turns:
            return []
        return self.logits_ids(self.inputs.build(batch_turns)).argmax(dim=-1).tolist()

    def predict(self, batch_turns: Sequence[Sequence[str]]) -> list[str]:
        return [self.labels[i] for i in self.predict_ids(batch_turns)]


def load_labeled_jsonl(path: str, num_classes: int = 30) -> tuple[list[dict], list[str]]:
    labels, label_map = labels_for(num_classes)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") in LABELS_30 and isinstance(r.get("conversation"), list):
                r["label"] = label_map[r["label"]] if label_map else r["label"]
                rows.append(r)
    return rows, labels


def label_ids(rows: list[dict], labels: list[str]) -> list[int]:
    label2id = {label: i for i, label in enumerate(labels)}
    return [label2id[r["label"]] for r in rows]


def stratified_split(
    rows: list[dict], labels: list[str], val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 7
) -> tuple[list[dict], list[dict], list[dict]]:
    from sklearn.model_selection import train_test_split

    y = label_ids(rows, labels)
    train, tmp = train_test_split(rows, test_size=val_ratio + test_ratio, random_state=seed, stratify=y)
    val_frac = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(tmp, test_size=1.0 - val_frac, random_state=seed, stratify=label_ids(tmp, labels))
    return train, val, test


@dataclass
class TrainConfig:
    num_classes: int = 30
    embedder: str = DEFAULT_EMBEDDER
    max_length: int = 512
    max_turns: int = MAX_CLASSIFIER_TURNS
    epochs: int = 25
    batch_size: int = 1024
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    hidden1: int = 2048
    hidden2: int = 1024
    dropout: float = 0.25
    seed: int = 7
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    embed_batch_size: int = 64


@dataclass
class TrainResult:
    checkpoint: HeadCheckpoint
    best_epoch: int
    val: dict
    test: dict
    report: str
    history: list[dict] = field(default_factory=list)


@torch.inference_mode()
def embed_conversations(
    conversations: Sequence[Sequence[str]], cfg: TrainConfig, device: torch.device, progress: bool = False
) -> torch.Tensor:
    from tqdm import tqdm
    from transformers import AutoModel

    builder = InputBuilder.from_pretrained(cfg.embedder, cfg.max_length, cfg.max_turns)
    model = AutoModel.from_pretrained(cfg.embedder, dtype=torch.float32).to(device).eval()
    autocast = embedding_precision(device) == "autocast-bf16"
    out = torch.empty((len(conversations), model.config.hidden_size), dtype=torch.float32)
    chunk = 64 * cfg.embed_batch_size
    with tqdm(total=len(conversations), desc="embedding", disable=not progress, dynamic_ncols=True) as bar:
        for start in range(0, len(conversations), chunk):
            ids = builder.build(conversations[start : start + chunk])
            order = sorted(range(len(ids)), key=lambda i: len(ids[i]), reverse=True)
            for b in range(0, len(order), cfg.embed_batch_size):
                rows = order[b : b + cfg.embed_batch_size]
                input_ids, attention_mask = builder.pad([ids[i] for i in rows])
                z = encode(model, input_ids.to(device), attention_mask.to(device), autocast)
                out[torch.tensor(rows) + start] = z.cpu()
                bar.update(len(rows))
    return out


def cached_embeddings(
    conversations: Sequence[Sequence[str]],
    data_md5: str,
    cfg: TrainConfig,
    cache_dir: Optional[str],
    device: torch.device,
    progress: bool = False,
) -> torch.Tensor:
    key = {
        "embedder": cfg.embedder,
        "max_length": cfg.max_length,
        "max_turns": cfg.max_turns,
        "data_md5": data_md5,
        "precision": embedding_precision(device),
        "num_rows": len(conversations),
    }
    if not cache_dir:
        return embed_conversations(conversations, cfg, device, progress)
    name = f"{cfg.embedder.replace('/', '__')}-L{cfg.max_length}-T{cfg.max_turns}-{data_md5}-{key['precision']}.pt"
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
        if all(blob.get(k) == v for k, v in key.items()):
            return blob["embeddings"]
    z = embed_conversations(conversations, cfg, device, progress)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({**key, "embeddings": z}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return z


@torch.no_grad()
def predict_head(head: nn.Module, z: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
    head.eval()
    device = next(head.parameters()).device
    preds = [head(z[i : i + batch_size].to(device)).argmax(dim=-1).cpu() for i in range(0, len(z), batch_size)]
    return torch.cat(preds).numpy()


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def train_head(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    cfg: TrainConfig,
    device: str | torch.device = "cuda",
    log: Callable[[str], None] = print,
) -> tuple[dict, int, dict, list[dict]]:
    from transformers import get_cosine_schedule_with_warmup

    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    head = IntentHead(z_train.shape[1], num_classes, cfg.hidden1, cfg.hidden2, cfg.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = math.ceil(len(z_train) / cfg.batch_size) * cfg.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * cfg.warmup_ratio), total_steps)
    shuffle = torch.Generator().manual_seed(cfg.seed)
    z_train, y_train = z_train.to(device), y_train.to(device)

    best_state, best_epoch, best_val, history = None, 0, {}, []
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        loss_sum, steps = 0.0, 0
        for idx in torch.randperm(len(z_train), generator=shuffle).split(cfg.batch_size):
            idx = idx.to(device)
            loss = F.cross_entropy(head(z_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item()
            steps += 1
        val = classification_metrics(y_val.numpy(), predict_head(head, z_val))
        val_log = {f"val_{k}": v for k, v in val.items() if k != "n"}
        history.append({"epoch": epoch, "train_loss": loss_sum / steps, **val_log})
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in history[-1].items()))
        if val["macro_f1"] > best_val.get("macro_f1", -1.0):
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch, best_val = epoch, val
    return best_state, best_epoch, best_val, history


def train_classifier(
    data_path: str,
    cfg: TrainConfig,
    cache_dir: Optional[str] = None,
    device: str | torch.device = "cuda",
    progress: bool = False,
    log: Callable[[str], None] = print,
) -> TrainResult:
    from sklearn.metrics import classification_report

    device = torch.device(device)
    rows, labels = load_labeled_jsonl(data_path, num_classes=cfg.num_classes)
    data_md5 = file_md5(data_path)
    z_all = cached_embeddings([r["conversation"] for r in rows], data_md5, cfg, cache_dir, device, progress)
    for i, r in enumerate(rows):
        r["row"] = i
    splits = stratified_split(rows, labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    z_train, z_val, z_test = (z_all[[r["row"] for r in split]] for split in splits)
    y_train, y_val, y_test = (torch.tensor(label_ids(split, labels)) for split in splits)
    log(f"{len(rows)} rows ({data_md5}), {len(labels)} classes, split {len(z_train)}/{len(z_val)}/{len(z_test)}")

    state, best_epoch, val, history = train_head(z_train, y_train, z_val, y_val, len(labels), cfg, device, log)
    head = IntentHead(z_all.shape[1], len(labels), cfg.hidden1, cfg.hidden2, cfg.dropout)
    head.load_state_dict(state)
    y_pred = predict_head(head.to(device), z_test)
    test = classification_metrics(y_test.numpy(), y_pred)
    report = classification_report(
        y_test.numpy(), y_pred, labels=list(range(len(labels))), target_names=labels, digits=4, zero_division=0
    )
    metadata = {
        "data_path": os.path.realpath(data_path),
        "data_md5": data_md5,
        "num_rows": len(rows),
        "seed": cfg.seed,
        "split_sizes": {"train": len(z_train), "val": len(z_val), "test": len(z_test)},
        "embedder_precision": embedding_precision(device),
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "val": val,
        "test": test,
    }
    ckpt = HeadCheckpoint(
        embedder=cfg.embedder,
        labels=labels,
        state_dict=state,
        emb_dim=int(z_all.shape[1]),
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        dropout=cfg.dropout,
        max_length=cfg.max_length,
        max_turns=cfg.max_turns,
        metadata=metadata,
    )
    return TrainResult(ckpt, best_epoch, val, test, report, history)


def label_inputs(inputs: Sequence[Sequence[str]], classifier, batch_size: int = BATCH_SIZE) -> list[str]:
    keys = [tuple(x) for x in inputs]
    unique = sorted(dict.fromkeys(keys), key=lambda k: sum(len(s) for s in k))
    labels: dict[tuple[str, ...], str] = {}
    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        labels.update(zip(batch, classifier.predict([list(k) for k in batch])))
    return [labels[k] for k in keys]


def label_conversations(
    conversations: Iterable[Conversation], classifier, batch_size: int = BATCH_SIZE
) -> dict[str, list[str]]:
    convs = list(conversations)
    inputs = [c.classifier_turns(t) for c in convs for t in range(1, min(len(c.turns), MAX_CLASSIFIER_TURNS) + 1)]
    flat = iter(label_inputs(inputs, classifier, batch_size))
    return {c.conv_id: [next(flat) for _ in range(min(len(c.turns), MAX_CLASSIFIER_TURNS))] for c in convs}


def conversations_fingerprint(conversations: Iterable[Conversation]) -> str:
    h = hashlib.md5()
    for c in conversations:
        h.update(json.dumps([c.conv_id, c.classifier_turns(len(c.turns))], ensure_ascii=False).encode())
    return h.hexdigest()


def conversation_intents_fn(labels: dict[str, list[str]]) -> IntentsFn:
    return lambda conv, turn: labels[conv.conv_id][min(turn, MAX_CLASSIFIER_TURNS) - 1]


def intents_artifact(ckpt_path: str) -> str:
    return f"intents-{os.path.splitext(os.path.basename(ckpt_path))[0]}.json"


def split_intents_path(split_path: str, ckpt_path: str) -> str:
    stem = os.path.basename(split_path).split(".")[0]
    return os.path.join(os.path.dirname(os.path.abspath(split_path)), f"{stem}.{intents_artifact(ckpt_path)}")


def _cached_labels(path: str, ckpt_path: str, embedder_dtype: str, fingerprint: str, compute: Callable, *args):
    key = {"md5": file_md5(ckpt_path), "embedder_dtype": embedder_dtype, "fingerprint": fingerprint}
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if all(cached.get(k) == v for k, v in key.items()):
            return cached["labels"]
    labels = compute(*args)
    with open(path, "w") as f:
        json.dump({"labels": labels, "ckpt": ckpt_path, **key}, f)
    return labels


def label_split(
    convs: list[Conversation],
    split_path: str,
    ckpt_path: str,
    classifier,
    embedder_dtype: str = "bf16",
    batch_size: int = BATCH_SIZE,
) -> dict[str, list[str]]:
    path, fingerprint = split_intents_path(split_path, ckpt_path), conversations_fingerprint(convs)
    return _cached_labels(
        path, ckpt_path, embedder_dtype, fingerprint, label_conversations, convs, classifier, batch_size
    )


def turn_records(
    conversations: Iterable[Conversation],
    intents_fn: IntentsFn,
    session_timeout: Optional[float] = DEFAULT_SESSION_TIMEOUT_S,
    synthetic_delay: SyntheticDelay = SyntheticDelay(),
    seed: int = 0,
) -> list[TurnRecord]:
    rng = np.random.default_rng(seed)
    records = []
    for conv in conversations:
        gaps = inter_turn_gaps(conv, rng, synthetic_delay)
        times = np.concatenate([[0.0], np.cumsum(gaps)])
        for t in range(1, session_turns(gaps, session_timeout) + 1):
            records.append(TurnRecord(conv.conv_id, t, intents_fn(conv, t), float(times[t - 1])))
    return records
