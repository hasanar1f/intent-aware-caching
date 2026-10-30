from __future__ import annotations

import hashlib
import logging
import math
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

from iac.classifier import IntentClassifier
from iac.data import DEFAULT_SESSION_TIMEOUT_S, MAX_CLASSIFIER_TURNS
from iac.intents import UNKNOWN_INTENT
from iac.policy import DEFAULT_HORIZON_S, DEFAULT_SHRINKAGE, DEFAULT_UPDATE_INTERVAL_S, DEFAULT_WINDOW, IACPolicy
from iac.stats import OnlineReuseStats, OnlineSnapshot, ReuseStats

logger = logging.getLogger(__name__)

TURN = "iac_turn"
USER_TURNS = "iac_user_turns"
TIMESTAMP = "iac_timestamp"
ARRIVAL = "iac_arrival"
CONV_ID = "iac_conv_id"
KEY = "iac_key"
PARENT_KEY = "iac_parent_key"

MAX_PENDING = 1_000_000
SUMMARY_EVERY = 5000
MAX_WAIT_S = 0.25
MAX_DEFER_S = 1.0
IDLE_POLL_S = 0.002
CACHE_SIZE = 200_000
_RUNTIME: Optional[IACRuntime] = None


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text") or "")
        elif getattr(part, "type", None) == "text":
            parts.append(getattr(part, "text", "") or "")
    return "\n".join(parts)


def conversation_key(messages: Iterable[tuple[str, str]]) -> str:
    text = "\x1e".join(f"{role}\x1f{content}" for role, content in messages)
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:20]


def chat_metadata(messages: Iterable[Any]) -> dict:
    msgs = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        msgs.append((str(role), message_text(content)))
    users = [i for i, (role, _) in enumerate(msgs) if role == "user"]
    meta = {TURN: max(1, len(users)), USER_TURNS: [msgs[i][1] for i in users[:MAX_CLASSIFIER_TURNS]]}
    if users:
        meta[KEY] = conversation_key(msgs[: users[-1] + 1])
        meta[PARENT_KEY] = conversation_key(msgs[: users[-2] + 1]) if len(users) > 1 else ""
    return meta


def attach_chat_metadata(sampling_params: dict, messages: Iterable[Any]) -> dict:
    custom = dict(sampling_params.get("custom_params") or {})
    for k, v in chat_metadata(messages).items():
        custom.setdefault(k, v)
    return {**sampling_params, "custom_params": custom}


def stamp_arrival(sampling_kwargs: dict, now: float) -> dict:
    return {**sampling_kwargs, "custom_params": {**(sampling_kwargs.get("custom_params") or {}), ARRIVAL: now}}


@dataclass
class IACLabelUpdate:
    labels: list[tuple[str, str]] = field(default_factory=list)


def custom_params_of(req: Any) -> dict:
    cp = getattr(getattr(req, "sampling_params", None), "custom_params", None)
    return cp if isinstance(cp, dict) else {}


def optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def parse_turn(value: Any) -> int:
    x = optional_float(value)
    return max(1, int(x)) if x is not None else 1


def parse_user_turns(value: Any) -> Optional[list[str]]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value:
        return [t if isinstance(t, str) else ("" if t is None else str(t)) for t in value]
    return None


@dataclass(slots=True, eq=False)
class IACTag:
    intent: str
    turn: int
    timestamp: float


class IACRuntime:
    def __init__(
        self,
        clock: str = "wall",
        stats_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
        classifier_device: Optional[str] = None,
        classifier_dtype: str = "bf16",
        classifier_max_batch: int = 64,
        horizon: float = DEFAULT_HORIZON_S,
        shrinkage: float = DEFAULT_SHRINKAGE,
        window: int = DEFAULT_WINDOW,
        update_interval: float = DEFAULT_UPDATE_INTERVAL_S,
        session_timeout: float = DEFAULT_SESSION_TIMEOUT_S,
        is_leader: bool = True,
        idle_fn: Optional[Callable[[], bool]] = None,
    ):
        if clock not in ("wall", "request"):
            raise ValueError(f"--iac-clock must be 'wall' or 'request', got {clock!r}")
        self.clock = clock
        self.now = 0.0
        self.is_leader = is_leader
        self.pending: dict[str, IACTag] = {}
        self.use_classifier = bool(classifier_path)
        self.worker = None
        self.num_requests = self.num_labels = 0
        self._warned_clock = False
        prior = ReuseStats.load(stats_path) if stats_path else None
        self.stats = OnlineReuseStats(window, update_interval, session_timeout, shrinkage, prior)
        self.iac = IACPolicy(self.stats, horizon=horizon)
        if self.use_classifier and is_leader:
            self.worker = build_worker(
                classifier_path, classifier_device, classifier_dtype, classifier_max_batch, idle_fn
            )
        logger.info(
            "IAC: prior=%s window=%d update_interval=%gs horizon=%gs shrinkage=%g classifier=%s clock=%s",
            stats_path,
            window,
            update_interval,
            horizon,
            shrinkage,
            classifier_path,
            clock,
        )

    def on_new_request(self, req: Any) -> None:
        if getattr(req, "iac_tag", None) is not None:
            return
        cp = custom_params_of(req)
        ts = optional_float(cp.get(TIMESTAMP if self.clock == "request" else ARRIVAL))
        if ts is None:
            if self.clock == "request" and not self._warned_clock:
                self._warned_clock = True
                logger.warning(
                    "IAC: --iac-clock request, but requests carry no iac_timestamp: the clock does not advance"
                )
            ts = self.now
        self.now = max(self.now, ts)
        tag = req.iac_tag = IACTag(UNKNOWN_INTENT, parse_turn(cp.get(TURN)), ts)
        self.num_requests += 1
        if self.num_requests % SUMMARY_EVERY == 0:
            logger.info("IAC runtime: %s", self.summary())
        if self.use_classifier:
            if len(self.pending) < MAX_PENDING:
                self.pending[req.rid] = tag
            if self.worker is not None:
                turns = parse_user_turns(cp.get(USER_TURNS)) or [getattr(req, "origin_input_text", None) or ""]
                self.worker.submit(req.rid, turns)
        version = self.stats.snapshot.version
        key, parent = self._conversation_keys(cp, tag.turn)
        self.stats.observe(key, parent, tag, self.now)
        if self.is_leader and self.stats.snapshot.version != version:
            logger.info("IAC statistics updated: %s", snapshot_view(self.stats.snapshot))

    @staticmethod
    def _conversation_keys(cp: dict, turn: int) -> tuple[str, str]:
        conv = cp.get(CONV_ID)
        if conv is not None:
            return f"{conv}#{turn}", (f"{conv}#{turn - 1}" if turn > 1 else "")
        key, parent = cp.get(KEY), cp.get(PARENT_KEY)
        return (str(key) if key else ""), (str(parent) if parent else "")

    def drain_label_messages(self) -> list:
        if self.worker is None:
            return []
        labels = self.worker.drain()
        return [IACLabelUpdate(labels=labels)] if labels else []

    def apply_labels(self, msg: IACLabelUpdate) -> None:
        for rid, intent in msg.labels:
            tag = self.pending.pop(rid, None)
            if tag is not None:
                tag.intent = intent
                self.num_labels += 1

    def priority(self, node: Any) -> tuple:
        tag = node.iac_tag
        if tag is None:
            return (0.0, len(node.key))
        return self.iac.priority(tag.intent, tag.turn, self.now - node.iac_time, len(node.key))

    def summary(self) -> dict:
        out = {
            "clock": self.clock,
            "now": self.now,
            "requests": self.num_requests,
            "labels_applied": self.num_labels,
            "pending": len(self.pending),
        }
        if self.worker is not None:
            out["classifier"] = self.worker.stats()
        out["stats"] = {**self.stats.summary(), "snapshot": snapshot_view(self.stats.snapshot)}
        return out


def snapshot_view(snapshot: OnlineSnapshot) -> dict:
    view: dict[str, Any] = {"version": snapshot.version, "states": len(snapshot.states)}
    for t in (1, 2, 3):
        acc = snapshot.turns.get(t)
        if acc is not None and acc[0] > 0:
            n, n_next, n_delay, sum_log_delay, _ = acc
            median = round(math.exp(sum_log_delay / n_delay), 1) if n_delay > 0 else None
            view[f"t{t}"] = {"P": round(float(n_next / n), 3), "M": median, "n": round(float(n), 1)}
    return view


def check_tree_cache(tree_cache: Any) -> None:
    name = type(tree_cache).__name__
    if name != "UnifiedRadixCache":
        raise ValueError(
            f"IAC eviction policies need SGLang's UnifiedRadixCache, got {name} (hybrid SWA/Mamba "
            "models, --disable-radix-cache, LMCache/FlexKV and the C++ tree are not supported)"
        )
    backend = getattr(tree_cache, "_tree_core_backend", "python")
    if backend != "python":
        raise ValueError(
            f"IAC needs the Python unified tree core, got SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND={backend!r}"
        )
    linker = getattr(tree_cache, "linker", None)
    if getattr(tree_cache, "host_memory_mode", None) == "buffer_only" or linker is not None:
        raise ValueError("IAC does not support HiCache buffer_only mode or external cache linkers")


def scheduler_idle_fn(scheduler: Any) -> Callable[[], bool]:
    def idle() -> bool:
        try:
            running = scheduler.running_batch
            busy = scheduler.last_batch is not None or scheduler.waiting_queue
            return not busy and (running is None or running.is_empty())
        except AttributeError:
            return True

    return idle


def init_runtime(scheduler: Any) -> IACRuntime:
    global _RUNTIME
    from sglang.srt.environ import envs

    if envs.SGLANG_RUST_SERVER.get():
        raise ValueError("IAC eviction policies are not supported with SGLANG_RUST_SERVER=1")
    check_tree_cache(scheduler.tree_cache)
    args, ps = scheduler.server_args, scheduler.ps
    _RUNTIME = IACRuntime(
        clock=args.iac_clock,
        stats_path=args.iac_stats_path,
        classifier_path=args.iac_classifier_path,
        classifier_device=args.iac_classifier_device,
        classifier_dtype=args.iac_classifier_dtype,
        classifier_max_batch=args.iac_classifier_max_batch,
        horizon=args.iac_horizon,
        shrinkage=args.iac_shrinkage,
        window=args.iac_window,
        update_interval=args.iac_update_interval,
        is_leader=ps.pp_rank == 0 and ps.attn_tp_rank == 0 and ps.attn_cp_rank == 0,
        idle_fn=scheduler_idle_fn(scheduler),
    )
    return _RUNTIME


def get_runtime() -> IACRuntime:
    if _RUNTIME is None:
        raise RuntimeError("IAC runtime is not initialized in this process")
    return _RUNTIME


def label_messages() -> list:
    return _RUNTIME.drain_label_messages() if _RUNTIME is not None else []


def eviction_strategy():
    from sglang.srt.mem_cache.evict_policy import EvictionStrategy

    class IACStrategy(EvictionStrategy):
        def get_priority(self, node):
            return get_runtime().priority(node)

    return IACStrategy()


class ClassifierWorker:
    def __init__(
        self,
        classifier,
        max_batch: int = 64,
        max_wait_s: float = MAX_WAIT_S,
        idle_fn: Optional[Callable[[], bool]] = None,
        max_defer_s: float = MAX_DEFER_S,
    ):
        self.classifier = classifier
        self.max_batch = max_batch
        self.max_wait_s = max_wait_s
        self.idle_fn = idle_fn
        self.max_defer_s = max_defer_s
        self._jobs: queue.SimpleQueue[tuple[str, tuple[str, ...]]] = queue.SimpleQueue()
        self._done: deque[tuple[str, str]] = deque()
        self._cache: OrderedDict[tuple[str, ...], str] = OrderedDict()
        self._stop = threading.Event()
        self._stream = torch.cuda.Stream(device=classifier.device) if classifier.device.type == "cuda" else None
        self.num_requests = self.num_inferences = self.num_batches = self.num_forced = 0
        self.busy_seconds = self.deferred_seconds = 0.0
        self._thread = threading.Thread(target=self._run, name="iac-intent-classifier", daemon=True)
        self._thread.start()

    def submit(self, rid: str, user_turns: Sequence[str]) -> None:
        key = tuple((t or "").strip() for t in user_turns[: self.classifier.inputs.max_turns]) or ("",)
        self._jobs.put((rid, key))

    def drain(self) -> list[tuple[str, str]]:
        out = []
        while self._done:
            out.append(self._done.popleft())
        return out

    def _collect(self) -> tuple[list[tuple[str, tuple[str, ...]]], float]:
        jobs = [self._jobs.get()]
        first = time.perf_counter()
        deadline = first + self.max_wait_s
        while len(jobs) < self.max_batch * 4:
            wait = deadline - time.perf_counter()
            try:
                block = wait > 0 and len(jobs) < self.max_batch
                jobs.append(self._jobs.get(timeout=wait) if block else self._jobs.get_nowait())
            except queue.Empty:
                break
        return jobs, first

    def _wait_idle(self, deadline: float) -> None:
        t0 = time.perf_counter()
        while not self.idle_fn():
            if time.perf_counter() >= deadline or self._stop.is_set():
                self.num_forced += 1
                break
            time.sleep(IDLE_POLL_S)
        self.deferred_seconds += time.perf_counter() - t0

    def _run(self) -> None:
        if self._stream is not None:
            torch.cuda.set_device(self.classifier.device)
        while not self._stop.is_set():
            jobs, first = self._collect()
            if self._stop.is_set():
                break
            misses = list(dict.fromkeys(k for _, k in jobs if k not in self._cache))
            try:
                for i in range(0, len(misses), self.max_batch):
                    chunk = misses[i : i + self.max_batch]
                    if self.idle_fn is not None:
                        self._wait_idle(first + self.max_defer_s)
                    t0 = time.perf_counter()
                    if self._stream is not None:
                        with torch.cuda.stream(self._stream):
                            labels = self.classifier.predict(chunk)
                    else:
                        labels = self.classifier.predict(chunk)
                    self._cache.update(zip(chunk, labels))
                    self.num_inferences += len(chunk)
                    self.num_batches += 1
                    self.busy_seconds += time.perf_counter() - t0
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
            except Exception:
                logger.exception("IAC intent classification failed for %d requests", len(jobs))
            for rid, key in jobs:
                label = self._cache.get(key)
                if label is None:
                    label = UNKNOWN_INTENT
                else:
                    self._cache.move_to_end(key)
                self._done.append((rid, label))
            self.num_requests += len(jobs)

    def stats(self) -> dict:
        return {
            "requests": self.num_requests,
            "inferences": self.num_inferences,
            "batches": self.num_batches,
            "busy_seconds": round(self.busy_seconds, 3),
            "deferred_seconds": round(self.deferred_seconds, 3),
            "forced": self.num_forced,
            "cache_entries": len(self._cache),
        }


def build_worker(
    ckpt_path: str, device: Optional[str], dtype: str, max_batch: int, idle_fn: Optional[Callable[[], bool]] = None
) -> ClassifierWorker:
    t0 = time.perf_counter()
    device = device or f"cuda:{torch.cuda.current_device()}"
    clf = IntentClassifier.from_checkpoint(ckpt_path, device=device, embedder_dtype=dtype)
    clf.predict([["warm up the classifier"]])
    clf.capture_cuda_graphs()
    logger.info(
        "IAC intent classifier %s (%s, %s) loaded in %.1fs",
        ckpt_path,
        clf.ckpt.embedder,
        dtype,
        time.perf_counter() - t0,
    )
    return ClassifierWorker(clf, max_batch=max_batch, idle_fn=idle_fn)
