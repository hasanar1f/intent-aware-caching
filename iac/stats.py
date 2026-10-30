from __future__ import annotations

import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
from scipy.special import ndtr

from iac.data import DEFAULT_SESSION_TIMEOUT_S
from iac.intents import UNKNOWN_INTENT
from iac.policy import DEFAULT_SHRINKAGE, DEFAULT_UPDATE_INTERVAL_S, DEFAULT_WINDOW

SCHEMA = "iac-reuse-stats/v1"
DEFAULT_P_NEXT = 0.5
DEFAULT_REUSE_WINDOW_S = 150.0
DEFAULT_SIGMA = 1.3
MAX_OPEN = 1 << 20
MAX_WINDOW = 1 << 21
_MIN_SIGMA = 0.05
_MAX_MEMO = 1 << 16
_N, _ND = 0, 2
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class StateStats:
    p_next: float
    reuse_window: float
    n: int
    n_next: int
    mu: Optional[float] = None
    sigma: Optional[float] = None
    n_delay: int = 0


@dataclass(frozen=True)
class TurnRecord:
    conv_id: str
    turn: int
    intent: str
    timestamp: float


def _fit_states(counts: Mapping, log_delays: Mapping) -> dict:
    table = {}
    for state, (n, n_next) in counts.items():
        logs = log_delays.get(state, [])
        mu = sigma = None
        window = DEFAULT_REUSE_WINDOW_S
        if logs:
            mu = sum(logs) / len(logs)
            sigma = math.sqrt(sum((x - mu) ** 2 for x in logs) / len(logs))
            window = math.exp(mu)
        table[state] = StateStats(n_next / n, window, n, n_next, mu, sigma, len(logs))
    return table


def _mu(s: StateStats) -> float:
    return s.mu if s.mu is not None else math.log(s.reuse_window)


class ReuseStats:
    def __init__(
        self,
        table: Mapping[tuple[str, int], StateStats],
        default_p_next: float = DEFAULT_P_NEXT,
        default_reuse_window: float = DEFAULT_REUSE_WINDOW_S,
        metadata: Optional[dict] = None,
        turn_table: Optional[Mapping[int, StateStats]] = None,
        shrinkage: float = 0.0,
    ):
        self.table = dict(table)
        self.turn_table = dict(turn_table or {})
        self.default_p_next = default_p_next
        self.default_reuse_window = default_reuse_window
        self.metadata = dict(metadata or {})
        self.shrinkage = shrinkage
        self._resolved: dict[tuple[str, int], tuple[float, float, float]] = {}

    def with_shrinkage(self, k: float) -> ReuseStats:
        if k > 0 and not self.turn_table:
            raise ValueError("shrinkage needs turn-level statistics")
        return ReuseStats(self.table, self.default_p_next, self.default_reuse_window, self.metadata, self.turn_table, k)

    def lookup_dist(self, intent: str, turn: int) -> tuple[float, float, float]:
        key = (intent, turn)
        out = self._resolved.get(key)
        if out is None:
            if len(self._resolved) >= _MAX_MEMO:
                self._resolved.clear()
            out = self._resolved[key] = self._resolve(intent, turn)
        return out

    def _resolve(self, intent: str, turn: int) -> tuple[float, float, float]:
        s = self.table.get((intent, turn))
        base = self.turn_table.get(turn) if self.shrinkage > 0 else None
        if base is None:
            if s is None:
                return self.default_p_next, math.log(self.default_reuse_window), DEFAULT_SIGMA
            return s.p_next, _mu(s), s.sigma or DEFAULT_SIGMA
        k = self.shrinkage
        n, n_next = (s.n, s.n_next) if s is not None else (0, 0)
        p = (n_next + k * base.p_next) / (n + k)
        mu_b, sg_b = _mu(base), base.sigma or DEFAULT_SIGMA
        if s is None or s.mu is None:
            return p, mu_b, sg_b
        sg = s.sigma or DEFAULT_SIGMA
        mu = (s.n_delay * s.mu + k * mu_b) / (s.n_delay + k)
        sigma = math.sqrt((s.n_delay * sg**2 + k * sg_b**2) / (s.n_delay + k))
        return p, mu, sigma

    def __len__(self) -> int:
        return len(self.table)

    @classmethod
    def learn(
        cls, records: Iterable[TurnRecord], session_timeout: Optional[float] = None, metadata: Optional[dict] = None
    ) -> ReuseStats:
        by_conv: dict[str, dict[int, TurnRecord]] = defaultdict(dict)
        for r in records:
            by_conv[r.conv_id][int(r.turn)] = r

        counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
        log_delays: dict[tuple[str, int], list[float]] = defaultdict(list)
        turn_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        turn_log_delays: dict[int, list[float]] = defaultdict(list)
        for turns in by_conv.values():
            for t, r in turns.items():
                state = (r.intent, t)
                counts[state][0] += 1
                turn_counts[t][0] += 1
                nxt = turns.get(t + 1)
                if nxt is None:
                    continue
                delay = nxt.timestamp - r.timestamp
                if session_timeout is not None and delay > session_timeout:
                    continue
                counts[state][1] += 1
                turn_counts[t][1] += 1
                if delay > 0:
                    log_delays[state].append(math.log(delay))
                    turn_log_delays[t].append(math.log(delay))
        return cls(
            _fit_states(counts, log_delays), metadata=metadata, turn_table=_fit_states(turn_counts, turn_log_delays)
        )

    def save(self, path: str) -> None:
        states: dict[str, dict[str, dict]] = defaultdict(dict)
        for (intent, turn), s in sorted(self.table.items()):
            states[intent][str(turn)] = asdict(s)
        out = {
            "schema": SCHEMA,
            "defaults": {"p_next": self.default_p_next, "reuse_window": self.default_reuse_window},
            "metadata": self.metadata,
            "states": states,
            "turn_states": {str(t): asdict(s) for t, s in sorted(self.turn_table.items())},
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=1)

    @classmethod
    def load(cls, path: str) -> ReuseStats:
        with open(path) as f:
            d = json.load(f)
        if d.get("schema") != SCHEMA:
            raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {d.get('schema')!r}")
        table = {(i, int(t)): StateStats(**s) for i, turns in d["states"].items() for t, s in turns.items()}
        turn_table = {int(t): StateStats(**s) for t, s in d["turn_states"].items()}
        return cls(table, d["defaults"]["p_next"], d["defaults"]["reuse_window"], d["metadata"], turn_table)


def _shrink(acc: Optional[np.ndarray], base: tuple[float, float, float], k: float) -> tuple[float, float, float]:
    p_b, mu_b, sg_b = base
    if acc is None or acc[_N] <= 0:
        return base
    n, n_next, n_d, s1, s2 = (float(x) for x in acc)
    p = min(max((n_next + k * p_b) / (n + k), 0.0), 1.0)
    if n_d <= 0:
        return p, mu_b, sg_b
    mu_s = s1 / n_d
    var_s = s2 / n_d - mu_s * mu_s
    if n_d < 2 or var_s <= 0:
        var_s = sg_b * sg_b
    mu = (n_d * mu_s + k * mu_b) / (n_d + k)
    sigma = math.sqrt((n_d * var_s + k * sg_b * sg_b) / (n_d + k))
    return p, mu, max(sigma, _MIN_SIGMA)


def _untruncate(m: np.ndarray, s: np.ndarray, upper: float, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    mu0, s0 = m.copy(), s.copy()
    for _ in range(iters):
        beta = np.maximum((upper - mu0) / s0, -1.0)
        lam = _INV_SQRT_2PI * np.exp(-0.5 * beta * beta) / ndtr(beta)
        s0 = s / np.sqrt(np.maximum(1.0 - beta * lam - lam * lam, 1e-6))
        mu0 = m + s0 * lam
    return mu0, s0


class OnlineSnapshot:
    def __init__(self, states: dict, turns: dict, prior: Optional[ReuseStats], shrinkage: float, version: int):
        self.states = states
        self.turns = turns
        self.prior = prior
        self.shrinkage = shrinkage
        self.version = version
        self._memo: dict[tuple[str, int], tuple[float, float, float]] = {}

    def _base(self, intent: str, turn: int) -> tuple[float, float, float]:
        if self.prior is not None:
            return self.prior.lookup_dist(intent, turn)
        default = (DEFAULT_P_NEXT, math.log(DEFAULT_REUSE_WINDOW_S), DEFAULT_SIGMA)
        return _shrink(self.turns.get(turn), default, self.shrinkage)

    def lookup_dist(self, intent: str, turn: int) -> tuple[float, float, float]:
        key = (intent, turn)
        out = self._memo.get(key)
        if out is None:
            if len(self._memo) >= _MAX_MEMO:
                self._memo.clear()
            out = self._memo[key] = _shrink(self.states.get(key), self._base(intent, turn), self.shrinkage)
        return out


class _Record:
    __slots__ = ("tag", "arrival", "state", "continued", "log_delay", "slot")

    def __init__(self, tag: Any):
        self.tag = tag
        self.arrival = tag.timestamp
        self.state: Optional[tuple[str, int]] = None
        self.continued: Optional[bool] = None
        self.log_delay = math.nan
        self.slot = -1


class OnlineReuseStats:
    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        update_interval: float = DEFAULT_UPDATE_INTERVAL_S,
        session_timeout: float = DEFAULT_SESSION_TIMEOUT_S,
        shrinkage: float = DEFAULT_SHRINKAGE,
        prior: Optional[ReuseStats] = None,
    ):
        if window <= 0 or update_interval <= 0 or session_timeout <= 0:
            raise ValueError("window, update_interval and session_timeout must be positive")
        if prior is not None and shrinkage > 0 and prior.turn_table:
            prior = prior.with_shrinkage(shrinkage)
        self.window = window
        self.update_interval = update_interval
        self.session_timeout = session_timeout
        self.shrinkage = shrinkage
        self.prior = prior
        self._open: OrderedDict[str, _Record] = OrderedDict()
        cap = min(window, 1 << 16)
        self._ring: list[Optional[_Record]] = [None] * cap
        self._ring_arrival = np.zeros(cap)
        self._ring_state = np.full(cap, -1, dtype=np.int64)
        self._ring_waiting = np.zeros(cap, dtype=bool)
        self._head = 0
        self._size = 0
        self._state_ids: dict[tuple[str, int], int] = {}
        self._state_keys: list[tuple[str, int]] = []
        self._unknown_ids: set[int] = set()
        self._states: dict[tuple[str, int], np.ndarray] = {}
        self._turns: dict[int, np.ndarray] = {}
        self._next_update: Optional[float] = None
        self.num_completed = 0
        self.snapshot = OnlineSnapshot({}, {}, prior, shrinkage, version=0)

    def observe(self, key: Optional[str], parent: Optional[str], tag: Any, now: float) -> None:
        if parent:
            prev = self._open.pop(parent, None)
            if prev is not None:
                delay = tag.timestamp - prev.arrival
                self._complete(prev, delay if delay <= self.session_timeout else None)
        self._expire(now)
        if key:
            repeated = self._open.pop(key, None)
            if repeated is not None:
                self._drop(repeated)
            rec = _Record(tag)
            self._open[key] = rec
            self._enter_window(rec, now)
            if len(self._open) > MAX_OPEN:
                self._drop(self._open.popitem(last=False)[1])
        if self._next_update is None:
            self._next_update = now + self.update_interval
        elif now >= self._next_update:
            self.update(now)

    def _expire(self, now: float) -> None:
        while self._open:
            rec = next(iter(self._open.values()))
            if now - rec.arrival <= self.session_timeout:
                break
            self._open.popitem(last=False)
            self._complete(rec, None)

    def _complete(self, rec: _Record, delay: Optional[float]) -> None:
        rec.state = (rec.tag.intent, rec.tag.turn)
        rec.continued = delay is not None
        if rec.continued and delay > 0:
            rec.log_delay = math.log(delay)
        self.num_completed += 1
        if rec.slot >= 0:
            self._ring_waiting[rec.slot] = False
            self._add(rec, 1.0)

    def _drop(self, rec: _Record) -> None:
        if rec.slot >= 0:
            self._ring[rec.slot] = None
            self._ring_waiting[rec.slot] = False
            rec.slot = -1

    def _enter_window(self, rec: _Record, now: float) -> None:
        while self._size >= self.window and now - self._ring_arrival[self._head] > self.session_timeout:
            self._evict_oldest()
        if self._size == len(self._ring):
            if len(self._ring) >= MAX_WINDOW:
                self._evict_oldest()
            else:
                self._grow()
        slot = (self._head + self._size) % len(self._ring)
        self._size += 1
        self._ring[slot] = rec
        rec.slot = slot
        self._ring_arrival[slot] = rec.arrival
        self._ring_state[slot] = self._state_id((rec.tag.intent, rec.tag.turn))
        self._ring_waiting[slot] = rec.continued is None

    def _evict_oldest(self) -> None:
        old = self._ring[self._head]
        if old is not None:
            old.slot = -1
            if old.continued is not None:
                self._add(old, -1.0)
        self._ring[self._head] = None
        self._ring_waiting[self._head] = False
        self._head = (self._head + 1) % len(self._ring)
        self._size -= 1

    def _grow(self) -> None:
        cap = len(self._ring)
        new_cap = min(2 * cap, MAX_WINDOW)
        order = [(self._head + j) % cap for j in range(self._size)]
        ring: list[Optional[_Record]] = [self._ring[i] for i in order] + [None] * (new_cap - self._size)
        for j, rec in enumerate(ring[: self._size]):
            if rec is not None:
                rec.slot = j
        pad = new_cap - self._size
        self._ring = ring
        self._ring_arrival = np.concatenate([self._ring_arrival[order], np.zeros(pad)])
        self._ring_state = np.concatenate([self._ring_state[order], np.full(pad, -1, dtype=np.int64)])
        self._ring_waiting = np.concatenate([self._ring_waiting[order], np.zeros(pad, dtype=bool)])
        self._head = 0

    def _state_id(self, state: tuple[str, int]) -> int:
        sid = self._state_ids.get(state)
        if sid is None:
            sid = self._state_ids[state] = len(self._state_keys)
            self._state_keys.append(state)
            if state[0] == UNKNOWN_INTENT:
                self._unknown_ids.add(sid)
        return sid

    def _compact_state_ids(self) -> None:
        keep = np.unique(self._ring_state[self._ring_waiting])
        remap = np.full(len(self._state_keys), -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        self._state_keys = [self._state_keys[i] for i in keep]
        self._state_ids = {s: i for i, s in enumerate(self._state_keys)}
        self._unknown_ids = {i for i, s in enumerate(self._state_keys) if s[0] == UNKNOWN_INTENT}
        self._ring_state = np.where(self._ring_state >= 0, remap[np.maximum(self._ring_state, 0)], -1)

    def _add(self, rec: _Record, sign: float) -> None:
        row = np.array([1.0, float(rec.continued), 0.0, 0.0, 0.0])
        if rec.continued and not math.isnan(rec.log_delay):
            row[_ND:] = (1.0, rec.log_delay, rec.log_delay * rec.log_delay)
        for table, key in ((self._states, rec.state), (self._turns, rec.state[1])):
            acc = table.get(key)
            if acc is None:
                acc = table[key] = np.zeros(5)
            acc += sign * row
            if acc[_N] < 0.5:
                del table[key]

    def update(self, now: float) -> None:
        self._expire(now)
        states = {k: v.copy() for k, v in self._states.items()}
        turns = {k: v.copy() for k, v in self._turns.items()}
        if len(self._state_keys) > 4 * len(self._states) + 4096:
            self._compact_state_ids()
        slots = np.flatnonzero(self._ring_waiting)
        if len(slots):
            self._add_waiting(states, turns, slots, now)
        self.snapshot = OnlineSnapshot(states, turns, self.prior, self.shrinkage, self.snapshot.version + 1)
        self._next_update = now + self.update_interval

    def _add_waiting(self, states: dict, turns: dict, slots: np.ndarray, now: float) -> None:
        sid = self._ring_state[slots]
        if self._unknown_ids:
            for j in np.flatnonzero(np.isin(sid, np.fromiter(self._unknown_ids, np.int64))):
                tag = self._ring[slots[j]].tag
                sid[j] = self._ring_state[slots[j]] = self._state_id((tag.intent, tag.turn))
        uniq, idx = np.unique(sid, return_inverse=True)
        keys = [self._state_keys[u] for u in uniq]
        dist = np.array([self.snapshot.lookup_dist(*k) for k in keys])
        upper = math.log(self.session_timeout)
        mu0, s0 = _untruncate(dist[:, 1], dist[:, 2], upper)
        p, mu, sigma = dist[idx, 0], mu0[idx], s0[idx]
        a = (np.log(np.maximum(now - self._ring_arrival[slots], 1e-3)) - mu) / sigma
        b = (upper - mu) / sigma
        cdf_a, cdf_b = ndtr(a), ndtr(b)
        mass = np.maximum(cdf_b - cdf_a, 0.0)
        survive = np.divide(mass, cdf_b, out=np.zeros_like(mass), where=cdf_b > 0)
        den = (1.0 - p) + p * survive
        w = np.divide(p * survive, den, out=np.zeros_like(den), where=den > 0)
        pdf_a = _INV_SQRT_2PI * np.exp(-0.5 * a * a)
        pdf_b = _INV_SQRT_2PI * np.exp(-0.5 * b * b)
        ok = mass > 1e-12
        safe = np.where(ok, mass, 1.0)
        zmean = np.where(ok, (pdf_a - pdf_b) / safe, a)
        zvar = np.where(ok, 1.0 + (a * pdf_a - b * pdf_b) / safe - zmean * zmean, 0.0)
        m1 = mu + sigma * zmean
        m2 = sigma * sigma * np.maximum(zvar, 0.0) + m1 * m1
        cols = (np.ones_like(w), w, w, w * m1, w * m2)
        by_state = np.stack([np.bincount(idx, weights=c, minlength=len(keys)) for c in cols], axis=1)
        for j, key in enumerate(keys):
            for table, k in ((states, key), (turns, key[1])):
                acc = table.get(k)
                if acc is None:
                    table[k] = by_state[j].copy()
                else:
                    acc += by_state[j]

    def lookup_dist(self, intent: str, turn: int) -> tuple[float, float, float]:
        return self.snapshot.lookup_dist(intent, turn)

    def summary(self) -> dict:
        return {
            "open": len(self._open),
            "window_records": self._size,
            "states": len(self._states),
            "completed": self.num_completed,
            "updates": self.snapshot.version,
        }
