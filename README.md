# Intent-Aware Caching for Efficient LLM Serving

Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S. Nikolopoulos, Hans Vandierendonck, Deepu John, Bo Ji (NeurIPS 2026)

## Abstract

Modern LLM serving systems use prefix caching to accelerate multi-turn conversational workloads. They cache the
key-value states of previous conversation prefixes and reuse them when subsequent requests share the same prefix,
to avoid redundant prefill computation and reduce serving latency. However, achieving a high cache hit rate remains
challenging because cache reuse depends on whether a conversation continues and how soon the next turn arrives. Our
analysis of real workloads shows that user intent is a strong signal for this behavior, as different intents exhibit
different continuation probabilities and inter-turn delays. Based on this insight, we propose Intent-Aware Caching
(IAC), an online eviction policy that classifies requests by intent, learns lightweight reuse statistics from recent
serving logs, and uses them to guide an oracle-inspired eviction decision. Extensive evaluation on real workloads
(e.g., ShareChat) shows that IAC reaches a 34.6% cache hit rate, substantially outperforming the ML-based LPC (18.8%)
and LRU (12.2%), while reducing average serving latency by up to 54.6%.

## Installation

```bash
conda create -n iac python=3.12 -y && conda activate iac
bash scripts/install.sh
conda deactivate && conda activate iac
```

`install.sh` clones SGLang v0.5.19, applies the IAC patch (`patches/`), and installs SGLang, the `iac` package and a
CUDA 12.9 toolchain into the environment. Building SGLang needs Rust (`rustup`).

## IAC Policy

Each workload state (intent *i*, turn *t*) keeps a continuation probability `P(t+1 | i, t)` and a log-normal
inter-turn delay `log D ~ N(mu, sigma^2)` with median `M = exp(mu)`, learned online from recent requests. A cached
entry of state (i, t) that has been idle for `age` seconds has the priority (paper Eq. 1)

```
IAC Priority(i, t, age) = ( 1[age < M_{i,t}],  P(t+1 | i, t),  size )
```


## Serving and Benchmark (WildChat)

WildChat-1M is gated on Hugging Face: log in and download it first.

```bash
hf download allenai/WildChat-1M --repo-type dataset
python scripts/prepare_data.py split
python scripts/prepare_data.py trace
python scripts/learn_stats.py --split data/splits/wildchat/history.jsonl.gz --out stats/wildchat.json
```

Start the server with IAC, then replay the WildChat trace against it.

```bash
bash scripts/serve.sh --policy iac --model Qwen/Qwen3-8B --stats stats/wildchat.json --clock request --capacity 100000
python scripts/replay.py --time-scale 1.6
```

Use `--policy lru` for SGLang's default eviction.

## Intent Classifier

`data/classifier/labels.jsonl.gz` holds the intent labels of the training conversations. Rebuild the training data
from the public datasets, then train and test a classifier:

```bash
python scripts/classifier.py rebuild --sharegpt-json ShareGPT_V3_unfiltered_cleaned_split.json
python scripts/classifier.py train --out checkpoints/intent_classifier.pt
python scripts/classifier.py eval --classifier checkpoints/intent_classifier.pt
```

The released 30-intent classifier is `checkpoints/intent_e5_large_instruct_30cls.pt`.
