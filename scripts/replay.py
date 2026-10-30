import argparse
import asyncio
import json
import math
import time

import aiohttp
import numpy as np

from iac.data import Trace


async def send(session, url, trace, req, max_tokens, t0, delay, out):
    await asyncio.sleep(max(0.0, t0 + delay - time.perf_counter()))
    body = {
        "model": "default",
        "messages": trace.messages(req),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "custom_params": {"iac_timestamp": req.arrival, "iac_conv_id": req.conv_id},
    }
    sent = time.perf_counter()
    rec = {"arrival": req.arrival, "ttft": None, "prompt_tokens": 0, "cached_tokens": 0, "error": None}
    try:
        async with session.post(f"{url}/v1/chat/completions", json=body) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
            else:
                async for line in resp.content:
                    line = line.strip()
                    if not line.startswith(b"data:") or line == b"data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    if chunk.get("choices") and rec["ttft"] is None:
                        rec["ttft"] = time.perf_counter() - sent
                    usage = chunk.get("usage")
                    if usage:
                        rec["prompt_tokens"] = usage.get("prompt_tokens", 0)
                        rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    except Exception as e:
        rec["error"] = repr(e)
    out.append(rec)


async def replay(args, trace):
    reqs = sorted((r for r in trace.requests if args.start <= r.arrival < args.end), key=lambda r: r.arrival)
    out = []
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                send(session, args.url, trace, r, args.max_tokens, t0, (r.arrival - args.start) / args.time_scale, out)
            )
            for r in reqs
        ]
        await asyncio.gather(*tasks)
    return out


def summarize(records, warmup_end):
    measured = [r for r in records if r["arrival"] >= warmup_end]
    ok = [r for r in measured if r["error"] is None and r["ttft"] is not None]
    ttft = np.array([r["ttft"] for r in ok]) if ok else np.array([math.nan])
    prompt = sum(r["prompt_tokens"] for r in ok)
    return {
        "requests": len(measured),
        "completed": len(ok),
        "errors": len(measured) - len(ok),
        "hit_rate": sum(r["cached_tokens"] for r in ok) / prompt if prompt else 0.0,
        "ttft_mean": float(ttft.mean()),
        "ttft_p50": float(np.percentile(ttft, 50)),
        "ttft_p90": float(np.percentile(ttft, 90)),
        "ttft_p99": float(np.percentile(ttft, 99)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="traces/wildchat_r3")
    p.add_argument("--url", default="http://localhost:30000")
    p.add_argument("--time-scale", type=float, default=1.6)
    p.add_argument("--start", type=float, default=1800.0)
    p.add_argument("--warmup-end", type=float, default=2100.0)
    p.add_argument("--end", type=float, default=3030.0)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--out")
    args = p.parse_args()

    records = asyncio.run(replay(args, Trace.load(args.trace)))
    errors = [r["error"] for r in records if r["error"]]
    if errors:
        print(f"{len(errors)} errors, first: {errors[0]}")
    summary = summarize(records, args.warmup_end)
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "summary": summary, "requests": records}, f)


if __name__ == "__main__":
    main()
