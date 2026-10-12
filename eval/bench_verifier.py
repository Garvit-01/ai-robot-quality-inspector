#!/usr/bin/env python3
"""Score the vision verifier against rendered scenes whose correct answer is known.

This is the M2 go/no-go, and later the Tier 1 vs Tier 2 comparison. Every scene goes
through the same call path the real pipeline will use: checkpoint prompt from
config/selfheal.yaml, images as base64 data URIs, strict JSON parsed into the Verdict
schema, one "valid JSON only" retry, anything still unparseable counted as a failure.

Usage:
    .venv/bin/python eval/bench_verifier.py runs/scenes/eval_v1
    .venv/bin/python eval/bench_verifier.py runs/scenes/eval_v1 --model openbmb/MiniCPM-V-4_5
    .venv/bin/python eval/bench_verifier.py runs/scenes/eval_v1 --views closeup --limit 24

Writes to runs/bench/<scene-set>__<model>__<views>/:
    results.jsonl   one line per scene: expected, predicted, raw reply, latency, tokens
    summary.json    the numbers
    summary.md      the same, readable
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from pydantic import ValidationError  # noqa: E402

from _common import (  # noqa: E402
    Verdict,
    client_for,
    image_data_uri,
    load_config,
    load_dotenv,
    require_env,
    strip_to_json,
)

GO_THRESHOLD = 0.85     # SPEC M2: needed on CP_POST_LIFT and CP_POST_PLACE

# The kind of recovery each verdict leads to. Strict failure-type accuracy is the headline;
# this is a SECONDARY metric for the cell's behaviour: calling a defective part "wrong" still
# sends it to the bin, while calling a misaligned seat "fine" ships a bad part.
ACTION = {
    "none": "continue",
    "part_displaced": "regrasp", "empty_grasp": "regrasp", "part_dropped_recoverable": "regrasp",
    "misaligned_seat": "reseat",
    "wrong_part": "reject", "defective_part": "reject",
    "part_dropped_unrecoverable": "stop", "unclear": "stop",
}
NO_GO_THRESHOLD = 0.70  # any checkpoint below this: change camera/crop, or cut the fault


def build_prompt(cfg: dict, checkpoint: str, views: list[str]) -> str:
    prompts = cfg["prompts"]
    question = prompts["checkpoints"][checkpoint].format(**cfg["part_reference"])
    view_note = {
        "overview": "an overview of the whole cell",
        "closeup": "a close-up of the region this checkpoint is about",
    }
    described = "; ".join(f"image {i + 1} is {view_note[v]}" for i, v in enumerate(views))
    return f"{question}\n\nYou are given {len(views)} image(s): {described}.\n\n{prompts['schema_block']}"


def call_verifier(client, model: str, cfg: dict, prompt: str, images: list[Path]) -> dict:
    """One scene, with the same retry policy the pipeline uses. Never raises."""
    v = cfg["verifier"]
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url",
         "image_url": {"url": image_data_uri(p, v["image_longest_side_px"], v["jpeg_quality"])}}
        for p in images
    ]
    out = {"attempts": 0, "latency_ms": 0.0, "tokens_in": 0, "tokens_out": 0,
           "raw": None, "parsed": None, "error": None}
    for attempt in range(v["max_retries"] + 1):
        msgs = [{"role": "system", "content": cfg["prompts"]["system"]},
                {"role": "user", "content": content}]
        if attempt > 0:
            msgs.append({"role": "user", "content":
                         "Return valid JSON only, matching the schema. No prose, no markdown."})
        out["attempts"] = attempt + 1
        resp = None
        for backoff in (0, 2, 5, 12):  # transport errors / rate limits only
            if backoff:
                time.sleep(backoff)
            t0 = time.monotonic()
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=v.get("temperature", 0.0), messages=msgs,
                    max_tokens=400)
                out["latency_ms"] += (time.monotonic() - t0) * 1000
                break
            except Exception as exc:  # noqa: BLE001 — recorded, never fatal
                out["latency_ms"] += (time.monotonic() - t0) * 1000
                out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        if resp is None:
            return out
        if resp.usage:
            out["tokens_in"] += resp.usage.prompt_tokens or 0
            out["tokens_out"] += resp.usage.completion_tokens or 0
        raw = resp.choices[0].message.content or ""
        out["raw"] = raw[:2000]
        try:
            out["parsed"] = Verdict.model_validate_json(strip_to_json(raw)).model_dump()
            out["error"] = None
            return out
        except (ValidationError, ValueError) as exc:
            out["error"] = f"parse: {str(exc)[:200]}"
    return out


def pct(n: int, d: int) -> float | None:
    return None if d == 0 else round(100.0 * n / d, 1)


def summarise(rows: list[dict]) -> dict:
    by_fault, by_cp = defaultdict(list), defaultdict(list)
    for r in rows:
        by_fault[r["fault"]].append(r)
        by_cp[r["checkpoint"]].append(r)

    def block(rs: list[dict]) -> dict:
        n = len(rs)
        parsed = [r for r in rs if r["pred"]]
        # Count over `parsed` but divide by `n`: an unparseable reply is a wrong answer,
        # never silently excluded. (Summing `r["pred"] and ...` over all rows crashed on the
        # first unparseable reply, because `None and x` is None.)
        ft = sum(r["pred"]["failure_type"] == r["exp"]["failure_type"] for r in parsed)
        ok = sum(r["pred"]["step_ok"] == r["exp"]["step_ok"] for r in parsed)
        loc = sum(r["pred"]["location_hint"] == r["exp"]["location_hint"] for r in parsed)
        act = sum(ACTION[r["pred"]["failure_type"]] == ACTION[r["exp"]["failure_type"]] for r in parsed)
        return {
            "n": n,
            "failure_type_acc": pct(ft, n),
            "step_ok_acc": pct(ok, n),
            "location_acc": pct(loc, n),
            "action_acc": pct(act, n),
            "unclear": sum(1 for r in parsed if r["pred"]["failure_type"] == "unclear"),
            "unparseable": n - len(parsed),
        }

    confusion = Counter((r["exp"]["failure_type"], r["pred"]["failure_type"] if r["pred"] else "<no answer>")
                        for r in rows)
    clean = [r for r in rows if r["exp"]["step_ok"]]
    faulty = [r for r in rows if not r["exp"]["step_ok"]]
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    cp_acc = {cp: block(rs)["failure_type_acc"] for cp, rs in by_cp.items()}
    gate = {cp: cp_acc.get(cp) for cp in ("CP_POST_LIFT", "CP_POST_PLACE")}
    return {
        "overall": block(rows),
        "by_fault": {k: block(v) for k, v in sorted(by_fault.items())},
        "by_checkpoint": {k: block(v) for k, v in sorted(by_cp.items())},
        # On clean scenes: how often did it cry wolf? On faulty ones: how often did it miss?
        "false_alarm_rate_on_clean": pct(sum(1 for r in clean if r["pred"] and not r["pred"]["step_ok"]), len(clean)),
        "miss_rate_on_faults": pct(sum(1 for r in faulty if r["pred"] and r["pred"]["step_ok"]), len(faulty)),
        "confusion": [{"expected": e, "predicted": p, "count": c} for (e, p), c in sorted(confusion.items())],
        "latency_ms": {"p50": round(statistics.median(lat)) if lat else None,
                       "p90": round(lat[int(0.9 * (len(lat) - 1))]) if lat else None},
        "tokens": {"in": sum(r["tokens_in"] for r in rows), "out": sum(r["tokens_out"] for r in rows)},
        "go_no_go": {
            "gate_checkpoints_acc": gate,
            "go": all(v is not None and v >= 100 * GO_THRESHOLD for v in gate.values()),
            "checkpoints_below_no_go": [cp for cp, v in cp_acc.items() if v is not None and v < 100 * NO_GO_THRESHOLD],
        },
    }


def render_md(meta: dict, s: dict) -> str:
    lines = [f"# Verifier benchmark — `{meta['model']}`", "",
             f"Scenes: `{meta['scene_dir']}` ({s['overall']['n']}) · views: {', '.join(meta['views'])} · "
             f"prompt {meta.get('prompt_version')} · "
             f"run {meta['started']}", ""]
    o = s["overall"]
    lines += ["## Headline", "",
              f"- Failure type correct: **{o['failure_type_acc']}%**",
              f"- Fault vs no-fault correct (step_ok): **{o['step_ok_acc']}%**",
              f"- Location correct: {o['location_acc']}%",
              f"- Leads to the right kind of recovery (secondary): {o['action_acc']}%",
              f"- False alarms on clean scenes: {s['false_alarm_rate_on_clean']}% · missed faults: {s['miss_rate_on_faults']}%",
              f"- Answered 'unclear': {o['unclear']} · unparseable: {o['unparseable']}",
              f"- Latency p50 {s['latency_ms']['p50']} ms, p90 {s['latency_ms']['p90']} ms · tokens in {s['tokens']['in']:,}, out {s['tokens']['out']:,}",
              ""]
    g = s["go_no_go"]
    verdict = "GO" if g["go"] else "NO-GO"
    lines += [f"## M2 gate: **{verdict}**", "",
              f"Needs ≥{int(GO_THRESHOLD * 100)}% failure-type accuracy on CP_POST_LIFT and CP_POST_PLACE: "
              + ", ".join(f"{k} {v}%" for k, v in g["gate_checkpoints_acc"].items()) + ".",
              f"Checkpoints below {int(NO_GO_THRESHOLD * 100)}%: {', '.join(g['checkpoints_below_no_go']) or 'none'}.", ""]
    for title, key in (("By fault", "by_fault"), ("By checkpoint", "by_checkpoint")):
        lines += [f"## {title}", "", "| | n | failure type | step_ok | location | action | unclear | unparseable |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, b in s[key].items():
            lines.append(f"| {k} | {b['n']} | {b['failure_type_acc']}% | {b['step_ok_acc']}% | "
                         f"{b['location_acc']}% | {b['action_acc']}% | {b['unclear']} | {b['unparseable']} |")
        lines.append("")
    lines += ["## Confusion (expected → predicted)", "", "| expected | predicted | count |", "|---|---|---|"]
    for c in s["confusion"]:
        mark = "" if c["expected"] == c["predicted"] else " ✗"
        lines.append(f"| {c['expected']} | {c['predicted']}{mark} | {c['count']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--model", default=None, help="override VERIFIER_MODEL_ID")
    ap.add_argument("--views", default="overview,closeup")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config()
    client = client_for(cfg["verifier"])
    model = args.model or require_env(cfg["verifier"]["model_id_env"])
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    labels = sorted((args.scene_dir / "labels").glob("*.json"), key=lambda p: int(p.stem))
    if args.limit:
        labels = labels[: args.limit]
    if not labels:
        sys.exit(f"no labels in {args.scene_dir}/labels")

    tag = f"{args.scene_dir.name}__{model.replace('/', '_')}__{'+'.join(views)}"
    out = args.out or REPO / "runs" / "bench" / tag
    out.mkdir(parents=True, exist_ok=True)
    meta = {"scene_dir": str(args.scene_dir), "model": model, "views": views,
            "prompt_version": cfg["prompts"].get("version", "unversioned"),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[bench] {len(labels)} scenes · {model} · views={views} -> {out}", flush=True)

    def job(lp: Path) -> dict:
        lab = json.loads(lp.read_text())
        sc = lab["scene"]
        prompt = build_prompt(cfg, sc["checkpoint"], views)
        imgs = [args.scene_dir / lab["images"][v] for v in views]
        res = call_verifier(client, model, cfg, prompt, imgs)
        return {"seed": lab["seed"], "fault": sc["fault"], "checkpoint": sc["checkpoint"],
                "exp": sc["expected"], "truth": sc.get("truth", {}), "pred": res["parsed"],
                "attempts": res["attempts"], "latency_ms": round(res["latency_ms"]),
                "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"],
                "error": res["error"], "raw": res["raw"]}

    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (out / "results.jsonl").open("w") as fh:
        futures = [pool.submit(job, lp) for lp in labels]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            got = row["pred"]["failure_type"] if row["pred"] else f"<{row['error']}>"
            mark = "✓" if row["pred"] and got == row["exp"]["failure_type"] else "✗"
            print(f"[bench] {i:>3}/{len(labels)} {mark} seed {row['seed']} {row['fault']} "
                  f"expected {row['exp']['failure_type']:<26} got {got}", flush=True)

    rows.sort(key=lambda r: r["seed"])
    summary = summarise(rows)
    summary["meta"] = {**meta, "elapsed_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(render_md(meta, summary))
    o = summary["overall"]
    print(f"[bench] failure_type {o['failure_type_acc']}% · step_ok {o['step_ok_acc']}% · "
          f"{'GO' if summary['go_no_go']['go'] else 'NO-GO'} -> {out / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
