#!/usr/bin/env python3
"""Build a PDF evaluation report from benchmark runs.

Re-run it whenever there are new results; it is the "where are we" document.

Usage:
    .venv/bin/python eval/make_report.py runs/bench/dev_v2__prompt_v1 runs/bench/dev_v2__prompt_v2 \
        --out reports/evaluation_report.pdf

Pages:
  1. Summary table of every run given, plus caveats that apply to all of them.
  2. Failure-type accuracy per fault class, one bar group per fault, one bar per run.
  3. Confusion matrix of the LAST run given (treated as the headline run).
  4. A gallery of the last run's mistakes: the image the model saw, what it said, what
     was true. Numbers say how often; this page says why.
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

# Categorical slots 1-4 of the dataviz skill's validated reference palette (light mode),
# assigned in fixed order. Every bar also carries its value as text, so identity and
# magnitude never depend on colour alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK_2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
FAULT_NAMES = {
    "F0": "F0 clean", "F1": "F1 displaced", "F2": "F2 dropped",
    "F3": "F3 misaligned", "F4": "F4 wrong part", "F5": "F5 defective",
}
CAVEATS = [
    "Synthetic renders only. Nothing here is evidence about real cameras or real parts.",
    "One run per configuration at temperature 0; with ~10 scenes per fault class, a single "
    "scene moves a per-fault number by ~10 points. Treat per-fault differences under ~20 "
    "points as noise.",
    "Self-reported confidence is not used: the model gave 1.0 on wrong answers.",
    "Dev-split numbers were used to tune prompts, so they are optimistic. Only eval-split "
    "numbers (seeds 5000-5119) are unbiased.",
]


def load(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    s["_dir"] = run
    s["_rows"] = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    return s


def label(s: dict) -> str:
    m = s["meta"]
    model = m["model"].split("/")[-1]
    return f"{Path(m['scene_dir']).name} · prompt {m.get('prompt_version', '?')} · {model} · {'+'.join(m['views'])}"


OUTCOME_LABELS = [("correct", "correct"), ("quality_escape", "bad part shipped"),
                  ("operator_call", "operator called"), ("silent_fail", "silent failure"),
                  ("false_alarm", "false alarm"), ("wrong_recovery", "wrong recovery")]
COND_NAMES = {"A": "A  no checks", "B": "B  industrial baseline\n    (gripper sensor only)",
              "C": "C  verifier + fusion\n    + Nemotron planner", "D": "D  verifier + fusion\n    + lookup table"}


def page_decisions(pdf, run: Path):
    """What each recovery policy does with the same 120 scenes."""
    s = json.loads((run / "summary.json").read_text())
    conds = ["A", "B", "C", "D"]
    n = s["meta"]["n"]
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.20, right=0.97)
    for i, c in enumerate(reversed(conds)):
        left = 0
        for j, (key, _) in enumerate(OUTCOME_LABELS):
            v = s["conditions"][c]["outcomes"][key]
            if v:
                ax.barh(i, v, left=left, color=SERIES[j], edgecolor="white", linewidth=2, height=0.62)
                if v >= 4:
                    ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=9,
                            color="white" if j in (0, 5) else INK)
                left += v
    ax.set_yticks(range(len(conds)), [COND_NAMES[c] for c in reversed(conds)], fontsize=10, color=INK)
    ax.set_xlim(0, n)
    ax.set_xlabel(f"scenes (of {n})", color=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[j]) for j in range(len(OUTCOME_LABELS))]
    ax.legend(handles, [lab for _, lab in OUTCOME_LABELS], ncol=6, loc="upper left",
              bbox_to_anchor=(0, 1.08), frameon=False, fontsize=9)
    fig.text(0.05, 0.93, "What each recovery policy does with the same scenes", fontsize=16, color=INK, weight="bold")
    p = s["planner"]
    fig.text(0.05, 0.895, f"Held-out scenes, one decision per scene. Planner: {p['calls']} calls, "
             f"{p['guardrail_overrides']} guardrail overrides, p50 {p['latency_ms_p50']} ms.", fontsize=9, color=INK_2)
    cd = s["c_vs_d"]
    notes = [
        f"B lets every wrong, defective and misaligned part through by construction: a finger gap "
        f"can't see identity, damage or seating ({s['conditions']['B']['outcomes']['quality_escape']} bad parts shipped).",
        f"C and D see identical fused verdicts and disagree on {cd['disagreements']} scenes: the table is right on "
        f"{cd['d_right']}, the LLM on {cd['c_right']}. With a five-tool menu, the LLM adds no accuracy over a table.",
        "Per-checkpoint decisions only: recovery success and whole cycles need the closed loop (future work).",
    ]
    y = 0.14
    for t in notes:
        fig.text(0.05, y, "• " + t, fontsize=9, color=INK, wrap=True)
        y -= 0.035
    pdf.savefig(fig)
    plt.close(fig)


def page_summary(pdf, runs):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.93, "Verifier evaluation report", fontsize=20, color=INK, weight="bold")
    fig.text(0.05, 0.895, f"Self-Healing Work Cell · generated {time.strftime('%Y-%m-%d %H:%M')}",
             fontsize=10, color=INK_2)
    head = runs[-1]
    o, g = head["overall"], head["go_no_go"]
    gate = "GO" if g["go"] else "NO-GO"
    fig.text(0.05, 0.855, f"Headline run: {label(head)}", fontsize=11, color=INK)
    fig.text(0.05, 0.775, f"{o['failure_type_acc']}%", fontsize=34, color=INK, weight="bold")
    fig.text(0.05, 0.745, "failure type correct", fontsize=10, color=INK_2)
    fig.text(0.27, 0.775, f"{o['step_ok_acc']}%", fontsize=34, color=INK, weight="bold")
    fig.text(0.27, 0.745, "fault vs no-fault correct", fontsize=10, color=INK_2)
    fig.text(0.49, 0.775, gate, fontsize=34, color=INK, weight="bold")
    fig.text(0.49, 0.745, "M2 gate (≥85% on lift and place)", fontsize=10, color=INK_2)

    cols = ["Run", "n", "Failure type", "Fault/no-fault", "Action*", "False alarms", "Missed", "p50 ms", "Gate"]
    rows = []
    for s in runs:
        so = s["overall"]
        rows.append([label(s), so["n"], f"{so['failure_type_acc']}%", f"{so['step_ok_acc']}%",
                     f"{so.get('action_acc', '–')}%", f"{s['false_alarm_rate_on_clean']}%",
                     f"{s['miss_rate_on_faults']}%", s["latency_ms"]["p50"],
                     "GO" if s["go_no_go"]["go"] else "NO-GO"])
    ax = fig.add_axes([0.05, 0.36, 0.9, 0.36])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="upper left", cellLoc="left",
                 colWidths=[0.40, 0.04, 0.08, 0.09, 0.07, 0.08, 0.06, 0.06, 0.06])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
    fig.text(0.05, 0.33, "* Action: did the verdict lead to the right kind of recovery? Secondary to "
             "failure type; e.g. calling a defective part 'wrong' still sends it to the bin.",
             fontsize=8, color=INK_2)
    y = 0.27
    fig.text(0.05, y, "Read every number on this page with these in mind", fontsize=11, color=INK, weight="bold")
    for c in CAVEATS:
        for i, line in enumerate(textwrap.wrap(c, 140)):
            y -= 0.028
            fig.text(0.06, y, ("• " if i == 0 else "  ") + line, fontsize=9, color=INK)
        y -= 0.006
    pdf.savefig(fig)
    plt.close(fig)


def page_per_fault(pdf, runs):
    runs = runs[-4:]  # at most 4 series: slots 1-4
    faults = sorted(runs[-1]["by_fault"])
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.98)
    width = 0.8 / len(runs)
    for i, s in enumerate(runs):
        xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(faults))]
        vals = [s["by_fault"].get(f, {}).get("failure_type_acc") or 0 for f in faults]
        bars = ax.bar(xs, vals, width * 0.92, color=SERIES[i], label=label(s), zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(faults)), [FAULT_NAMES.get(f, f) for f in faults], color=INK)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Failure type correct (%)", color=INK_2)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    fig.text(0.07, 0.93, "Which faults the verifier gets right", fontsize=16, color=INK, weight="bold")
    fig.text(0.07, 0.895, "Failure-type accuracy per fault class. ~10 scenes per class on dev, 20 on eval: "
             "differences under ~20 points are within noise.", fontsize=9, color=INK_2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=1, frameon=False, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def page_confusion(pdf, s):
    exp = sorted({c["expected"] for c in s["confusion"]})
    pred = sorted({c["predicted"] for c in s["confusion"]} | set(exp))
    grid = [[0] * len(pred) for _ in exp]
    for c in s["confusion"]:
        grid[exp.index(c["expected"])][pred.index(c["predicted"])] = c["count"]
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.subplots_adjust(top=0.84, bottom=0.30, left=0.24, right=0.97)
    ax.imshow(grid, cmap="Blues", aspect="auto")
    top = max(max(r) for r in grid) or 1
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                        color="white" if v > top * 0.55 else INK,
                        weight="bold" if exp[i] == pred[j] else "normal")
    ax.set_xticks(range(len(pred)), pred, rotation=40, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(len(exp)), exp, fontsize=9, color=INK)
    ax.set_xlabel("What the model said", color=INK_2)
    ax.set_ylabel("What was true", color=INK_2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.05, 0.93, "Where the mistakes go", fontsize=16, color=INK, weight="bold")
    fig.text(0.05, 0.895, f"Confusion matrix · {label(s)}. Diagonal (bold) = correct.", fontsize=9, color=INK_2)
    pdf.savefig(fig)
    plt.close(fig)


def page_errors(pdf, s, max_items=12):
    scene_dir = Path(s["meta"]["scene_dir"])
    wrong = [r for r in s["_rows"] if not r["pred"] or r["pred"]["failure_type"] != r["exp"]["failure_type"]]
    if not wrong:
        return
    # Spread the gallery across faults instead of showing 12 of the same mistake.
    by_fault: dict[str, list] = {}
    for r in wrong:
        by_fault.setdefault(r["fault"], []).append(r)
    picked = []
    while len(picked) < max_items and any(by_fault.values()):
        for f in sorted(by_fault):
            if by_fault[f] and len(picked) < max_items:
                picked.append(by_fault[f].pop(0))
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.04, 0.95, "Mistakes, with the image the model saw", fontsize=16, color=INK, weight="bold")
    fig.text(0.04, 0.925, f"{len(wrong)} wrong of {len(s['_rows'])} · close-up view · {label(s)}",
             fontsize=9, color=INK_2)
    cols = 4
    for k, r in enumerate(picked):
        lab = json.loads((scene_dir / "labels" / f"{r['seed']}.json").read_text())
        img = Image.open(scene_dir / lab["images"]["closeup"]).convert("RGB")
        ax = fig.add_axes([0.03 + (k % cols) * 0.24, 0.66 - (k // cols) * 0.295, 0.22, 0.18])
        ax.imshow(img)
        ax.axis("off")
        said = r["pred"]["failure_type"] if r["pred"] else "(no valid answer)"
        ev = (r["pred"] or {}).get("evidence", "") if r["pred"] else ""
        ax.set_title(f"{r['seed']} {r['fault']} · true: {r['exp']['failure_type']}\nsaid: {said}",
                     fontsize=7, color=INK, loc="left")
        lines = textwrap.wrap(f"“{ev}”", 58)
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] = lines[-1].rstrip() + " …"
        fig.text(0.03 + (k % cols) * 0.24, 0.65 - (k // cols) * 0.295,
                 "\n".join(lines), fontsize=6, color=INK_2, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path, help="benchmark run dirs; the LAST is the headline")
    ap.add_argument("--out", type=Path, default=Path("reports/evaluation_report.pdf"))
    ap.add_argument("--decisions", type=Path, default=None, help="a bench_decisions.py run dir")
    args = ap.parse_args()
    runs = [load(r) for r in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        page_summary(pdf, runs)
        if args.decisions:
            page_decisions(pdf, args.decisions)
        page_per_fault(pdf, runs)
        page_confusion(pdf, runs[-1])
        page_errors(pdf, runs[-1])
        d = pdf.infodict()
        d["Title"] = "Self-Healing Work Cell — verifier evaluation"
    print(f"report -> {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
