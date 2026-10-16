"""Operator screen for the sorting line: what the robot is inspecting, what the inspector
said and why, and the running tally against today's factory.

Reads a run folder written by isaac/run_sorting.py (parts.jsonl + images/) and refreshes
while the run is going. No simulation or model code here: it only reads files.

    .venv/bin/streamlit run dashboard/app.py -- --run runs/sorting/live_A_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from bench_sorting import NAMES, summarise  # noqa: E402

sys.path.insert(0, str(REPO / "isaac"))
from trays import BAD_KINDS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, default=None, help="run_sorting.py output folder")
ap.add_argument("--refresh", type=float, default=5.0, help="seconds between refreshes")
args = ap.parse_args()

st.set_page_config(page_title="Sorting line · operator", layout="wide")

runs = sorted({p.parent for name in ("parts.jsonl", "events.jsonl")
               for p in (REPO / "runs" / "sorting").glob(f"*/{name}")},
              key=lambda p: max(f.stat().st_mtime for f in p.glob("*.jsonl")), reverse=True)
default = runs.index(args.run.resolve()) if args.run and args.run.resolve() in runs else 0
run = st.sidebar.selectbox("Run", runs, index=default, format_func=lambda p: p.name) if runs else None
st.sidebar.caption("Newest first. The page refreshes itself while a run is going.")

KIND_NAMES = {"wrong:nut_m12": "wrong part: M12 nut", "wrong:nut_m20": "wrong part: M20 nut",
              "wrong:bolt_m16": "wrong part: M16 bolt", "chip:top": "chip, top", "crack:top": "crack, top",
              "rust:top": "rust, top", "chip:underside": "chip, underside (hidden from the top camera)",
              "crack:underside": "crack, underside (hidden)", "rust:underside": "rust, underside (hidden)"}
if run is not None:
    st.sidebar.divider()
    st.sidebar.subheader("Inject a fault")
    kind = st.sidebar.selectbox("Next part on the feed tray becomes", BAD_KINDS, format_func=KIND_NAMES.get)
    if st.sidebar.button("Inject", type="primary"):
        (run / "inject.json").write_text(json.dumps({"kind": kind}))
    if (run / "inject.json").exists():
        st.sidebar.info("Queued: the robot swaps it in before its next pick.")

LABEL = {"none": "OK", "wrong_part": "wrong part", "defective_part": "damaged", "unclear": "unclear"}


def evidence(verdict: dict) -> str:
    """The inspector's one-line reason, from its JSON answer."""
    try:
        return json.loads(verdict.get("raw", "")).get("evidence", "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def load_rows(run: Path) -> list[dict]:
    """Finished trays (parts.jsonl, with where each part ended up) plus the parts of the tray
    in progress (events.jsonl, written as each part is decided)."""
    def read(name):
        f = run / name
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    done = read("parts.jsonl")
    seen = {(r["tray"], r["slot"]) for r in done}
    return done + [r for r in read("events.jsonl") if (r["tray"], r["slot"]) not in seen]


def truth(r: dict) -> str:
    if r["kind"] == "good":
        return "good M16 nut"
    if r["kind"] == "wrong":
        return f"wrong part ({r['part'].replace('_', ' ').upper()})" + (" · injected" if r.get("injected") else "")
    return f"{r['defect']} on the {r['side']}" + (" · injected" if r.get("injected") else "")


@st.fragment(run_every=args.refresh)
def screen(run: Path) -> None:
    rows = load_rows(run)
    if not rows:
        st.info("Waiting for the first part…")
        return
    s = summarise(rows)
    ours, today = s["policies"]["ours"], s["policies"]["B"]
    good_parts = s["parts"] - s["bad_parts"]

    st.title("Sorting line")
    st.caption(f"`{run.name}` · {s['trays']} trays · {s['parts']} parts inspected · "
               f"{s['seconds_per_part']} s per part")

    # The tally: the same parts, the same photos, three ways of deciding.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bad parts shipped · ours", f"{ours.get('shipped_bad', 0)} / {s['bad_parts']}")
    c2.metric("Bad parts shipped · today's factory", f"{today.get('shipped_bad', 0)} / {s['bad_parts']}")
    c3.metric("Good parts scrapped · ours", f"{ours.get('scrapped_good', 0)} / {good_parts}")
    placed = sum("ended_in" in r for r in rows)  # the tray in progress isn't checked yet
    c4.metric("Parts handled cleanly", f"{s['physical']['handled_ok']} / {placed}")

    # The part just inspected.
    r = rows[-1]
    st.subheader(f"Last part · tray {r['tray']}, slot {r['slot']}")
    img = run / "images"
    left, mid, right = st.columns([2, 2, 3])
    top = img / f"{r['tray']}_{r['slot']}_top_closeup.png"
    under = img / f"{r['tray']}_{r['slot']}_under_underside.png"
    if top.exists():
        left.image(str(top), caption="Top camera, on the inspection nest")
    if under.exists():
        mid.image(str(under), caption="Underside camera, held over it")
    with right:
        vt, vu = r["verdict_top"], r["verdict_underside"]
        st.markdown(f"**Top:** {LABEL.get(vt['failure_type'], vt['failure_type'])}  \n_{evidence(vt)}_")
        st.markdown(f"**Underside:** {LABEL.get(vu['failure_type'], vu['failure_type'])}  \n_{evidence(vu)}_")
        verb = "Accepted → good-parts rack" if r["action"] == "accept" else "Rejected → scrap bin"
        st.markdown(f"### {verb}")
        ok = r.get("decision_correct")
        st.markdown(f"Truth: **{truth(r)}** · decision {'✅ correct' if ok else '❌ wrong'} · "
                    f"ended in **{r.get('ended_in', 'moving…')}** · {r.get('seconds', '?')} s")

    # Every part so far, newest first.
    st.subheader("All parts")
    table = [{"tray": x["tray"], "slot": x["slot"], "truth": truth(x),
              "top": LABEL.get(x["verdict_top"]["failure_type"], "?"),
              "underside": LABEL.get(x["verdict_underside"]["failure_type"], "?"),
              "action": x["action"], "correct": "✅" if x.get("decision_correct") else "❌",
              "ended in": x.get("ended_in", "?")} for x in reversed(rows)]
    st.dataframe(table, hide_index=True, width="stretch", height=320)
    with st.expander("How the three ways of deciding compare"):
        st.table([{"policy": NAMES[p], "bad shipped": f"{c.get('shipped_bad', 0)}/{s['bad_parts']}",
                   "good scrapped": f"{c.get('scrapped_good', 0)}/{good_parts}"}
                  for p, c in s["policies"].items()])


if run is None:
    st.info("No sorting runs yet: start isaac/run_sorting.py.")
else:
    screen(run)
