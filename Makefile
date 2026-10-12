# One entry point for everything you can run. `make help` lists it.
#
# PART selects the part set: fastener (default) or cube.  e.g.  make demo PART=cube
# Costs: rendering uses the local T4 (free). Model calls use Nebius Token Factory credits;
# every target says roughly how much.

PART      ?= fastener
PY        := .venv/bin/python
ISAAC     := ~/isaacsim/python.sh
CONFIG    := $(if $(filter fastener,$(PART)),config/selfheal_fastener.yaml,config/selfheal.yaml)
PREFIX    := $(if $(filter fastener,$(PART)),f,)
DEMO      := runs/scenes/demo_$(PART)
EVAL      := runs/scenes/$(PREFIX)eval_check
DEMO_SEEDS := 1000,1001,1002,1003,1005,1007,1009,1012,1014,1031

SEED      ?= 1031

.PHONY: help status test smoke demo view bench-demo eval report

help:  ## list what you can run
	@echo "make status       progress dashboard: milestones, latest results, what's running (free, instant)"
	@echo "make test         all unit tests (free, ~2 min)"
	@echo "make smoke        live calls: Nemotron planner + vision verifier (< 1 cent, ~20 s)"
	@echo "make demo         render one scene of each fault type and open them (free, ~1-2 min)"
	@echo "make view SEED=1031  open ONE scene in Isaac Sim with its window, to look around in 3D"
	@echo "make bench-demo   score the verifier on those demo scenes (~1 cent, ~1 min)"
	@echo "make eval         full held-out check: render 120 + score + A/B/C/D + PDF (~5 cents, ~15 min)"
	@echo "make report       rebuild the PDF from the latest results (free, seconds)"
	@echo ""
	@echo "Add PART=cube to use the original cubes instead of fasteners, e.g. make demo PART=cube"

status:
	@$(PY) scripts/status.py

test:
	$(PY) -m pytest

smoke:
	$(PY) scripts/smoke_nemotron.py --failure-type wrong_part
	$(PY) scripts/smoke_verifier.py --selftest

demo:
	rm -rf $(DEMO)
	$(ISAAC) isaac/stage_scene.py --part-set $(PART) --seeds $(DEMO_SEEDS) --out $(DEMO) 2>&1 | grep --line-buffered "\[stage\]"
	$(PY) eval/contact_sheet.py $(DEMO) --view closeup --cols 5
	-xdg-open $(DEMO)/contact_closeup.png >/dev/null 2>&1 &

view:  ## needs the DCV desktop (a screen); don't run while a render is going
	$(ISAAC) isaac/view_scene.py --part-set $(PART) --seed $(SEED)

bench-demo:
	$(PY) eval/bench_verifier.py $(DEMO) --config $(CONFIG) --workers 3 --out runs/bench/demo_$(PART)
	@cat runs/bench/demo_$(PART)/summary.md | sed -n '1,20p'

eval:
	rm -rf $(EVAL)
	$(ISAAC) isaac/stage_scene.py --part-set $(PART) --start 5000 --count 120 --out $(EVAL) 2>&1 | grep --line-buffered "\[stage\]"
	$(PY) eval/bench_verifier.py $(EVAL) --config $(CONFIG) --workers 3 --out runs/bench/$(PREFIX)eval_check__$(PART)
	$(PY) eval/bench_decisions.py $(EVAL) runs/bench/$(PREFIX)eval_check__$(PART) --config $(CONFIG) --out runs/decisions/$(PREFIX)eval_check
	$(PY) eval/make_report.py runs/bench/$(PREFIX)eval_check__$(PART) --decisions runs/decisions/$(PREFIX)eval_check --out reports/eval_check_$(PART).pdf
	@echo "report: reports/eval_check_$(PART).pdf"

report:
	$(PY) eval/make_report.py runs/bench/fdev_v2__fastener-v1 runs/bench/feval_v1__fastener-v1 --decisions runs/decisions/feval_v1 --out reports/evaluation_report_fastener.pdf
	$(PY) eval/make_report.py runs/bench/dev_v2__prompt_v3 runs/bench/eval_v2__prompt_v3__gemma --decisions runs/decisions/eval_v2__guardrails_v2 --out reports/evaluation_report.pdf
