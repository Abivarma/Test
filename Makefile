PY ?= python3
.PHONY: setup data test run smoke
setup:   ## create venv + install
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
data:    ## regenerate deterministic datasets
	$(PY) scripts/generate_data.py
test:
	$(PY) -m pytest -q
smoke:   ## headless before/after printout
	$(PY) -m simulations.scenario_engine
run:
	streamlit run app.py
