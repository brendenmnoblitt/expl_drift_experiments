PYTHON ?= python
NBEXEC ?= jupyter nbconvert --to notebook --execute

.PHONY: paper-results test

paper-results:
	mkdir -p results/executed
	$(NBEXEC) notebooks/statistical_analysis.ipynb --output-dir results/executed --output statistical_analysis.executed.ipynb

test:
	$(PYTHON) -m pytest -q tests
