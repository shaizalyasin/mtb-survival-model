# Reproduces from a clean checkout in two commands: `make setup` then `make train`.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup download verify-data features train ablation test demo serve figures artifact-size clean

setup:
	python3 -m venv .venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.lock
	@echo "ready. next: make train"

download:
	PYTHONPATH=src $(PY) src/download.py

verify-data: download
	PYTHONPATH=src $(PY) src/load.py data/msk_met_2021

features: download
	PYTHONPATH=src $(PY) src/train.py --features-only

train: download
	PYTHONPATH=src $(PY) src/train.py

# All four arms, including the metastasis-count probe that is never served.
ablation: download
	PYTHONPATH=src $(PY) src/ablation.py

test:
	$(PY) -m pytest tests/ -q

demo:
	PYTHONPATH=src $(PY) src/demo_patient.py

figures:
	PYTHONPATH=src $(PY) src/evaluate.py --oncotree LUAD

serve:
	PYTHONPATH=src .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000

# Artefacts are committed so a clone can serve without retraining. If this
# target fails, move them to a release rather than committing a large binary.
artifact-size:
	@size=$$(du -sk artifacts | cut -f1); \
	echo "artifacts: $$size KB"; \
	if [ $$size -gt 51200 ]; then echo "over 50 MB, do not commit"; exit 1; fi

# Named files only, so this cannot widen into the repository.
clean:
	rm -f artifacts/model.json artifacts/feature_schema.json artifacts/model_card.json
	rm -f artifacts/reference_cohort.parquet artifacts/baseline_hazard.parquet
	rm -f figures/km_*.png

# Random search, ranked on out-of-fold C-index over the training split only.
tune: download
	PYTHONPATH=src $(PY) src/tune.py --trials 120 --folds 5
