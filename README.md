# Traffic Accident Predictor

## Quick start

- Create and activate a virtual environment, then install deps:
```bash
pip install -r requirements.txt
```

- Preprocess raw data to produce a clean CSV:
```bash
python src/preprocess.py --input data/raw/accidents.csv --output data/processed/accidents_clean.csv
```

- Train baseline model using the processed CSV and default config:
```bash
python src/train.py --data data/processed/accidents_clean.csv --config config/config.yaml
```

Artifacts are saved to `model/`.
# TRAFFIC-ACCIDENT-SEVERITY-PREDICTOR
