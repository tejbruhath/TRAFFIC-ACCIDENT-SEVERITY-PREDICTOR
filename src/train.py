import argparse
import json
import os
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(processed_csv: str, config_path: str):
    cfg = load_config(config_path) if config_path else {}
    df = pd.read_csv(processed_csv)

    target = cfg.get('target', 'severity')
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    features = cfg.get('features')
    if not features:
        features = [c for c in df.columns if c != target]

    X = df[features].copy()
    y = df[target].copy()

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.get('split', {}).get('test_size', 0.2),
        stratify=y if y.nunique() > 1 else None,
        random_state=cfg.get('split', {}).get('random_state', 42),
    )

    model_cfg = cfg.get('model', {})
    params = model_cfg.get('params', {'n_estimators': 200, 'random_state': 42})
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    model_dir = cfg.get('paths', {}).get('model_dir', 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'rf_v1.joblib')
    joblib.dump(clf, model_path)

    metrics = {
        'accuracy': acc,
        'classification_report': report,
        'features': features,
        'model_path': model_path,
    }

    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--data', required=True, help='Path to processed CSV')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config YAML')
    args = parser.parse_args()

    train(args.data, args.config)


if __name__ == '__main__':
    main()
