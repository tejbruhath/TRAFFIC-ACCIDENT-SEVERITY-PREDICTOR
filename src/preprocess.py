import argparse
import os
import pandas as pd


def preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    if 'severity' not in df.columns:
        raise ValueError("Expected 'severity' column in input data")

    df = df.dropna(subset=['severity']).copy()

    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'], errors='coerce')
        df['year'] = dt.dt.year
        df['month'] = dt.dt.month
        df['day'] = dt.dt.day
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek

    severity_map = {
        'minor': 'minor',
        'serious': 'serious',
        'fatal': 'fatal',
        0: 'minor',
        1: 'serious',
        2: 'fatal',
    }
    df['severity'] = df['severity'].map(lambda x: severity_map.get(x, str(x).lower()))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Preprocess accidents dataset')
    parser.add_argument('--input', required=True, help='Path to raw CSV')
    parser.add_argument('--output', required=True, help='Path to write processed CSV')
    args = parser.parse_args()

    preprocess(args.input, args.output)


if __name__ == '__main__':
    main()
