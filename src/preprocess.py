import argparse
import os
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase, replace spaces and special chars with underscores
    mapper = {c: (
        c.strip()
         .lower()
         .replace(' - ', '_')
         .replace(' ', '_')
         .replace('/', '_')
         .replace('.', '')
         .replace('__', '_')
    ) for c in df.columns}
    return df.rename(columns=mapper)


def _derive_severity_quantiles(df: pd.DataFrame, ratio_col: str) -> pd.Series:
    # Use quantile-based binning into three classes: minor, serious, fatal
    try:
        labels = ['minor', 'serious', 'fatal']
        return pd.qcut(df[ratio_col].rank(method='first'), q=3, labels=labels)
    except Exception:
        # Fallback: simple tertile thresholds on the observed values
        q1 = df[ratio_col].quantile(1/3)
        q2 = df[ratio_col].quantile(2/3)
        def lab(x):
            if x <= q1:
                return 'minor'
            if x <= q2:
                return 'serious'
            return 'fatal'
        return df[ratio_col].apply(lab)


def preprocess(input_path: str, output_path: str, target_strategy: str = 'quantile', fatality_thresholds: str = ''):
    """
    Preprocess ADSI-style accident tables into a modeling-ready CSV.

    - Normalizes column names
    - Computes fatality ratios
    - Derives a categorical 'severity' target (minor/serious/fatal)

    target_strategy:
      - 'quantile' (default): uses tertiles of total fatality ratio
      - 'threshold': expects fatality_thresholds like "0.01,0.02" for minor<=t1, serious<=t2, else fatal
    """
    df = pd.read_csv(input_path)
    df = _normalize_columns(df)

    # Expected numeric columns after normalization (best-effort based on ADSI Table 1A.2)
    # Examples:
    # 'state_ut_city', 'road_accidents_cases', 'road_accidents_injured', 'road_accidents_died',
    # 'total_traffic_accidents_cases', 'total_traffic_accidents_injured', 'total_traffic_accidents_died'

    # Create helpful ratios if totals are present
    if 'total_traffic_accidents_cases' in df.columns and 'total_traffic_accidents_died' in df.columns:
        df['fatality_ratio_total'] = df['total_traffic_accidents_died'] / df['total_traffic_accidents_cases'].replace({0: pd.NA})
    else:
        # Fallback to road-only if total not present
        if 'road_accidents_cases' in df.columns and 'road_accidents_died' in df.columns:
            df['fatality_ratio_total'] = df['road_accidents_died'] / df['road_accidents_cases'].replace({0: pd.NA})

    if 'road_accidents_cases' in df.columns and 'road_accidents_died' in df.columns:
        df['fatality_ratio_road'] = df['road_accidents_died'] / df['road_accidents_cases'].replace({0: pd.NA})

    # Derive severity label
    if target_strategy == 'threshold' and fatality_thresholds:
        try:
            t1_str, t2_str = fatality_thresholds.split(',')
            t1, t2 = float(t1_str), float(t2_str)
        except Exception as e:
            raise ValueError("fatality_thresholds must be like '0.01,0.02'") from e

        def severity_from_threshold(x):
            if pd.isna(x):
                return 'minor'
            if x <= t1:
                return 'minor'
            if x <= t2:
                return 'serious'
            return 'fatal'

        df['severity'] = df['fatality_ratio_total'].apply(severity_from_threshold)
    else:
        df['severity'] = _derive_severity_quantiles(df, 'fatality_ratio_total')

    # Persist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Preprocess accidents dataset')
    parser.add_argument('--input', required=True, help='Path to raw CSV')
    parser.add_argument('--output', required=True, help='Path to write processed CSV')
    parser.add_argument('--target_strategy', default='quantile', choices=['quantile', 'threshold'], help='How to derive severity')
    parser.add_argument('--fatality_thresholds', default='', help="Used with --target_strategy threshold, e.g. '0.01,0.02'")
    args = parser.parse_args()

    preprocess(args.input, args.output, target_strategy=args.target_strategy, fatality_thresholds=args.fatality_thresholds)


if __name__ == '__main__':
    main()
