#!/usr/bin/env python3
"""Fetch daily economic indicator snapshots from FRED (Federal Reserve) and IMF.

Usage:
  export FRED_API_KEY="your_key"
  python fetch_economic_indicators.py --output-dir data/economic
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

# Default indicators (you can override by editing these lists or extending the script)
DEFAULT_FRED_SERIES = {
    "DFF": "Effective Federal Funds Rate",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "UNRATE": "Unemployment Rate",
    "GDP": "Gross Domestic Product",
}

# IMF SDMX format: <database>, <frequency>, <indicator>, <country>
# Example endpoint: https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/M.US.PCPI_IX.US
DEFAULT_IMF_SERIES = [
    {
        "database": "IFS",
        "frequency": "M",
        "indicator": "PCPI_IX",
        "country": "US",
        "label": "IMF CPI Index (US)",
    },
    {
        "database": "IFS",
        "frequency": "M",
        "indicator": "LUR_PT",
        "country": "US",
        "label": "IMF Unemployment Rate (US)",
    },
]


def fetch_fred_series(series_id: str, api_key: str) -> Dict:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    observations = payload.get("observations", [])
    if not observations:
        raise ValueError(f"No observations returned for FRED series {series_id}")

    latest = observations[0]
    return {
        "source": "FRED",
        "series_id": series_id,
        "date": latest.get("date"),
        "value": latest.get("value"),
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def fetch_imf_series(database: str, frequency: str, indicator: str, country: str, label: str) -> Dict:
    # IMF SDMX key format: <frequency>.<country>.<indicator>
    key = f"{frequency}.{country}.{indicator}"
    url = f"https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/{database}/{key}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    series = (
        payload.get("CompactData", {})
        .get("DataSet", {})
        .get("Series", {})
    )

    observations = series.get("Obs", []) if isinstance(series, dict) else []
    if isinstance(observations, dict):
        observations = [observations]

    if not observations:
        raise ValueError(f"No observations returned for IMF series {database}/{key}")

    latest = observations[-1]
    return {
        "source": "IMF",
        "series_id": f"{database}:{key}",
        "label": label,
        "date": latest.get("@TIME_PERIOD"),
        "value": latest.get("@OBS_VALUE"),
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def collect_data(fred_api_key: str) -> List[Dict]:
    records: List[Dict] = []

    for series_id, label in DEFAULT_FRED_SERIES.items():
        row = fetch_fred_series(series_id=series_id, api_key=fred_api_key)
        row["label"] = label
        records.append(row)

    for cfg in DEFAULT_IMF_SERIES:
        records.append(fetch_imf_series(**cfg))

    return records


def write_outputs(records: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    json_path = output_dir / f"indicators_{run_date}.json"
    csv_path = output_dir / "indicators_latest.csv"

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    print(f"Wrote daily snapshot: {json_path}")
    print(f"Updated latest CSV:   {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily economic indicators from FRED and IMF."
    )
    parser.add_argument(
        "--output-dir",
        default="data/economic",
        help="Directory for output files (default: data/economic)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fred_api_key = os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise EnvironmentError(
            "Missing FRED_API_KEY. Set it in your environment before running this script."
        )

    records = collect_data(fred_api_key=fred_api_key)
    write_outputs(records=records, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
