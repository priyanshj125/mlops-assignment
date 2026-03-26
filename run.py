"""
MLOps batch job — rolling mean signal generator.

Reads OHLCV data, computes a rolling mean on the close price,
generates a binary signal (1 if close > rolling_mean, else 0),
and writes structured metrics + logs.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to both file and console."""
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # file handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console handler (so docker logs show something useful)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> dict:
    """Load and validate the YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("Config file is empty or invalid YAML")

    required = ["seed", "window", "version"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # basic type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"seed must be an integer, got {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"window must be a positive integer, got {cfg['window']}")

    return cfg


def load_data(input_path: str) -> pd.DataFrame:
    """Load and validate the input CSV."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        raise ValueError("Input CSV is empty (no rows)")

    # the csv might have quote-wrapped rows — check if we got a single column
    # containing comma-separated values (common Google Sheets export issue)
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        # re-read with proper quoting
        import io
        raw = path.read_text()
        # strip the outer quotes from each line
        lines = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            lines.append(line)
        df = pd.read_csv(io.StringIO("\n".join(lines)))

    if "close" not in df.columns:
        raise ValueError(
            f"Missing required column 'close'. Got columns: {list(df.columns)}"
        )

    return df


def compute_signals(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling mean and binary signal on close price."""
    df = df.copy()

    df["rolling_mean"] = df["close"].rolling(window=window).mean()

    # for the first (window-1) rows rolling_mean is NaN — we skip those for signal
    df["signal"] = np.nan
    mask = df["rolling_mean"].notna()
    df.loc[mask, "signal"] = (
        (df.loc[mask, "close"] > df.loc[mask, "rolling_mean"]).astype(int)
    )

    return df


def write_metrics(output_path: str, metrics: dict):
    """Write metrics dict to JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MLOps batch signal generator")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path for log file")
    args = parser.parse_args()

    # start timing
    start_time = time.time()

    # set up logging early
    logger = setup_logging(args.log_file)
    logger.info("=" * 50)
    logger.info("Job started")
    logger.info("=" * 50)

    # we'll track version separately so error output can include it
    version = "unknown"

    try:
        # --- load config ---
        logger.info(f"Loading config from: {args.config}")
        cfg = load_config(args.config)
        version = cfg["version"]

        logger.info(f"Config validated — seed={cfg['seed']}, window={cfg['window']}, version={cfg['version']}")

        # --- set seed for reproducibility ---
        np.random.seed(cfg["seed"])
        logger.info(f"Random seed set to {cfg['seed']}")

        # --- load data ---
        logger.info(f"Loading data from: {args.input}")
        df = load_data(args.input)
        logger.info(f"Data loaded — {len(df)} rows, columns: {list(df.columns)}")

        # --- compute rolling mean + signal ---
        logger.info(f"Computing rolling mean with window={cfg['window']}")
        df = compute_signals(df, cfg["window"])

        valid_signals = df["signal"].dropna()
        logger.info(f"Signal generation complete — {len(valid_signals)} valid signals out of {len(df)} rows")

        # --- calculate metrics ---
        rows_processed = len(df)
        signal_rate = round(float(valid_signals.mean()), 4)
        latency_ms = round((time.time() - start_time) * 1000)

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": latency_ms,
            "seed": cfg["seed"],
            "status": "success",
        }

        logger.info(f"Metrics: rows_processed={rows_processed}, signal_rate={signal_rate}, latency_ms={latency_ms}")

        # --- write output ---
        write_metrics(args.output, metrics)
        logger.info(f"Metrics written to: {args.output}")

        # also print to stdout (useful for docker)
        print(json.dumps(metrics, indent=2))

        logger.info("=" * 50)
        logger.info("Job completed — status: success")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)

        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": str(e),
        }
        write_metrics(args.output, error_metrics)
        print(json.dumps(error_metrics, indent=2))

        logger.info("=" * 50)
        logger.info("Job completed — status: error")
        logger.info("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
