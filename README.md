

A minimal MLOps-style batch job that reads OHLCV data, computes a rolling mean on the `close` price, and generates a binary trading signal.

## What it does



**Note:** The first `window - 1` rows produce NaN for the rolling mean and are excluded from signal computation.

## Local Setup

```bash
pip install -r requirements.txt

python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Docker

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

## Example Output (`metrics.json`)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.499,
  "latency_ms": 127,
  "seed": 42,
  "status": "success"
}
```

## Project Structure

| File | Purpose |
|------|---------|
| `run.py` | Main batch job script |
| `config.yaml` | Job configuration (seed, window, version) |
| `data.csv` | Input OHLCV dataset |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition |
| `metrics.json` | Output metrics (generated on run) |
| `run.log` | Output logs (generated on run) |
# mlops-assignment
