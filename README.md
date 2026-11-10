# Software revenue forecasting

Can linear regression improve on carrying forward the latest year-over-year revenue growth rate? This project uses SEC filings to forecast next-quarter growth for eight software companies.

On the saved 2024-2026 replay, persistence achieved **1.92 percentage points of MAE** across 82 scored forecasts, compared with **2.28** for linear regression. The replay is retrospective, and Microsoft history was explored before it. The newest year is partial.

## Run the analysis

Use Python 3.12. Install dependencies and build the dataset:

```powershell
python -m pip install -r requirements.txt
python data.py
python features.py
```

If the raw snapshots are missing, set `SEC_USER_AGENT` to a project name and your contact email, then use `python data.py --download` for the first step. Use `--refresh` to fetch newer snapshots and retain earlier filing versions.

Open `analysis.ipynb` for data exploration, forecast comparison, and error analysis. Optionally, run `python backtest.py` to generate an HTML scorecard and CSV under `data/results/`.

## How it works

- `data.py` extracts quarterly revenue and derives missing Q4s from annual and nine-month filings.
- `features.py` builds inputs using figures available at each historical forecast date.
- `backtest.py` fits on earlier forecasts with known outcomes and compares regression with persistence and a four-quarter mean.
- `python -m pytest -q` checks fiscal calendars, revisions, and forecast timing. Tests use fixtures and run offline.

Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
# sec-filing-modeling
