"""Replay forecasts in filing-date order and write an offline HTML scorecard."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from features import VALIDATION_START

ROOT = Path(__file__).resolve().parent
PREDICTORS = ["growth_latest_pct", "growth_previous_pct", "growth_mean_4_pct", "log_revenue"]
TARGET = "target_growth_pct"
MODELS = {
    "persistence": "Persistence",
    "mean_4": "Four-quarter mean",
    "linear": "Linear regression",
}


def backtest(features, start=VALIDATION_START):
    """Fit on earlier forecasts whose outcomes are known by the cutoff."""
    labeled = features[features[TARGET].notna()]
    replay = features[features.forecast_as_of.ge(start)]
    windows = []
    for cutoff, rows in replay.groupby("forecast_as_of", sort=True):
        train = labeled[labeled.forecast_as_of.lt(cutoff) & labeled.target_filed.le(cutoff)]
        if len(train) <= len(PREDICTORS):
            raise ValueError(f"Insufficient training history at {cutoff}")
        model = LinearRegression().fit(train[PREDICTORS], train[TARGET])
        records = rows[["ticker", "source_quarter_end", "forecast_as_of",
                        "target_quarter_end", "target_filed", TARGET]].copy()
        records["train_rows"] = len(train)
        records["latest_training_label_filed"] = train.target_filed.max()
        records["persistence"] = rows.growth_latest_pct
        records["mean_4"] = rows.growth_mean_4_pct
        records["linear"] = model.predict(rows[PREDICTORS])
        for name in MODELS:
            records[f"{name}_error_pp"] = (records[name] - records[TARGET]).abs()
        records["linear_minus_persistence_pp"] = records.linear_error_pp - records.persistence_error_pp
        windows.append(records)
    if not windows:
        raise ValueError(f"No forecast rows on or after {start}")
    return pd.concat(windows, ignore_index=True).sort_values(["forecast_as_of", "ticker"])


def summarize(scored, group):
    errors = [f"{name}_error_pp" for name in MODELS]
    grouped = scored.groupby(group)
    table = grouped[errors].mean()
    table.columns = MODELS.values()
    table["Linear minus persistence"] = table["Linear regression"] - table["Persistence"]
    table.insert(0, "Forecasts", grouped.size())
    return table


def scorecard(predictions):
    scored = predictions.dropna(subset=[TARGET])
    by_year = scored.assign(Year=scored.forecast_as_of.str[:4])
    by_company = summarize(scored, "ticker")
    annual = summarize(by_year, "Year")
    overall = summarize(scored.assign(Year="All"), "Year")
    pending = predictions[predictions[TARGET].isna()]
    forecasts = ["ticker", "forecast_as_of", "target_quarter_end", "persistence", "mean_4", "linear"]
    outcomes = [TARGET, "persistence_error_pp", "mean_4_error_pp", "linear_error_pp",
                "linear_minus_persistence_pp"]
    labels = dict(ticker="Ticker", forecast_as_of="Forecast date", target_quarter_end="Target fiscal end",
                  persistence="Persistence (%)", mean_4="Four-quarter mean (%)", linear="Linear (%)",
                  target_growth_pct="Actual (%)", persistence_error_pp="Persistence error (pp)",
                  mean_4_error_pp="Four-quarter mean error (pp)", linear_error_pp="Linear error (pp)",
                  linear_minus_persistence_pp="Linear minus persistence (pp)")

    def table(frame, index=True):
        return frame.rename(columns=labels).to_html(
            index=index, border=0, na_rep="Pending", float_format="{:.2f}".format
        )

    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Software revenue forecast scorecard</title>
<style>
body {{font: 16px/1.5 system-ui; color: #172433; margin: 2rem auto; max-width: 72rem; padding: 0 1rem}}
section {{overflow-x: auto; margin-bottom: 2rem}}
table {{border-collapse: collapse; font-variant-numeric: tabular-nums; width: 100%}}
th, td {{padding: .45rem .6rem; border-bottom: 1px solid #ced9e3; text-align: right}}
thead {{background: #e8f1f8}}
</style>
<h1>Software revenue forecast scorecard</h1>
<p>Retrospective replay of next-fiscal-quarter year-over-year revenue growth.
Predictions and actuals are percentages. Absolute errors and mean absolute error
are in percentage points (pp). A negative linear-minus-persistence value means
linear regression had lower error. Missing outcomes are excluded from scoring.</p>
<p>Each fit uses earlier forecasts with outcomes available by that day's filing
cutoff. Microsoft history was previously explored. These are historical
estimates, with an incomplete newest year and a small selected company panel.</p>
<section><h2>Mean absolute error by forecast year</h2>{table(pd.concat([annual, overall]))}</section>
<section><h2>Mean absolute error by company</h2>{table(by_company)}</section>
<section><h2>Forecasts awaiting an observed outcome</h2>{table(pending[forecasts], index=False)}</section>
<section><h2>Scored forecasts</h2>{table(scored[forecasts + outcomes].iloc[::-1], index=False)}</section>
<p>Download the <a href="walk_forward_predictions.csv">forecast records and training cutoffs</a>.</p>
</html>
"""


def main():
    features = pd.read_parquet(ROOT / "data" / "features" / "software_features.parquet")
    predictions = backtest(features)
    output = ROOT / "data" / "results"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "walk_forward_predictions.csv", index=False)
    (output / "scorecard.html").write_text(scorecard(predictions), encoding="utf-8")
    print(summarize(predictions.dropna(subset=[TARGET]).assign(window="Overall"), "window"))


if __name__ == "__main__":
    main()


