"""Build model inputs from prepared revenue. Run: python features.py"""
import json
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
VALIDATION_START = "2024-01-01"
TEST_START = "2025-01-01"
FEATURES = ["growth_latest_pct", "growth_previous_pct", "growth_mean_4_pct",
            "growth_change_pp", "log_revenue", "period_end_month"]


def revenue_as_of(versions, cutoff):
    """Latest known revenue per actual fiscal period, for one company."""
    return (versions[versions.filed.le(cutoff)].sort_values(["end", "filed"])
            .drop_duplicates("end", keep="last").set_index("end"))


def feature_values(history, ends):
    """Use eight consecutive fiscal quarters to calculate four YoY rates."""
    if len(ends) != 8 or any(end not in history.index for end in ends):
        return None
    dates = pd.to_datetime(ends)
    if not pd.Series(dates).diff().dropna().dt.days.between(70, 110).all():
        raise ValueError("Missing or irregular fiscal quarter in feature window")
    for i in range(4, 8):
        if not 350 <= (dates[i] - dates[i-4]).days <= 380:
            raise ValueError("Prior-year fiscal period is not comparable")
    revenue = history.loc[ends, "revenue_usd"].astype(float).tolist()
    if min(revenue) <= 0:
        raise ValueError("Revenue must be positive")
    rates = [100 * (revenue[i] / revenue[i-4] - 1) for i in range(4, 8)]
    return dict(growth_latest_pct=rates[-1], growth_previous_pct=rates[-2],
                growth_mean_4_pct=sum(rates)/4, growth_change_pp=rates[-1]-rates[-2],
                log_revenue=math.log(revenue[-1]), period_end_month=dates[-1].month)


def assign_split(forecast_as_of, target_filed):
    """Exclude labels that would cross the next model-selection boundary."""
    if pd.isna(target_filed):
        return "unscored"
    if forecast_as_of < VALIDATION_START:
        return "train" if target_filed < VALIDATION_START else "boundary_excluded"
    if forecast_as_of < TEST_START:
        return "validation" if target_filed < TEST_START else "boundary_excluded"
    return "test"


def build_features(panel):
    records = []
    exclusions = []
    for ticker, versions in panel.groupby("ticker"):
        first = versions.sort_values(["end", "filed"]).drop_duplicates("end").reset_index(drop=True)
        for i in range(7, len(first)):
            current = first.iloc[i]
            cutoff = current.filed
            ends = first.end.iloc[i-7:i+1].tolist()
            history = revenue_as_of(versions, cutoff)
            values = feature_values(history, ends)
            if values is None:
                exclusions.append(dict(ticker=ticker, source_quarter_end=current.end, forecast_as_of=cutoff, reason="Required historical revenue not available at forecast cutoff"))
                continue
            target_end, target_filed, actual = None, None, None
            if i + 1 < len(first):
                target = first.iloc[i+1]
                target_end, target_filed = target.end, target.filed
                gap = (pd.Timestamp(target_end)-pd.Timestamp(current.end)).days
                if not 70 <= gap <= 110:
                    raise ValueError(f"{ticker}: irregular target period")
                if target_filed <= cutoff or target_end <= cutoff:
                    exclusions.append(dict(ticker=ticker, source_quarter_end=current.end, forecast_as_of=cutoff, reason="Next quarter ended or was already filed at forecast time"))
                    continue
                target_history = revenue_as_of(versions, target_filed)
                prior_end = first.iloc[i-3].end
                year_days = (pd.Timestamp(target_end)-pd.Timestamp(prior_end)).days
                if not 350 <= year_days <= 380:
                    raise ValueError(f"{ticker}: target prior-year mismatch")
                actual = 100 * (target_history.loc[target_end, "revenue_usd"] /
                                target_history.loc[prior_end, "revenue_usd"] - 1)
            records.append(dict(ticker=ticker, source_quarter_end=current.end,
                forecast_as_of=cutoff, latest_feature_filed=history.loc[ends,"filed"].max(),
                target_quarter_end=target_end, target_filed=target_filed,
                target_growth_pct=actual, split=assign_split(cutoff,target_filed), **values))
    result = pd.DataFrame(records)
    result.attrs["excluded_forecasts"] = exclusions
    return result


def main():
    folder = ROOT / "data" / "processed"
    review = json.loads((folder / "software_panel_review.json").read_text())
    if any(c["status"] != "structural_checks_passed" for c in review["companies"]):
        raise ValueError("Resolve the current panel review before building features")
    panel = pd.read_parquet(folder / "software_quarterly_versions.parquet")
    features = build_features(panel)
    assert features[FEATURES].notna().all().all()
    assert (features.latest_feature_filed <= features.forecast_as_of).all()
    trained = set(features.loc[features.split.eq("train"), "ticker"])
    evaluated = set(features.loc[features.split.isin(["validation", "test"]), "ticker"])
    assert evaluated <= trained
    output = ROOT / "data" / "features"
    output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output / "software_features.parquet", index=False)
    counts = pd.crosstab(features.ticker, features.split)
    report = dict(features=FEATURES, validation_start=VALIDATION_START, test_start=TEST_START,
                  excluded_forecasts=features.attrs["excluded_forecasts"],
                  rows=len(features), counts=features.split.value_counts().to_dict(),
                  notes=["Splits use forecast dates and label-availability boundaries.",
                         "Test performance must not guide model selection; Microsoft history was previously explored.",
                         "Unscored rows have features but no observed next-quarter label.",
                         "Ticker and date/label/audit columns are metadata, not numerical predictors."])
    (output / "feature_review.json").write_text(json.dumps(report,indent=2))
    print(counts.to_string())
    print(f"Saved {len(features)} rows. No models fitted or test errors calculated.")


if __name__ == "__main__":
    main()
