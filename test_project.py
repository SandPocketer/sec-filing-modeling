import pandas as pd
import pytest
from data import TAG, build_quarterly_versions, validate_company, prepare_company
from features import build_features, feature_values, assign_split, FEATURES

def row(start, end, days, value, filed):
    return dict(concept=TAG, start=start, end=end, duration_days=days,
                revenue_usd=value, filed=filed, accession=filed, form="10-K", priority=0)


def test_q4_excludes_future_revision():
    source = pd.DataFrame([
        row("2023-07-01", "2024-03-31", 275, 300, "2024-04-25"),
        row("2023-07-01", "2024-03-31", 275, 320, "2025-04-25"),
        row("2023-07-01", "2024-06-30", 366, 450, "2024-07-25"),
    ])
    q4 = build_quarterly_versions(source).iloc[0]
    assert q4.revenue_usd == 150
    assert q4.nine_month_filed == "2024-04-25"
    assert q4.start == "2024-04-01"



def rows(ends):
    return pd.DataFrame([dict(ticker='EXAMPLE', start=(pd.Timestamp(end)-pd.Timedelta(days=89)).strftime('%Y-%m-%d'), end=end, filed=(pd.Timestamp(end)+pd.Timedelta(days=25)).strftime('%Y-%m-%d'), revenue_usd=100) for end in ends])


def test_week_based_calendar_is_allowed():
    assert validate_company(rows(['2024-03-01', '2024-05-31', '2024-08-30'])) == []


def test_missing_quarter_is_flagged():
    issues = validate_company(rows(['2024-03-31', '2024-09-30']))
    assert any('Irregular quarter spacing' in issue for issue in issues)


def test_duplicate_keys_are_flagged():
    data = rows(['2024-03-31'])
    assert 'Duplicate company/period/filing keys' in validate_company(pd.concat([data, data]))


def test_adobe_mapping_and_53_week_year():
    facts = [dict(start="2020-11-28", end="2021-09-03", val=300, filed="2021-09-20", accn="nine", form="10-Q"),
             dict(start="2020-11-28", end="2021-12-03", val=450, filed="2022-01-20", accn="annual", form="10-K")]
    company = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": facts}}}}}
    result = prepare_company(company, "ADBE", "0000796343")
    assert result.iloc[0].revenue_usd == 150
    assert result.iloc[0].ticker == "ADBE"
    assert result.iloc[0].concept == "Revenues"


def panel():
    ends = pd.date_range('2020-03-06', periods=16, freq='13W-FRI')
    return pd.DataFrame([dict(ticker='EXAMPLE', end=end.strftime('%Y-%m-%d'),
        filed=(end+pd.Timedelta(days=25)).strftime('%Y-%m-%d'), revenue_usd=100+i*10)
        for i,end in enumerate(ends)])


def test_week_based_calendar_and_known_growth():
    data = panel()
    result = build_features(data)
    assert len(result) == 9
    assert result.iloc[0].growth_latest_pct == pytest.approx(100*(170/130-1))
    assert pd.isna(result.iloc[-1].target_growth_pct)


def test_future_revisions_do_not_change_features():
    data = panel()
    revised = pd.concat([data, pd.DataFrame([{**data.iloc[0].to_dict(), 'filed':'2026-01-01','revenue_usd':999}])])
    pd.testing.assert_frame_equal(build_features(data)[FEATURES], build_features(revised)[FEATURES])


def test_company_histories_are_separate():
    data = panel()
    other = data.assign(ticker='OTHER', revenue_usd=data.revenue_usd*100)
    result = build_features(pd.concat([data,other]))
    a = result[result.ticker.eq('EXAMPLE')].reset_index(drop=True)
    b = result[result.ticker.eq('OTHER')].reset_index(drop=True)
    assert a.growth_latest_pct.tolist() == b.growth_latest_pct.tolist()
    assert (b.log_revenue > a.log_revenue).all()


def test_split_respects_label_availability():
    assert assign_split('2023-10-01','2024-01-15') == 'boundary_excluded'
    assert assign_split('2024-10-01','2025-01-15') == 'boundary_excluded'
    assert assign_split('2023-04-01','2023-07-01') == 'train'
    assert assign_split('2024-04-01','2024-07-01') == 'validation'
    assert assign_split('2025-04-01','2025-07-01') == 'test'
    assert assign_split('2026-10-01',None) == 'unscored'


def test_missing_quarter_is_rejected():
    data = panel().drop(index=4)
    with pytest.raises(ValueError, match='Missing or irregular'):
        build_features(data)


def test_late_filing_forecast_is_excluded_and_explained():
    data = panel()
    data.loc[7, 'filed'] = data.loc[8, 'filed']
    result = build_features(data)
    assert data.loc[7, 'end'] not in result.source_quarter_end.tolist()
    assert len(result.attrs['excluded_forecasts']) == 1


def test_refresh_keeps_prior_facts_and_is_idempotent(tmp_path, monkeypatch):
    import json
    import data

    ticker, cik = "MSFT", "0000789019"

    def snapshot(facts):
        return json.dumps({"cik": int(cik), "facts": {"us-gaap": {
            TAG: {"units": {"USD": facts}}
        }}}).encode()

    facts = []
    for i, end in enumerate(pd.date_range("2022-03-31", periods=10, freq=pd.offsets.QuarterEnd())):
        facts.append(dict(start=end.to_period("Q").start_time.date().isoformat(),
                          end=end.date().isoformat(),
                          filed=(end + pd.Timedelta(days=25)).date().isoformat(),
                          val=100 + 10 * i, accn=f"original-{i}", form="10-Q"))
    original = snapshot(facts)
    revision = snapshot([{**facts[5], "filed": "2024-02-01", "val": 155, "accn": "later"}])
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "fetch_snapshot", lambda cik: original)
    before, problems, _ = data.load_versions(ticker, cik, download=True)
    assert not problems
    monkeypatch.setattr(data, "fetch_snapshot", lambda cik: revision)
    updated, problems, status = data.load_versions(ticker, cik, refresh=True)
    assert not problems
    assert status == "new_snapshot"
    is_revision = updated.accession.eq("later")
    assert updated.loc[is_revision, "revenue_usd"].tolist() == [155]
    pd.testing.assert_frame_equal(before, updated.loc[~is_revision].reset_index(drop=True))
    prior = build_features(before)
    refreshed = build_features(updated)
    historical = prior[prior.forecast_as_of.lt("2024-02-01")].reset_index(drop=True)
    assert len(historical) == 1
    pd.testing.assert_frame_equal(
        historical, refreshed[refreshed.forecast_as_of.lt("2024-02-01")].reset_index(drop=True)
    )
    assert prior.iloc[-1].growth_latest_pct != refreshed.iloc[-1].growth_latest_pct
    replayed, problems, status = data.load_versions(ticker, cik, refresh=True)
    assert not problems
    assert status == "unchanged"
    pd.testing.assert_frame_equal(updated, replayed)


def test_backtest_excludes_labels_unavailable_at_forecast():
    from backtest import backtest

    source = panel()
    ends = pd.date_range("2024-03-01", periods=12, freq="13W-FRI")
    for i, end in enumerate(ends, start=len(source)):
        source.loc[i] = ("EXAMPLE", end.strftime("%Y-%m-%d"),
                         (end + pd.Timedelta(days=25)).strftime("%Y-%m-%d"),
                         100 + i * 10)
    features = build_features(source)
    start = features.iloc[10].forecast_as_of
    forecast = backtest(features, start=start).iloc[0]
    unavailable = features.target_filed.gt(start)
    modified = features.copy()
    modified.loc[unavailable, "target_growth_pct"] += 100
    replayed = backtest(modified, start=start).iloc[0]
    assert forecast.latest_training_label_filed <= start
    assert forecast.train_rows == (features.forecast_as_of.lt(start)
                                   & features.target_filed.le(start)).sum()
    assert forecast.linear == replayed.linear
    assert forecast.target_growth_pct != replayed.target_growth_pct

