"""Download eight companies' SEC facts and save quarterly revenue to Parquet.

Run: python data.py --download   (downloads only missing files)
     python data.py --refresh    (fetches and archives new snapshots)
     python data.py              (uses cached files)
"""
import argparse
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent
TAG = "RevenueFromContractWithCustomerExcludingAssessedTax"
TAGS = (TAG, "SalesRevenueNet", "Revenues")
COMPANIES = {
    "MSFT": ("Microsoft", "0000789019"),
    "CRM": ("Salesforce", "0001108524"),
    "NOW": ("ServiceNow", "0001373715"),
    "ORCL": ("Oracle", "0001341439"),
    "WDAY": ("Workday", "0001327811"),
    "ADSK": ("Autodesk", "0000769397"),
    "INTU": ("Intuit", "0000896878"),
    "ADBE": ("Adobe", "0000796343"),
}


# 1. Read source facts and normalize quarterly revenue.
def extract_observations(company):
    """Preserve filing versions and source tags before selecting observations."""
    rows = []
    concepts = company["facts"]["us-gaap"]
    for priority, tag in enumerate(TAGS):
        for obs in concepts.get(tag, {}).get("units", {}).get("USD", []):
            if not all(key in obs for key in ("start", "end", "filed", "val", "accn", "form")):
                continue
            start, end = date.fromisoformat(obs["start"]), date.fromisoformat(obs["end"])
            filed = date.fromisoformat(obs["filed"])
            if end < start or filed < end:
                raise ValueError(f"Invalid dates in {tag}: {obs}")
            if obs["val"] < 0:
                raise ValueError(f"Negative revenue in {tag}: {obs}")
            rows.append({
                "concept": tag, "priority": priority,
                "start": start.isoformat(), "end": end.isoformat(),
                "duration_days": (end - start).days + 1,
                "revenue_usd": obs["val"], "filed": filed.isoformat(),
                "form": obs["form"], "accession": obs["accn"],
            })
    if not rows:
        raise ValueError("No supported USD revenue observations found")
    return rows


def build_quarterly_versions(source, ticker="MSFT", cik="0000789019", concept=TAG):
    # Keep one accounting concept to avoid mixing historical definitions.
    source = source[source.concept.eq(concept)].copy()
    direct = source[source.duration_days.between(80, 100)].copy()
    direct["method"] = "reported"
    direct["nine_month_accession"] = ""
    direct["nine_month_filed"] = ""
    direct["annual_revenue_usd"] = pd.NA
    direct["nine_month_revenue_usd"] = pd.NA
    derived = []
    for annual in source[source.duration_days.between(360, 380)].to_dict("records"):
        # Use the latest nine-month version available by the annual filing.
        nine = source[
            source.start.eq(annual["start"])
            & source.duration_days.between(260, 285)
            & source.filed.le(annual["filed"])
        ].sort_values(["filed", "accession"])
        if nine.empty:
            continue
        previous = nine.iloc[-1]
        start = pd.Timestamp(previous.end) + pd.Timedelta(days=1)
        days = (pd.Timestamp(annual["end"]) - start).days + 1
        value = annual["revenue_usd"] - previous.revenue_usd
        if not 80 <= days <= 100 or value <= 0:
            raise ValueError("Invalid derived Q4; review source figures")
        derived.append({**annual, "start": start.strftime("%Y-%m-%d"),
                        "duration_days": days, "revenue_usd": value,
                        "method": "annual_minus_nine_months",
                        "nine_month_accession": previous.accession,
                        "nine_month_filed": previous.filed,
                        "annual_revenue_usd": annual["revenue_usd"],
                        "nine_month_revenue_usd": previous.revenue_usd})
    versions = pd.concat([direct, pd.DataFrame(derived)], ignore_index=True)
    # Prefer directly reported values on the same filing date.
    versions["derived"] = versions.method.ne("reported")
    versions = versions.sort_values(["end", "filed", "derived", "accession"])
    versions = versions.drop_duplicates(["end", "filed"], keep="first")
    versions["ticker"] = ticker
    versions["cik"] = cik
    return versions.drop(columns="derived").reset_index(drop=True)


# 2. Collect each company and check the combined dataset.
def parse_snapshot(raw, ticker, cik):
    company = json.loads(raw)
    if int(company["cik"]) != int(cik):
        raise ValueError(f"Snapshot CIK mismatch for {ticker}")
    return company


def fetch_snapshot(cik):
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        raise ValueError("Set SEC_USER_AGENT to the project name and contact email")
    request = Request(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                      headers={"User-Agent": user_agent})
    time.sleep(0.5)
    with urlopen(request, timeout=60) as response:
        return response.read()


def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(raw)
            temporary.close()
            temp_path.replace(path)
        finally:
            temporary.close()
            temp_path.unlink(missing_ok=True)


def load_versions(ticker, cik, download=False, refresh=False):
    """Combine archived facts before deriving quarters. Save only validated downloads."""
    folder = ROOT / "data" / "raw"
    base = folder / f"{ticker.lower()}_companyfacts.json"
    archive = folder / "snapshots" / ticker.lower()
    paths = sorted(archive.glob("*.json"))
    if base.exists():
        paths.insert(0, base)
    if not paths and not (download or refresh):
        raise FileNotFoundError(f"Missing {ticker} snapshot; run with --download")

    snapshots = {hashlib.sha256(raw).hexdigest(): raw for raw in
                 (path.read_bytes() for path in paths)}
    new_path = None
    fetch_status = "cached"
    if refresh or not snapshots:
        raw = fetch_snapshot(cik)
        digest = hashlib.sha256(raw).hexdigest()
        fetch_status = "unchanged"
        if digest not in snapshots:
            new_path = archive / f"{digest}.json" if refresh else base
            snapshots[digest] = raw
            fetch_status = "new_snapshot"

    companies = [parse_snapshot(raw, ticker, cik) for raw in snapshots.values()]
    versions = prepare_company(companies[0], ticker, cik, archives=companies[1:])
    problems = validate_company(versions)
    if new_path is not None and not problems:
        atomic_write(new_path, raw)
    return versions, problems, fetch_status


def validate_company(versions):
    """Report structural problems without pretending they establish comparability."""
    first = versions.sort_values(["end", "filed"]).drop_duplicates("end")
    problems = []
    if first.empty:
        return ["No supported quarterly observations"]
    if versions.duplicated(["ticker", "end", "filed"]).any():
        problems.append("Duplicate company/period/filing keys")
    if (versions.revenue_usd <= 0).any():
        problems.append("Nonpositive quarterly revenue")
    if (versions.filed < versions.end).any():
        problems.append("Filing precedes period end")
    # Use actual dates: Adobe's week-based calendar cannot use calendar-quarter ends.
    ends = pd.to_datetime(first.end)
    gaps = ends.diff().dt.days
    for end in first.loc[(gaps < 70) | (gaps > 110), "end"]:
        problems.append(f"Irregular quarter spacing before {end}")
    starts = pd.to_datetime(first.start)
    for end in first.loc[(starts - ends.shift()).dt.days.lt(0), "end"]:
        problems.append(f"Overlapping periods ending {end}")
    return problems


def prepare_company(company, ticker, cik, archives=()):
    observations = [obs for snapshot in (company, *archives)
                    for obs in extract_observations(snapshot)]
    source = pd.DataFrame(observations).drop_duplicates()
    concept = "Revenues" if ticker == "ADBE" else TAG
    source = source[source.concept.eq(concept) & source.form.isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])]
    if source.empty:
        raise ValueError("Selected revenue concept unavailable; review required")
    # Detect conflicts before the shared preparation function chooses a version.
    conflicts = source.groupby(["start", "end", "filed"]).revenue_usd.nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting revenue values on the same filing date")
    versions = build_quarterly_versions(source, ticker=ticker, cik=cik, concept=concept)
    versions = versions[versions.end.ge("2019-01-01")].reset_index(drop=True)
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Download missing SEC snapshots")
    mode.add_argument("--refresh", action="store_true", help="Fetch and archive updated SEC snapshots")
    args = parser.parse_args()
    output = ROOT / "data" / "processed"
    panel_path = output / "software_quarterly_versions.parquet"
    key = ["ticker", "end", "filed"]
    previous = (pd.read_parquet(panel_path, columns=key) if panel_path.exists()
                else pd.DataFrame(columns=key))
    previous_keys = set(previous.itertuples(index=False, name=None))
    reports, panels = [], []
    for ticker, (name, cik) in COMPANIES.items():
        report = dict(ticker=ticker, company=name, status="needs_review", quarters=0)
        try:
            versions, problems, fetch_status = load_versions(ticker, cik, args.download, args.refresh)
            version_keys = set(versions[key].itertuples(index=False, name=None))
            first = versions.sort_values(["end", "filed"]).drop_duplicates("end")
            report.update(quarters=len(first), earliest=first.end.min(), latest=first.end.max(),
                          snapshot_status=fetch_status, filing_versions=len(versions),
                          added_filing_versions=len(version_keys - previous_keys),
                          derived_quarters=int(first.method.ne("reported").sum()),
                          issues=problems, status="structural_checks_passed" if not problems else "needs_review")
            panels.append(versions)
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            report["issues"] = [str(exc)]
        reports.append(report)
        print(ticker, report["status"], report["quarters"], report.get("issues", []))
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "software_panel_review.json"
    report_path.write_text(json.dumps({"companies": reports,
        "notes": ["No model training performed. Structural checks are not accounting validation.",
                  "Current surviving-company sample introduces selection bias.",
                  "Review fiscal calendars and derived Q4 inputs before building features."]}, indent=2))
    if len(panels) != len(COMPANIES) or any(r["status"] != "structural_checks_passed" for r in reports):
        raise SystemExit("Panel not published: resolve the issues in software_panel_review.json first.")
    panel = pd.concat(panels, ignore_index=True).sort_values(["ticker", "end", "filed"])
    atomic_write(panel_path, panel.to_parquet(index=False))
    print(f"Saved {len(panel)} filing versions. Review required before model training.")


if __name__ == "__main__":
    main()
