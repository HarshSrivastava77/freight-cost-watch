"""Core numeric pipeline: shipments -> weekly cost per tonne-km -> baselines -> rule flag.

Everything here is pure pandas/numpy and fully deterministic. No AI involved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

REQUIRED_COLUMNS = [
    "origin", "destination", "route_type", "quantity_tonnes",
    "distance_km", "freight_cost_inr", "shipment_date",
]


def load_shipments(path: Path = config.SHIPMENTS_CSV) -> pd.DataFrame:
    """Read shipments and derive `route` and `week_of` (Monday of the shipment's week)."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"shipment file is missing columns: {sorted(missing)}")

    for col in ("origin", "destination", "route_type"):
        df[col] = df[col].astype(str).str.strip()
    df["route"] = df["origin"] + "-" + df["destination"]

    dates = pd.to_datetime(df["shipment_date"].astype(str).str.strip(), format="%Y-%m-%d")
    monday = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    df["week_of"] = monday.dt.strftime("%Y-%m-%d")
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def weekly_costs(shipments: pd.DataFrame) -> pd.DataFrame:
    """Weekly cost per tonne-km per (route, route_type, week_of).

    Uses sum(cost) / sum(tonne*km) -- a tonne-km-weighted figure -- NOT the mean of
    per-shipment ratios, so one tiny shipment cannot swing the week.
    """
    weekly = (
        shipments.groupby(["route", "route_type", "week_of"], as_index=False, sort=True)
        .agg(
            total_cost_inr=("freight_cost_inr", "sum"),
            tonne_km=("tonne_km", "sum"),
            n_shipments=("freight_cost_inr", "size"),
        )
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost_inr"] / weekly["tonne_km"]
    return weekly


def _own_baseline(series: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean over the previous `window` calendar weeks, excluding the current week.

    The series is reindexed onto a complete weekly calendar first so that a missing
    week counts as a week (shift(1) == "the previous calendar week", not "the previous
    row"). With min_periods=1, weeks with fewer than `window` prior weeks use all the
    prior weeks available; the very first week has none and gets NaN.
    """
    idx = pd.to_datetime(series.index)
    s = pd.Series(series.to_numpy(), index=idx)
    calendar = pd.date_range(idx.min(), idx.max(), freq="7D")
    prior = s.reindex(calendar).shift(1)
    baseline = prior.rolling(window, min_periods=1).mean().reindex(idx)
    weeks_used = prior.rolling(window, min_periods=1).count().reindex(idx)
    return baseline.to_numpy(), weeks_used.to_numpy()


def add_own_baseline(weekly: pd.DataFrame, window: int = config.OWN_HISTORY_WEEKS) -> pd.DataFrame:
    out = weekly.sort_values(["route", "route_type", "week_of"]).reset_index(drop=True)
    out["own_baseline"] = np.nan
    out["own_weeks_used"] = 0
    for _, idx in out.groupby(["route", "route_type"], sort=True).groups.items():
        g = out.loc[idx]
        base, used = _own_baseline(g.set_index("week_of")["cost_per_tonne_km"], window)
        out.loc[idx, "own_baseline"] = base
        out.loc[idx, "own_weeks_used"] = np.nan_to_num(used).astype(int)
    out["own_weeks_used"] = out["own_weeks_used"].astype(int)
    return out


def add_peer_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    """Simple mean of OTHER routes' weekly cost, same route_type, same week."""
    out = weekly.reset_index(drop=True)  # group indices below must be unique row labels
    out["peer_baseline"] = np.nan
    out["n_peers"] = 0
    for _, idx in out.groupby(["route_type", "week_of"], sort=True).groups.items():
        values = out.loc[idx, "cost_per_tonne_km"].to_numpy()
        n = len(values)
        peer = [values[np.arange(n) != i].mean() if n > 1 else np.nan for i in range(n)]
        out.loc[idx, "peer_baseline"] = peer
        out.loc[idx, "n_peers"] = n - 1
    out["n_peers"] = out["n_peers"].astype(int)
    return out


def pct_change(value: pd.Series, baseline: pd.Series) -> pd.Series:
    """(value / baseline - 1) * 100 on unrounded numbers; NaN when no baseline."""
    return (value / baseline - 1.0) * 100.0


def add_flags(
    weekly: pd.DataFrame,
    own_threshold: float = config.OWN_RISE_FLAG_PCT,
    peer_threshold: float = config.PEER_GAP_FLAG_PCT,
) -> pd.DataFrame:
    out = weekly.copy()
    out["vs_own_pct"] = pct_change(out["cost_per_tonne_km"], out["own_baseline"])
    out["vs_peer_pct"] = pct_change(out["cost_per_tonne_km"], out["peer_baseline"])
    # NaN comparisons evaluate False, so a route's first week (no history) is never flagged.
    out["rising"] = out["vs_own_pct"] > 0
    out["own_trigger"] = out["vs_own_pct"] >= own_threshold
    out["peer_trigger"] = out["vs_peer_pct"] >= peer_threshold
    out["rule_flagged"] = out["rising"] & (out["own_trigger"] | out["peer_trigger"])
    return out


def peer_routes(weekly: pd.DataFrame) -> dict[str, list[str]]:
    """route -> other routes with the same route_type."""
    pairs = weekly[["route", "route_type"]].drop_duplicates()
    by_type = pairs.groupby("route_type")["route"].apply(sorted).to_dict()
    return {r: [p for p in by_type[t] if p != r] for r, t in pairs.itertuples(index=False)}


def build_weekly_table(path: Path = config.SHIPMENTS_CSV) -> pd.DataFrame:
    weekly = weekly_costs(load_shipments(path))
    weekly = add_own_baseline(weekly)
    weekly = add_peer_baseline(weekly)
    weekly = add_flags(weekly)
    return weekly.sort_values(["route", "week_of"]).reset_index(drop=True)
