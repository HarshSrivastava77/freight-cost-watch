import math

import numpy as np
import pandas as pd
import pytest

from fcw import costs
from fcw.output import fmt_cost, fmt_vs_own, fmt_vs_peer


@pytest.fixture(scope="module")
def weekly():
    return costs.build_weekly_table()


def _row(weekly, route, week):
    r = weekly[(weekly.route == route) & (weekly.week_of == week)]
    assert len(r) == 1
    return r.iloc[0]


@pytest.mark.parametrize("route,week,cost,own,peer", [
    ("Delhi-Jaipur", "2024-11-11", "4.17", "+35.5%", "+21.0%"),
    ("Ahmedabad-Mumbai", "2025-01-20", "3.29", "+29.5%", "+22.5%"),
    ("Mumbai-Pune", "2025-09-15", "3.98", "+9.2%", "+23.6%"),
])
def test_reproduces_sample_rows(weekly, route, week, cost, own, peer):
    r = _row(weekly, route, week)
    assert fmt_cost(r.cost_per_tonne_km) == cost
    assert fmt_vs_own(r.vs_own_pct).startswith(own + " ")
    assert fmt_vs_peer(r.vs_peer_pct).startswith(peer + " ")
    assert r.rule_flagged


def _shipments(rows):
    """rows: (origin, destination, route_type, tonnes, km, cost, date)"""
    df = pd.DataFrame(rows, columns=["origin", "destination", "route_type", "quantity_tonnes",
                                     "distance_km", "freight_cost_inr", "shipment_date"])
    return df


def _load(tmp_path, rows):
    p = tmp_path / "s.csv"
    _shipments(rows).to_csv(p, index=False)
    return costs.load_shipments(p)


def test_week_of_is_monday_and_whitespace_stripped(tmp_path):
    df = _load(tmp_path, [
        (" Mumbai ", "Pune ", " Short", 10, 100, 1000, "2025-03-16"),  # Sunday
        ("Mumbai", "Pune", "Short", 10, 100, 1000, "2025-03-17"),      # Monday
    ])
    assert list(df.week_of) == ["2025-03-10", "2025-03-17"]
    assert set(df.route) == {"Mumbai-Pune"} and set(df.route_type) == {"Short"}


def test_weekly_ratio_is_sum_over_sum_not_mean_of_ratios(tmp_path):
    df = _load(tmp_path, [
        ("A", "B", "Short", 1, 100, 1000, "2025-01-06"),   # ratio 10.0 on 100 t-km
        ("A", "B", "Short", 10, 100, 3000, "2025-01-07"),  # ratio 3.0 on 1000 t-km
    ])
    w = costs.weekly_costs(df)
    assert w.cost_per_tonne_km.iloc[0] == pytest.approx(4000 / 1100)  # not (10 + 3) / 2


def _series_table(values_by_route, route_type="Short", start="2025-01-06"):
    rows = []
    for route, vals in values_by_route.items():
        for i, v in enumerate(vals):
            if v is None:
                continue
            rows.append({"route": route, "route_type": route_type,
                         "week_of": (pd.Timestamp(start) + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
                         "cost_per_tonne_km": v})
    return pd.DataFrame(rows)


def test_own_baseline_first_week_nan_and_fewer_than_8_weeks():
    t = costs.add_own_baseline(_series_table({"A-B": [1, 2, 3, 4]}))
    assert math.isnan(t.own_baseline.iloc[0]) and t.own_weeks_used.iloc[0] == 0
    assert t.own_baseline.iloc[1] == 1 and t.own_weeks_used.iloc[1] == 1
    assert t.own_baseline.iloc[3] == pytest.approx(2.0) and t.own_weeks_used.iloc[3] == 3


def test_own_baseline_uses_exactly_8_prior_weeks():
    vals = list(range(1, 12))  # 1..11
    t = costs.add_own_baseline(_series_table({"A-B": vals}))
    # week index 10 (value 11) -> mean of values at weeks 2..9 = 3..10
    assert t.own_baseline.iloc[10] == pytest.approx(np.mean(range(3, 11)))
    assert t.own_weeks_used.iloc[10] == 8


def test_no_look_ahead():
    base = _series_table({"A-B": [1, 2, 3, 4, 5, 6]})
    spiked = _series_table({"A-B": [1, 2, 3, 4, 5, 600]})
    a = costs.add_own_baseline(base).own_baseline
    b = costs.add_own_baseline(spiked).own_baseline
    pd.testing.assert_series_equal(a, b)  # changing the last week never changes any baseline


def test_missing_week_counts_as_a_calendar_week():
    # weeks: 0..9 present except week 1; at week 9 the 8-week window is weeks 1..8
    vals = [100, None, 2, 2, 2, 2, 2, 2, 2, 5]
    t = costs.add_own_baseline(_series_table({"A-B": vals}))
    last = t.iloc[-1]
    assert last.own_baseline == pytest.approx(2.0)  # week 0 (100) is outside the window
    assert last.own_weeks_used == 7


def test_peer_baseline_excludes_self_and_respects_route_type():
    t = pd.concat([
        _series_table({"A-B": [2.0], "C-D": [4.0], "E-F": [6.0]}, route_type="Medium"),
        _series_table({"X-Y": [100.0]}, route_type="Long"),
    ])
    t = costs.add_peer_baseline(t).set_index("route")
    assert t.loc["A-B", "peer_baseline"] == pytest.approx(5.0)
    assert t.loc["C-D", "peer_baseline"] == pytest.approx(4.0)
    assert math.isnan(t.loc["X-Y", "peer_baseline"])  # no other Long route


def test_flag_rule(weekly):
    expected = (weekly.vs_own_pct > 0) & ((weekly.vs_own_pct >= 15) | (weekly.vs_peer_pct >= 20))
    assert (weekly.rule_flagged == expected).all()
    assert not weekly[weekly.own_weeks_used == 0].rule_flagged.any()


def test_every_route_has_complete_weeks(weekly):
    assert weekly.groupby("route").size().nunique() == 1
    assert weekly.groupby("route")["route_type"].nunique().max() == 1
