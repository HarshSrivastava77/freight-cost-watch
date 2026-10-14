"""Regenerate the README charts from output.csv.

    pip install matplotlib
    python docs/make_charts.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from fcw.costs import build_weekly_table

OUT = str(ROOT / "docs" / "images") + "/"
out = pd.read_csv(ROOT / "output.csv", dtype=str, keep_default_na=False)
w = build_weekly_table().merge(out[["route", "week_of", "flagged", "matched_note_id"]], on=["route", "week_of"])
w["date"] = pd.to_datetime(w["week_of"])

GREEN, RED, BLUE, GREY, ORANGE = "#2e7d32", "#c62828", "#1f4e79", "#7f7f7f", "#e08a00"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})

def mark_flags(ax, d):
    j = d[d.flagged == "No (justified)"]; u = d[d.flagged == "Yes"]
    if len(j):
        ax.scatter(j.date, j.cost_per_tonne_km, s=90, color=GREEN, marker="o", zorder=5, label="Flagged, justified by a note")
    if len(u):
        ax.scatter(u.date, u.cost_per_tonne_km, s=110, color=RED, marker="X", zorder=5, label="Flagged, unexplained")

def fmt_dates(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))

# 1. Chennai-Bangalore flood
d = w[(w.route == "Chennai-Bangalore") & (w.date >= "2025-01-06") & (w.date <= "2025-04-28")]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axvspan(pd.Timestamp("2025-02-24"), pd.Timestamp("2025-03-08") + pd.Timedelta(days=1), color=BLUE, alpha=0.10,
           label="N001 flood window (24 Feb - 8 Mar)")
ax.plot(d.date, d.cost_per_tonne_km, color=BLUE, lw=2, marker=".", label="Weekly cost / tonne-km")
ax.plot(d.date, d.own_baseline, color=GREY, lw=1.5, ls="--", label="Own 8-week baseline")
ax.plot(d.date, d.peer_baseline, color=ORANGE, lw=1.5, ls=":", label="Peer average (other Medium routes)")
mark_flags(ax, d)
for _, r in d[d.flagged != "No"].iterrows():
    txt = r.matched_note_id if r.flagged == "No (justified)" else "unexplained"
    ax.annotate(txt, (r.date, r.cost_per_tonne_km), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8.5,
                color=GREEN if r.flagged != "Yes" else RED)
ax.set_title("Chennai-Bangalore: the flood note covers two flagged weeks, not four")
ax.set_ylabel("INR per tonne-km"); fmt_dates(ax)
ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
ax.set_ylim(d.cost_per_tonne_km.min() * 0.92, d.cost_per_tonne_km.max() * 1.12)
fig.tight_layout(); fig.savefig(OUT + "chennai_bangalore_flood.png", dpi=150); plt.close(fig)

# 2. Mumbai-Pune peer gap
mp = w[(w.route == "Mumbai-Pune") & (w.date >= "2025-01-06")]
dj = w[(w.route == "Delhi-Jaipur") & (w.date >= "2025-01-06")]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axvline(pd.Timestamp("2025-05-05"), color=GREY, lw=1, ls="-.")
ax.text(pd.Timestamp("2025-05-09"), 4.15, "N003 diesel rise starts\n(all routes, ~5-7%):\nlifts the peer too, so it\ncannot explain the gap", fontsize=8.5, color=GREY, va="top")
ax.plot(mp.date, mp.cost_per_tonne_km, color=BLUE, lw=2, label="Mumbai-Pune weekly cost")
ax.plot(mp.date, mp.own_baseline, color=GREY, lw=1.5, ls="--", label="Mumbai-Pune own 8-week baseline")
ax.plot(dj.date, dj.cost_per_tonne_km, color=ORANGE, lw=1.5, ls=":", label="Delhi-Jaipur (its only Short peer)")
mark_flags(ax, mp)
ax.set_title("Mumbai-Pune: slow creep vs its own history, 20-39% above its peer")
ax.set_ylabel("INR per tonne-km"); fmt_dates(ax)
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
ax.set_ylim(2.9, 4.6)
fig.tight_layout(); fig.savefig(OUT + "mumbai_pune_peer_gap.png", dpi=150); plt.close(fig)

# 3. Overview of all flags
routes = sorted(w.route.unique())
fig, ax = plt.subplots(figsize=(10, 3.8))
for i, r in enumerate(routes):
    ax.axhline(i, color="#e6e6e6", lw=1, zorder=0)
f = w[w.flagged != "No"].copy(); f["y"] = f.route.map({r: i for i, r in enumerate(routes)})
j = f[f.flagged == "No (justified)"]; u = f[f.flagged == "Yes"]
ax.scatter(j.date, j.y, s=90, color=GREEN, marker="o", label=f"Justified ({len(j)})", zorder=3)
ax.scatter(u.date, u.y, s=100, color=RED, marker="X", label=f"Unexplained ({len(u)})", zorder=3)
for _, r in j.drop_duplicates(["route", "matched_note_id"]).iterrows():
    ax.annotate(r.matched_note_id, (r.date, r.y), xytext=(4, 9), textcoords="offset points", ha="center", fontsize=8, color=GREEN)
ax.set_yticks(range(len(routes))); ax.set_yticklabels(routes)
ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_title(f"All {len(f)} flagged route-weeks out of {len(w)} (2024-2025)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=9, frameon=True, facecolor="white", edgecolor="#dddddd")
ax.spines["left"].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "flags_overview.png", dpi=150); plt.close(fig)
print(f"wrote 3 charts to {OUT} ({len(f)} flags: {len(j)} justified, {len(u)} unexplained)")
