#!/usr/bin/env python3
"""Figures for docs/report/clashbot_report.tex, built only from docs/data outputs.

Run from the repo root:
    ../.venv/bin/python docs/report/make_figures.py
    ../.venv/bin/python docs/report/make_figures.py --extra-data <dir>   # look here too when a file
                                                                          # is not yet merged into docs/data
Writes docs/report/figs/{fig_popularity,fig_retrieval_selective}.{pdf,png}.
The dashboard screenshot (figs/dashboard.png) is copied in by hand, not generated.

Sources (every plotted number is read from these files, nothing is typed in):
  fig_popularity           docs/data/stats_audit_v1.json  -> popularity.accuracy.test, popularity.n.test
  fig_retrieval_selective  docs/data/retrieval_ablation_v1.json -> hnsw
                           docs/data/stats_audit_v1_risk_coverage.csv (d1_tac val/test)
                           docs/data/stats_audit_v1.json  -> selective (threshold, operating point)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figs"

# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light mode);
# grey = reference / baseline (not a categorical slot). Aqua is below 3:1 contrast, so every
# series also has a distinct marker and a legend entry (secondary encoding).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


def find(name: str, extra: list[Path]) -> Path:
    for d in [REPO / "docs" / "data", *extra]:
        p = d / name
        if p.exists():
            return p
    sys.exit(f"missing input {name}: looked in docs/data and {', '.join(map(str, extra)) or '-'}")


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{stem}.pdf")


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_popularity(audit: dict) -> None:
    """Test accuracy by popularity tercile of the gold key (train counts), 95% cluster-bootstrap CI."""
    acc = audit["popularity"]["accuracy"]["test"]
    n = audit["popularity"]["n"]["test"]
    series = [  # (key in stats_audit, label, colour, marker)
        ("popularity", "popularity (train counts)", GREY, "s"),
        ("placement_prior", "placement prior", AQUA, "^"),
        ("lgbm", "LightGBM LambdaRank", BLUE, "o"),
        ("d1_tac", "d1-ft-joint-tac (live)", ORANGE, "D"),
    ]
    terciles = ["low", "mid", "high"]
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    width = 0.17
    for i, (key, label, colour, marker) in enumerate(series):
        xs, ys, lo, hi = [], [], [], []
        for t, cell in enumerate(acc[key]):
            xs.append(t + (i - (len(series) - 1) / 2) * width)
            ys.append(cell["est"])
            lo.append(cell["est"] - cell["lo"])
            hi.append(cell["hi"] - cell["est"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=marker, color=colour, ms=4.2, mec="white",
                    mew=0.6, elinewidth=1.0, capsize=0, label=label, zorder=3)
    ax.axhline(0.25, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.text(2.55, 0.255, "chance", color=INK2, fontsize=6, ha="right", va="bottom")
    edges = audit["popularity"]["edges"]
    ax.set_xticks(range(3))
    ax.set_xticklabels([
        f"low (count $\\leq$ {edges[0]:.0f})\nn = {n[0]}",
        f"mid ($\\leq$ {edges[1]:.0f})\nn = {n[1]}",
        f"high (> {edges[1]:.0f})\nn = {n[2]}",
    ])
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("test accuracy (4-way)")
    style(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8,
              borderaxespad=0.1)
    save(fig, "fig_popularity")


def read_risk(path: Path) -> dict:
    rows: dict[str, list[tuple[float, float]]] = {"val": [], "test": []}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["model"] != "d1_tac":
                continue
            rows[r["split"]].append((float(r["coverage"]), float(r["accuracy"])))
    for k in rows:
        rows[k].sort()
    return rows


def fig_retrieval_selective(retr: dict, risk: dict, audit: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.55), gridspec_kw={"wspace": 0.5})
    # (a) HNSW recall vs latency (struct channel, k = 10, 10,923 vectors)
    pts = retr["hnsw"]
    approx = [p for p in pts if p["ef"] != "exact"]
    exact = [p for p in pts if p["ef"] == "exact"]
    a.plot([p["ms_p50"] for p in approx], [p["recall@10"] for p in approx], "-o", color=BLUE,
           lw=1.4, ms=3.8, mec="white", mew=0.5, label="HNSW", zorder=3)
    for p in approx:
        if p["ef"] in (10, 40, 160):
            a.annotate(f"ef {p['ef']}", (p["ms_p50"], p["recall@10"]), textcoords="offset points",
                       xytext=(3, -8), fontsize=5.8, color=INK2)
    if exact:
        e = exact[0]
        a.plot([e["ms_p50"]], [e["recall@10"]], "s", color=GREY, ms=4.2, mec="white", mew=0.5,
               label="exact", zorder=3)
    a.set_xscale("log")
    a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.3, 1.0]))
    a.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.1", "0.3", "1"]))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("ms per query (p50, log)")
    a.set_ylabel("recall@10 vs exact")
    a.set_ylim(0.87, 1.01)
    a.set_title("(a) HNSW vs exact", loc="left", pad=3)
    style(a)
    a.legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.1)

    # (b) risk-coverage of d1-ft-joint-tac: accuracy on answered lines vs coverage
    for split, colour in (("val", BLUE), ("test", ORANGE)):
        cov = [c for c, _ in risk[split]]
        acc = [x for _, x in risk[split]]
        b.plot(cov, acc, color=colour, lw=1.4, label=split, zorder=3)
        op = audit["selective"]["at_threshold"][split]
        b.plot([op["coverage"]["est"]], [op["accuracy_answered"]["est"]], "o", color=colour,
               ms=4.2, mec="white", mew=0.6, zorder=4)
    thr = audit["selective"]["threshold"]
    b.annotate(f"hold if p < {thr:.3f}", (audit["selective"]["at_threshold"]["test"]["coverage"]["est"],
               audit["selective"]["at_threshold"]["test"]["accuracy_answered"]["est"]),
               textcoords="offset points", xytext=(-6, -16), fontsize=5.8, color=INK2)
    b.axhline(0.25, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    b.set_xlim(0, 1.0)
    b.set_ylim(0.2, 1.02)
    b.set_xlabel("coverage (share answered)")
    b.set_ylabel("accuracy when answering")
    b.set_title("(b) d1 risk-coverage", loc="left", pad=3)
    style(b)
    b.legend(loc="upper right", frameon=False, handletextpad=0.3, borderaxespad=0.1)
    save(fig, "fig_retrieval_selective")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extra-data", action="append", default=[], type=Path,
                    help="extra directory searched after docs/data (repeatable)")
    args = ap.parse_args()
    audit = json.loads(find("stats_audit_v1.json", args.extra_data).read_text())
    retr = json.loads(find("retrieval_ablation_v1.json", args.extra_data).read_text())
    risk = read_risk(find("stats_audit_v1_risk_coverage.csv", args.extra_data))
    fig_popularity(audit)
    fig_retrieval_selective(retr, risk, audit)


if __name__ == "__main__":
    main()
