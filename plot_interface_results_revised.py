#!/usr/bin/env python3
"""
Revised plotting script for binder-target interface analysis outputs.

Input CSVs expected in --input-dir:
    model_summary.csv
    interface_contacts.csv
    hotspot_contacts.csv   optional / can be empty
    errors.csv             optional / can be empty

Main outputs:
    00_design_metric_overview_absolute.png
    01_design_metric_overview_normalized.png
    02_heuristic_scan_score.png
    03_hotspot_contact_count_matrix.png
    04_hotspot_min_distance_matrix.png
    05_target_interface_burden.png
    figure_legend_notes.txt

Per-design outputs:
    per_design/<design_label>/contact_map.png
    per_design/<design_label>/hotspot_contact_map.png
    per_design/<design_label>/top_contacts_table.png

The script is HPC-safe: it uses the non-interactive Agg backend and saves PNGs only.

Example:
    conda activate your_env_name

    python plot_interface_results_revised.py \
        --input-dir ./interface_results \
        --outdir ./interface_plots_revised \
        --top-n-contacts 25 \
        --dpi 220
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV. Return an empty dataframe if missing or empty."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:180]


def savefig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def choose_fig_height(n_rows: int, base: float = 3.0, per_row: float = 0.45, max_h: float = 18.0) -> float:
    return min(max_h, max(base, base + per_row * max(n_rows, 1)))


def wrap_labels(labels: Iterable[str], width: int = 20) -> list[str]:
    return ["\n".join(textwrap.wrap(str(x), width=width, break_long_words=False)) for x in labels]


def as_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def as_bool(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            if df[col].dtype == bool:
                continue
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    return df


def add_design_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create compact labels from config/model/id when available; otherwise use file stem."""
    if df.empty:
        return df
    df = df.copy()

    def label_row(row: pd.Series) -> str:
        cfg = row.get("config_idx", np.nan)
        model = row.get("model_idx", np.nan)
        did = row.get("design_id", np.nan)
        if pd.notna(cfg) and pd.notna(model) and pd.notna(did):
            try:
                return f"config{int(cfg)}_model{int(model)}_id{int(did)}"
            except Exception:
                pass
        return Path(str(row.get("pdb_file", "design"))).stem

    df["design_label"] = df.apply(label_row, axis=1)
    return df


def metric_label_map() -> dict[str, str]:
    return {
        "n_residue_contacts": "Residue-pair\ncontacts",
        "n_target_interface_residues": "Target interface\nresidues",
        "n_binder_interface_residues": "Binder interface\nresidues",
        "n_target_hotspot_residues_contacted": "Hotspot residues\ncontacted",
        "n_hotspot_residue_contacts": "Hotspot-binder\nresidue pairs",
        "n_putative_polar_contacts_residue_pairs": "Putative polar\nresidue pairs",
        "n_putative_salt_bridge_residue_pairs": "Putative salt-bridge\nresidue pairs",
        "min_interface_distance_A": "Minimum interface\ndistance, Å",
    }


def normalize_columns(df: pd.DataFrame, invert_cols: set[str] | None = None) -> pd.DataFrame:
    """
    Min-max normalize each column to [0, 1].
    For columns in invert_cols, lower raw values become higher normalized values.
    """
    invert_cols = invert_cols or set()
    norm = df.copy().astype(float)
    for col in norm.columns:
        vals = norm[col].astype(float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            norm[col] = np.nan
        elif math.isclose(vmin, vmax):
            norm[col] = 0.5
        else:
            x = (vals - vmin) / (vmax - vmin)
            norm[col] = 1.0 - x if col in invert_cols else x
    return norm


# -----------------------------------------------------------------------------
# Aggregate plots
# -----------------------------------------------------------------------------

def plot_absolute_metric_overview(summary: pd.DataFrame, outdir: Path, dpi: int) -> None:
    """Raw values split into interpretable panels, avoiding mixed-scale confusion."""
    if summary.empty:
        return

    groups = [
        (
            "Interface size metrics",
            ["n_residue_contacts", "n_target_interface_residues", "n_binder_interface_residues"],
            "count",
        ),
        (
            "Hotspot engagement metrics",
            ["n_target_hotspot_residues_contacted", "n_hotspot_residue_contacts"],
            "count",
        ),
        (
            "Interaction-type metrics",
            ["n_putative_polar_contacts_residue_pairs", "n_putative_salt_bridge_residue_pairs"],
            "count",
        ),
        (
            "Closest approach metric",
            ["min_interface_distance_A"],
            "Å; lower means closer contact",
        ),
    ]

    label_map = metric_label_map()
    available_groups = []
    for title, cols, ylabel in groups:
        cols = [c for c in cols if c in summary.columns]
        if cols:
            available_groups.append((title, cols, ylabel))
    if not available_groups:
        return

    designs = list(summary["design_label"])
    x = np.arange(len(designs))
    fig_h = 3.3 * len(available_groups)
    fig, axes = plt.subplots(len(available_groups), 1, figsize=(max(10, len(designs) * 1.4), fig_h))
    if len(available_groups) == 1:
        axes = [axes]

    for ax, (title, cols, ylabel) in zip(axes, available_groups):
        width = min(0.8 / max(len(cols), 1), 0.35)
        offsets = (np.arange(len(cols)) - (len(cols) - 1) / 2.0) * width
        for idx, col in enumerate(cols):
            values = pd.to_numeric(summary[col], errors="coerce").fillna(0).values
            bars = ax.bar(x + offsets[idx], values, width=width, label=label_map.get(col, col))
            for bar, val in zip(bars, values):
                fmt = f"{val:.2f}" if col.endswith("_A") else f"{val:.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt,
                        ha="center", va="bottom", fontsize=8, rotation=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(wrap_labels(designs, width=24), rotation=25, ha="right")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Absolute design-level metrics from model_summary.csv", y=1.005, fontsize=14)
    savefig(outdir / "00_design_metric_overview_absolute.png", dpi=dpi)


def plot_normalized_metric_overview(summary: pd.DataFrame, outdir: Path, dpi: int) -> None:
    """Heatmap with each metric normalized across designs."""
    if summary.empty:
        return

    metric_cols = [
        "n_residue_contacts",
        "n_target_interface_residues",
        "n_binder_interface_residues",
        "n_target_hotspot_residues_contacted",
        "n_hotspot_residue_contacts",
        "n_putative_polar_contacts_residue_pairs",
        "n_putative_salt_bridge_residue_pairs",
        "min_interface_distance_A",
    ]
    metric_cols = [c for c in metric_cols if c in summary.columns]
    if not metric_cols:
        return

    df = summary.set_index("design_label")[metric_cols].apply(pd.to_numeric, errors="coerce")
    norm = normalize_columns(df, invert_cols={"min_interface_distance_A"})
    label_map = metric_label_map()
    xlabels = [label_map.get(c, c) for c in metric_cols]

    fig_w = max(10, 1.25 * len(metric_cols))
    fig_h = choose_fig_height(len(df), base=3.4, per_row=0.55)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(norm.values, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, label="relative value within this dataset: 0 = lowest, 1 = highest")
    plt.xticks(range(len(metric_cols)), wrap_labels(xlabels, width=18), rotation=35, ha="right")
    plt.yticks(range(len(df.index)), df.index)
    plt.title("Normalized design-level metric overview\nRaw values are printed in cells; color is normalized per column")

    for i in range(df.shape[0]):
        for j, col in enumerate(metric_cols):
            value = df.iloc[i, j]
            if pd.isna(value):
                txt = "NA"
            elif col == "min_interface_distance_A":
                txt = f"{value:.2f} Å"
            else:
                txt = f"{value:.0f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    savefig(outdir / "01_design_metric_overview_normalized.png", dpi=dpi)


def plot_heuristic_scan_score(summary: pd.DataFrame, outdir: Path, dpi: int) -> pd.DataFrame:
    """
    Create a transparent heuristic ranking based on normalized columns.
    Returns the score table.
    """
    if summary.empty:
        return pd.DataFrame()
    score_terms = {
        "n_target_hotspot_residues_contacted": 3.0,
        "n_hotspot_residue_contacts": 2.0,
        "n_putative_salt_bridge_residue_pairs": 1.5,
        "n_putative_polar_contacts_residue_pairs": 1.0,
        "n_residue_contacts": 0.5,
    }
    distance_col = "min_interface_distance_A"
    if distance_col in summary.columns:
        score_terms[distance_col] = 0.5
    df = summary.copy()
    for col in score_terms:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    raw = df[list(score_terms)].copy()
    norm = normalize_columns(raw, invert_cols={distance_col})
    score = pd.Series(0.0, index=df.index)
    for col, weight in score_terms.items():
        score += weight * norm[col].fillna(0.0)
    df["heuristic_scan_score"] = score
    df = df.sort_values("heuristic_scan_score", ascending=True)
    fig_h = choose_fig_height(len(df), base=3.5, per_row=0.55)
    fig, ax = plt.subplots(figsize=(12, fig_h))  # ← use subplots to get ax
    bars = ax.barh(df["design_label"], df["heuristic_scan_score"])
    ax.set_xlabel("heuristic scan score, weighted normalized units")
    ax.set_title(
        "Heuristic interface scan score for manual inspection\n"
        "Higher = stronger hotspot/interface signal under the stated weights; not a binding-energy score",
        pad=12,  # ← adds breathing room between title and plot
    )
    ax.grid(axis="x", alpha=0.25)

    # Remove right and top spines for a cleaner look
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for bar, (_, row) in zip(bars, df.iterrows()):
        txt = (
            f"hotspot residues={int(row.get('n_target_hotspot_residues_contacted', 0))}; "
            f"hotspot pairs={int(row.get('n_hotspot_residue_contacts', 0))}; "
            f"salt={int(row.get('n_putative_salt_bridge_residue_pairs', 0))}; "
            f"polar={int(row.get('n_putative_polar_contacts_residue_pairs', 0))}; "
            f"residue pairs={int(row.get('n_residue_contacts', 0))}"
        )
        ax.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            "  " + txt, va="center", fontsize=8,
        )
    formula = (
        "Score = 3.0*norm(hotspot residues contacted) + 2.0*norm(hotspot-binder residue pairs)\n"
        "+ 1.5*norm(salt-bridge pairs) + 1.0*norm(polar pairs) + 0.5*norm(total residue pairs)"
    )
    if distance_col in summary.columns:
        formula += "\n+ 0.5*norm_inverse(minimum interface distance)"

    # Count formula lines to scale bottom margin dynamically
    n_formula_lines = formula.count("\n") + 1
    bottom_margin = 0.01 + n_formula_lines * 0.03  # ~0.03 per line of text

    fig.text(0.01, 0.01, formula, ha="left", va="bottom", fontsize=8)

    # Reserve space: top for title (2 lines), bottom for formula text
    plt.tight_layout(rect=[0, bottom_margin, 1, 1])

    savefig(outdir / "02_heuristic_scan_score.png", dpi=dpi)
    score_table = df[["design_label", "heuristic_scan_score"] + list(score_terms)].copy()
    score_table = score_table.sort_values("heuristic_scan_score", ascending=False)
    score_table.to_csv(outdir / "heuristic_scan_score_table.csv", index=False)
    return score_table


def plot_hotspot_contact_count_matrix(hotspot: pd.DataFrame, outdir: Path, dpi: int) -> None:
    """Color = number of binder residues contacting each target hotspot."""
    if hotspot.empty:
        return
    required = {"design_label", "target_reslabel", "binder_reslabel"}
    if not required.issubset(set(hotspot.columns)):
        return

    counts = (
        hotspot.groupby(["design_label", "target_reslabel"])["binder_reslabel"]
        .nunique()
        .unstack(fill_value=0)
    )
    # Sort hotspot columns numerically when possible.
    counts = counts.reindex(columns=sorted(counts.columns, key=lambda x: int(re.findall(r"\d+", str(x))[-1]) if re.findall(r"\d+", str(x)) else str(x)))
    counts = counts.sort_index()

    fig_w = max(8, 0.9 * len(counts.columns) + 4)
    fig_h = choose_fig_height(len(counts), base=3.5, per_row=0.55)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(counts.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="number of unique binder residues contacting hotspot")
    plt.xticks(range(len(counts.columns)), counts.columns, rotation=35, ha="right")
    plt.yticks(range(len(counts.index)), counts.index)
    plt.title("Hotspot contact-count matrix\nCell value = number of binder residues within the contact cutoff")

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            val = int(counts.iloc[i, j])
            plt.text(j, i, str(val), ha="center", va="center", fontsize=9)

    savefig(outdir / "03_hotspot_contact_count_matrix.png", dpi=dpi)


def plot_hotspot_min_distance_matrix(hotspot: pd.DataFrame, outdir: Path, dpi: int) -> None:
    """Color = minimum heavy-atom distance from hotspot to binder."""
    if hotspot.empty:
        return
    required = {"design_label", "target_reslabel", "min_heavy_atom_distance_A"}
    if not required.issubset(set(hotspot.columns)):
        return

    hotspot = hotspot.copy()
    hotspot["min_heavy_atom_distance_A"] = pd.to_numeric(hotspot["min_heavy_atom_distance_A"], errors="coerce")
    dist = (
        hotspot.groupby(["design_label", "target_reslabel"])["min_heavy_atom_distance_A"]
        .min()
        .unstack()
    )
    dist = dist.reindex(columns=sorted(dist.columns, key=lambda x: int(re.findall(r"\d+", str(x))[-1]) if re.findall(r"\d+", str(x)) else str(x)))
    dist = dist.sort_index()

    fig_w = max(8, 0.9 * len(dist.columns) + 4)
    fig_h = choose_fig_height(len(dist), base=3.5, per_row=0.55)
    plt.figure(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(dist.values.astype(float))
    im = plt.imshow(masked, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="minimum heavy-atom distance to binder, Å")
    plt.xticks(range(len(dist.columns)), dist.columns, rotation=35, ha="right")
    plt.yticks(range(len(dist.index)), dist.index)
    plt.title("Hotspot minimum-distance matrix\nCell value = closest heavy-atom distance between hotspot and binder, Å")

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            val = dist.iloc[i, j]
            txt = "NA" if pd.isna(val) else f"{val:.2f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

    savefig(outdir / "04_hotspot_min_distance_matrix.png", dpi=dpi)


def plot_target_interface_burden(contacts: pd.DataFrame, outdir: Path, dpi: int, top_n: int = 40) -> None:
    """Across all designs, show which target residues are most frequently contacted."""
    if contacts.empty or "target_reslabel" not in contacts.columns:
        return

    burden = (
        contacts.groupby("target_reslabel")
        .agg(
            n_designs_contacted=("design_label", "nunique"),
            n_binder_residue_pairs=("binder_reslabel", "count"),
            min_distance_A=("min_heavy_atom_distance_A", "min"),
        )
        .reset_index()
    )
    burden = burden.sort_values(["n_designs_contacted", "n_binder_residue_pairs"], ascending=False).head(top_n)
    burden = burden.sort_values(["n_designs_contacted", "n_binder_residue_pairs"], ascending=True)

    fig_h = choose_fig_height(len(burden), base=4, per_row=0.32, max_h=20)
    plt.figure(figsize=(10, fig_h))
    plt.barh(burden["target_reslabel"], burden["n_binder_residue_pairs"])
    plt.xlabel("total binder-residue contact pairs across designs")
    plt.ylabel("target residue")
    plt.title(f"Top {len(burden)} contacted target residues across all designs")
    plt.grid(axis="x", alpha=0.25)

    for y, (_, row) in enumerate(burden.iterrows()):
        txt = f"designs={int(row['n_designs_contacted'])}; min d={row['min_distance_A']:.2f} Å"
        plt.text(row["n_binder_residue_pairs"], y, "  " + txt, va="center", fontsize=8)

    savefig(outdir / "05_target_interface_burden.png", dpi=dpi)


# -----------------------------------------------------------------------------
# Per-design plots
# -----------------------------------------------------------------------------

def plot_contact_map_for_design(df: pd.DataFrame, outpath: Path, dpi: int, title_prefix: str) -> None:
    if df.empty:
        return
    required = {"target_reslabel", "binder_reslabel", "min_heavy_atom_distance_A"}
    if not required.issubset(set(df.columns)):
        return

    tmp = df.copy()
    tmp["min_heavy_atom_distance_A"] = pd.to_numeric(tmp["min_heavy_atom_distance_A"], errors="coerce")
    mat = tmp.pivot_table(
        index="target_reslabel",
        columns="binder_reslabel",
        values="min_heavy_atom_distance_A",
        aggfunc="min",
    )

    # Sort by residue numbers when possible.
    def residue_sort_key(label: str):
        nums = re.findall(r"\d+", str(label))
        return (int(nums[-1]) if nums else 10**9, str(label))

    mat = mat.reindex(index=sorted(mat.index, key=residue_sort_key))
    mat = mat.reindex(columns=sorted(mat.columns, key=residue_sort_key))

    fig_w = max(8, 0.28 * len(mat.columns) + 4)
    fig_h = max(6, 0.25 * len(mat.index) + 3)
    plt.figure(figsize=(min(fig_w, 24), min(fig_h, 24)))
    masked = np.ma.masked_invalid(mat.values.astype(float))
    im = plt.imshow(masked, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="minimum heavy-atom distance, Å")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(mat.index)), mat.index, fontsize=7)
    plt.xlabel("binder residue")
    plt.ylabel("target residue")
    plt.title(f"{title_prefix}\nResidue-pair contact map; color = minimum heavy-atom distance, Å")
    savefig(outpath, dpi=dpi)


def plot_top_contacts_table(df: pd.DataFrame, outpath: Path, dpi: int, top_n: int) -> None:
    if df.empty:
        return
    cols = [
        "target_reslabel", "binder_reslabel", "min_heavy_atom_distance_A", "n_atom_contacts",
        "n_sidechain_sidechain_contacts", "n_backbone_sidechain_contacts",
        "putative_polar_contact", "putative_salt_bridge", "closest_target_atom", "closest_binder_atom"
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return

    tmp = df.copy()
    if "min_heavy_atom_distance_A" in tmp.columns:
        tmp["min_heavy_atom_distance_A"] = pd.to_numeric(tmp["min_heavy_atom_distance_A"], errors="coerce")
        tmp = tmp.sort_values(["min_heavy_atom_distance_A", "n_atom_contacts" if "n_atom_contacts" in tmp.columns else "min_heavy_atom_distance_A"], ascending=[True, False])
    tmp = tmp.head(top_n)[cols].copy()

    rename = {
        "target_reslabel": "target",
        "binder_reslabel": "binder",
        "min_heavy_atom_distance_A": "min dist Å",
        "n_atom_contacts": "atom contacts",
        "n_sidechain_sidechain_contacts": "SC-SC",
        "n_backbone_sidechain_contacts": "BB-SC",
        "putative_polar_contact": "polar",
        "putative_salt_bridge": "salt bridge",
        "closest_target_atom": "closest target atom",
        "closest_binder_atom": "closest binder atom",
    }
    tmp = tmp.rename(columns=rename)
    if "min dist Å" in tmp.columns:
        tmp["min dist Å"] = tmp["min dist Å"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    fig_h = max(3.5, 0.38 * len(tmp) + 1.2)
    fig_w = 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=tmp.astype(str).values,
        colLabels=list(tmp.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    ax.set_title(f"Top {len(tmp)} residue-pair contacts ranked by minimum distance", pad=12)
    savefig(outpath, dpi=dpi)


def make_per_design_plots(contacts: pd.DataFrame, hotspot: pd.DataFrame, outdir: Path, dpi: int, top_n: int) -> None:
    if contacts.empty or "design_label" not in contacts.columns:
        return
    per_dir = outdir / "per_design"
    for design, df_design in contacts.groupby("design_label"):
        design_dir = per_dir / sanitize_filename(design)
        plot_contact_map_for_design(
            df_design,
            design_dir / "contact_map.png",
            dpi=dpi,
            title_prefix=str(design),
        )
        plot_top_contacts_table(
            df_design,
            design_dir / "top_contacts_table.png",
            dpi=dpi,
            top_n=top_n,
        )
        if not hotspot.empty and "design_label" in hotspot.columns:
            hot_design = hotspot[hotspot["design_label"] == design]
            if not hot_design.empty:
                plot_contact_map_for_design(
                    hot_design,
                    design_dir / "hotspot_contact_map.png",
                    dpi=dpi,
                    title_prefix=f"{design} hotspot contacts",
                )


# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------

def write_notes(outdir: Path) -> None:
    notes = """Figure interpretation notes
===========================

00_design_metric_overview_absolute.png
- Raw values from model_summary.csv.
- Metrics are split into panels because they have different scales.
- Counts are absolute counts. Minimum interface distance is reported in Å; lower means the closest residue pair is closer.

01_design_metric_overview_normalized.png
- Same core metrics as the absolute plot, but each column is min-max normalized across the designs in the current dataset.
- Color scale: 0 = lowest value among the designs; 1 = highest value among the designs.
- Raw values are printed inside the cells.
- For minimum interface distance, the color is inverted: lower distance is shown as a higher relative visual score.

02_heuristic_scan_score.png
- This is a practical scan score for prioritizing manual inspection.
- It is not a physical binding-energy score and should not be interpreted as affinity.
- Score = 3.0*norm(hotspot residues contacted)
        + 2.0*norm(hotspot-binder residue pairs)
        + 1.5*norm(salt-bridge pairs)
        + 1.0*norm(polar pairs)
        + 0.5*norm(total residue-pair contacts)
        + 0.5*norm_inverse(minimum interface distance), when that column exists.
- All terms are normalized across the current dataset before weighting.

03_hotspot_contact_count_matrix.png
- Cell value = number of unique binder residues contacting each target hotspot residue.
- This is a count, not a distance.
- A value of 0 means that hotspot was not contacted by the binder under the contact cutoff used in the analysis.

04_hotspot_min_distance_matrix.png
- Cell value = minimum heavy-atom distance between that target hotspot residue and the binder, in Å.
- This is a distance, not a contact count.
- NA means no contact for that hotspot in that design in hotspot_contacts.csv.

05_target_interface_burden.png
- Across all designs, this shows which target residues are contacted most often.
- Useful to identify recurring target-surface engagement and potential hotspot convergence.

Per-design contact_map.png
- Rows = target residues; columns = binder residues.
- Color = minimum heavy-atom distance in Å for residue pairs that pass the contact cutoff.

Per-design top_contacts_table.png
- Ranked table of closest residue-pair contacts.
- Useful for fast manual inspection of dominant contacts, polar pairs, and salt bridges.
"""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figure_legend_notes.txt").write_text(notes)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clear PNG visualizations for binder-target interface analysis CSV outputs."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing model_summary.csv, interface_contacts.csv, hotspot_contacts.csv.")
    parser.add_argument("--outdir", type=Path, default=Path("interface_plots_revised"), help="Output directory for PNG files.")
    parser.add_argument("--top-n-contacts", type=int, default=25, help="Number of top contacts to show in per-design table PNGs.")
    parser.add_argument("--top-n-target-residues", type=int, default=40, help="Number of target residues to show in target interface burden plot.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    summary = safe_read_csv(input_dir / "model_summary.csv")
    contacts = safe_read_csv(input_dir / "interface_contacts.csv")
    hotspot = safe_read_csv(input_dir / "hotspot_contacts.csv")
    errors = safe_read_csv(input_dir / "errors.csv")

    summary = add_design_label(summary)
    contacts = add_design_label(contacts)
    hotspot = add_design_label(hotspot)

    numeric_cols = [
        "n_residue_contacts", "n_target_interface_residues", "n_binder_interface_residues",
        "n_target_hotspot_residues_contacted", "n_hotspot_residue_contacts",
        "n_putative_polar_contacts_residue_pairs", "n_putative_salt_bridge_residue_pairs",
        "min_interface_distance_A", "target_resid", "binder_resid", "min_heavy_atom_distance_A",
        "n_atom_contacts", "n_sidechain_sidechain_contacts", "n_backbone_sidechain_contacts",
        "n_backbone_backbone_contacts",
    ]
    summary = as_numeric(summary, numeric_cols)
    contacts = as_numeric(contacts, numeric_cols)
    hotspot = as_numeric(hotspot, numeric_cols)

    bool_cols = ["target_is_hotspot", "putative_polar_contact", "putative_salt_bridge"]
    contacts = as_bool(contacts, bool_cols)
    hotspot = as_bool(hotspot, bool_cols)

    plot_absolute_metric_overview(summary, outdir, args.dpi)
    plot_normalized_metric_overview(summary, outdir, args.dpi)
    score_table = plot_heuristic_scan_score(summary, outdir, args.dpi)
    plot_hotspot_contact_count_matrix(hotspot, outdir, args.dpi)
    plot_hotspot_min_distance_matrix(hotspot, outdir, args.dpi)
    plot_target_interface_burden(contacts, outdir, args.dpi, top_n=args.top_n_target_residues)
    make_per_design_plots(contacts, hotspot, outdir, args.dpi, top_n=args.top_n_contacts)
    write_notes(outdir)

    manifest = []
    for p in sorted(outdir.rglob("*.png")):
        manifest.append(str(p.relative_to(outdir)))
    (outdir / "plot_manifest.txt").write_text("\n".join(manifest) + "\n")

    print(f"Wrote plots to: {outdir}")
    print(f"Number of PNG files: {len(manifest)}")
    if not errors.empty:
        print(f"Warning: errors.csv contains {len(errors)} rows. Check it separately.")
    if not score_table.empty:
        best = score_table.iloc[0]
        print(f"Top heuristic design: {best['design_label']}  score={best['heuristic_scan_score']:.3f}")


if __name__ == "__main__":
    main()
