#!/usr/bin/env python3
"""
Batch interface-contact analysis for AlphaFold-predicted binder:target complexes.

Designed for hundreds/thousands of static PDB models.

Outputs:
  1) interface_contacts.csv
     Residue-residue interface contacts for every PDB.
  2) hotspot_contacts.csv
     Subset of contacts involving user-defined hotspot residues on the target.
  3) model_summary.csv
     Per-model summary metrics.

Contact definition:
  A residue-residue contact is counted when any heavy atom pair between the
  target chain and binder chain is within --contact-cutoff Angstrom.

Optional approximate interaction annotations:
  - putative_salt_bridge: acidic O atom to basic N atom within --salt-cutoff
  - putative_polar_contact: polar N/O/S atom pair within --polar-cutoff

Example:
  python interface_batch_analysis.py \
    --input-dir ./pdbs \
    --pattern 'config_myostatin_*_model_*_id*_model.pdb' \
    --target-chain A \
    --binder-chain B \
    --target-hotspots A:45,A:46,A:98,A:99 \
    --outdir ./interface_results \
    --n-workers 8

Dependencies:
  pip install MDAnalysis pandas numpy
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance


# -----------------------------
# Residue/atom chemistry helpers
# -----------------------------

ACIDIC_RESNAMES = {"ASP", "GLU"}
BASIC_RESNAMES = {"ARG", "LYS", "HIS", "HSD", "HSE", "HSP", "HIP", "HID", "HIE"}
CHARGED_RESNAMES = ACIDIC_RESNAMES | BASIC_RESNAMES
POLAR_RESNAMES = {
    "SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "HSD", "HSE", "HSP", "TRP"
}
HYDROPHOBIC_RESNAMES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP"}

# PDB atom names used as charged groups. This is intentionally conservative.
ACIDIC_O_ATOMS = {"OD1", "OD2", "OE1", "OE2"}
BASIC_N_ATOMS = {
    "NZ",              # Lys
    "NE", "NH1", "NH2",  # Arg
    "ND1", "NE2",         # His, approximate/protonation-state agnostic
}
POLAR_ELEMENTS = {"N", "O", "S"}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def residue_class(resname: str) -> str:
    resname = resname.upper()
    if resname in ACIDIC_RESNAMES:
        return "acidic"
    if resname in BASIC_RESNAMES:
        return "basic"
    if resname in POLAR_RESNAMES:
        return "polar"
    if resname in HYDROPHOBIC_RESNAMES:
        return "hydrophobic"
    return "other"


def atom_region(atom_name: str) -> str:
    return "backbone" if atom_name.strip().upper() in BACKBONE_ATOMS else "sidechain"


def parse_hotspots(text: Optional[str]) -> Set[Tuple[str, int, str]]:
    """
    Parse hotspot residues from a comma-separated string.

    Accepted formats:
      A:45,A:46,A:99
      A:45:TYR,A:46:ASP
      45,46,99               # chain left blank; useful only if target chain is fixed

    Returns a set of tuples: (chain, resid, resname_or_empty)
    """
    hotspots: Set[Tuple[str, int, str]] = set()
    if not text:
        return hotspots

    for raw in re.split(r"[,\s]+", text.strip()):
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) == 1:
            chain, resid, resname = "", int(parts[0]), ""
        elif len(parts) == 2:
            chain, resid, resname = parts[0], int(parts[1]), ""
        elif len(parts) == 3:
            chain, resid, resname = parts[0], int(parts[1]), parts[2].upper()
        else:
            raise ValueError(f"Cannot parse hotspot token: {raw}")
        hotspots.add((chain, resid, resname))
    return hotspots


def read_hotspots_file(path: Optional[str]) -> Set[Tuple[str, int, str]]:
    """
    Read target hotspots from CSV/TSV/plain text.

    Accepted columns if CSV/TSV:
      chain,resid,resname
    For plain text, one hotspot per line as A:45 or A:45:TYR.
    """
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        required = {"resid"}
        if not required.issubset(df.columns):
            raise ValueError("Hotspot file must contain at least a 'resid' column.")
        hotspots = set()
        for _, row in df.iterrows():
            hotspots.add((
                str(row.get("chain", "")),
                int(row["resid"]),
                str(row.get("resname", "")).upper() if not pd.isna(row.get("resname", "")) else "",
            ))
        return hotspots

    return parse_hotspots(p.read_text())


def is_hotspot(chain: str, resid: int, resname: str, hotspots: Set[Tuple[str, int, str]]) -> bool:
    """Match either exact chain/resid[/resname] or resid-only hotspots."""
    resname = resname.upper()
    candidates = {
        (chain, resid, resname),
        (chain, resid, ""),
        ("", resid, resname),
        ("", resid, ""),
    }
    return bool(candidates & hotspots)


def parse_filename_metadata(pdb_path: Path) -> Dict[str, object]:
    """
    Extract metadata from the expected filename:
      config_myostatin_X_model_Y_idZ_model.pdb
    Returns empty values if the name does not match.
    """
    m = re.search(
        r"config_(?P<target>.+?)_(?P<config>\d+)_model_(?P<model>\d+)_id(?P<id>\d+)_model\.pdb$",
        pdb_path.name,
    )
    if not m:
        return {"design_target": "", "config_idx": np.nan, "model_idx": np.nan, "design_id": np.nan}
    return {
        "design_target": m.group("target"),
        "config_idx": int(m.group("config")),
        "model_idx": int(m.group("model")),
        "design_id": int(m.group("id")),
    }


def _safe_chain_selection(chain_id: str) -> str:
    """MDAnalysis selection string for chain IDs."""
    # PDB chain IDs are usually one char; quote-free selection is fine for alphanumeric IDs.
    return f"chainID {chain_id}"


# -----------------------------
# Core analysis
# -----------------------------


def analyze_pdb(
    pdb_file: str,
    target_chain: str,
    binder_chain: str,
    target_hotspots: Set[Tuple[str, int, str]],
    contact_cutoff: float,
    polar_cutoff: float,
    salt_cutoff: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    pdb_path = Path(pdb_file)
    metadata = parse_filename_metadata(pdb_path)

    base = {
        "pdb_file": pdb_path.name,
        "pdb_path": str(pdb_path),
        **metadata,
        "target_chain": target_chain,
        "binder_chain": binder_chain,
    }

    u = mda.Universe(str(pdb_path))

    # Heavy atoms only: faster and avoids hydrogen-dependent behavior across models.
    target = u.select_atoms(f"({_safe_chain_selection(target_chain)}) and protein and not name H* and not element H")
    binder = u.select_atoms(f"({_safe_chain_selection(binder_chain)}) and protein and not name H* and not element H")

    if len(target) == 0:
        raise ValueError(f"No target atoms selected for chain {target_chain} in {pdb_path.name}")
    if len(binder) == 0:
        raise ValueError(f"No binder atoms selected for chain {binder_chain} in {pdb_path.name}")

    # Fast neighbor search. Returns atom-index pairs within the cutoff.
    # Indices are local to target.positions and binder.positions.
    pairs, distances = capped_distance(
        target.positions,
        binder.positions,
        max_cutoff=contact_cutoff,
        box=None,
        return_distances=True,
    )

    residue_pair_data: Dict[Tuple[int, int], Dict[str, object]] = {}
    atom_contact_rows: List[Dict[str, object]] = []

    for (ti, bi), dist in zip(pairs, distances):
        ta = target[int(ti)]
        ba = binder[int(bi)]
        tr = ta.residue
        br = ba.residue

        # Use residue indices internal to the Universe as stable grouping keys.
        key = (tr.ix, br.ix)

        if key not in residue_pair_data:
            target_is_hs = is_hotspot(
                target_chain, int(tr.resid), str(tr.resname), target_hotspots
            )
            residue_pair_data[key] = {
                **base,
                "target_resid": int(tr.resid),
                "target_resname": str(tr.resname),
                "target_reslabel": f"{target_chain}:{tr.resname}{int(tr.resid)}",
                "target_resclass": residue_class(str(tr.resname)),
                "target_is_hotspot": target_is_hs,
                "binder_resid": int(br.resid),
                "binder_resname": str(br.resname),
                "binder_reslabel": f"{binder_chain}:{br.resname}{int(br.resid)}",
                "binder_resclass": residue_class(str(br.resname)),
                "min_heavy_atom_distance_A": float(dist),
                "n_atom_contacts": 0,
                "n_sidechain_sidechain_contacts": 0,
                "n_backbone_sidechain_contacts": 0,
                "n_backbone_backbone_contacts": 0,
                "putative_polar_contact": False,
                "putative_salt_bridge": False,
                "closest_target_atom": str(ta.name),
                "closest_binder_atom": str(ba.name),
            }

        row = residue_pair_data[key]
        row["n_atom_contacts"] += 1

        t_region = atom_region(str(ta.name))
        b_region = atom_region(str(ba.name))
        if t_region == "sidechain" and b_region == "sidechain":
            row["n_sidechain_sidechain_contacts"] += 1
        elif t_region == "backbone" and b_region == "backbone":
            row["n_backbone_backbone_contacts"] += 1
        else:
            row["n_backbone_sidechain_contacts"] += 1

        if float(dist) < row["min_heavy_atom_distance_A"]:
            row["min_heavy_atom_distance_A"] = float(dist)
            row["closest_target_atom"] = str(ta.name)
            row["closest_binder_atom"] = str(ba.name)

        te = str(getattr(ta, "element", "")).upper() or str(ta.name)[0].upper()
        be = str(getattr(ba, "element", "")).upper() or str(ba.name)[0].upper()
        t_atom_name = str(ta.name).upper()
        b_atom_name = str(ba.name).upper()
        t_resname = str(tr.resname).upper()
        b_resname = str(br.resname).upper()

        if dist <= polar_cutoff and te in POLAR_ELEMENTS and be in POLAR_ELEMENTS:
            row["putative_polar_contact"] = True

        target_acid_binder_base = (
            t_resname in ACIDIC_RESNAMES
            and t_atom_name in ACIDIC_O_ATOMS
            and b_resname in BASIC_RESNAMES
            and b_atom_name in BASIC_N_ATOMS
        )
        binder_acid_target_base = (
            b_resname in ACIDIC_RESNAMES
            and b_atom_name in ACIDIC_O_ATOMS
            and t_resname in BASIC_RESNAMES
            and t_atom_name in BASIC_N_ATOMS
        )
        if dist <= salt_cutoff and (target_acid_binder_base or binder_acid_target_base):
            row["putative_salt_bridge"] = True

    contact_rows = list(residue_pair_data.values())
    hotspot_rows = [r for r in contact_rows if bool(r["target_is_hotspot"])]

    target_interface_res = {(r["target_resid"], r["target_resname"]) for r in contact_rows}
    binder_interface_res = {(r["binder_resid"], r["binder_resname"]) for r in contact_rows}
    hotspot_res_contacted = {(r["target_resid"], r["target_resname"]) for r in hotspot_rows}

    summary = {
        **base,
        "n_target_atoms": int(len(target)),
        "n_binder_atoms": int(len(binder)),
        "n_residue_contacts": int(len(contact_rows)),
        "n_target_interface_residues": int(len(target_interface_res)),
        "n_binder_interface_residues": int(len(binder_interface_res)),
        "n_target_hotspot_residues_contacted": int(len(hotspot_res_contacted)),
        "n_hotspot_residue_contacts": int(len(hotspot_rows)),
        "n_putative_polar_contacts_residue_pairs": int(sum(bool(r["putative_polar_contact"]) for r in contact_rows)),
        "n_putative_salt_bridge_residue_pairs": int(sum(bool(r["putative_salt_bridge"]) for r in contact_rows)),
        "min_interface_distance_A": float(min([r["min_heavy_atom_distance_A"] for r in contact_rows], default=np.nan)),
    }

    return contact_rows, hotspot_rows, summary


def write_empty_outputs(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(outdir / "interface_contacts.csv", index=False)
    pd.DataFrame().to_csv(outdir / "hotspot_contacts.csv", index=False)
    pd.DataFrame().to_csv(outdir / "model_summary.csv", index=False)


def run_batch(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hotspots = parse_hotspots(args.target_hotspots)
    hotspots |= read_hotspots_file(args.target_hotspots_file)

    pdb_files = sorted(input_dir.glob(args.pattern))
    if args.max_files is not None:
        pdb_files = pdb_files[: args.max_files]

    if not pdb_files:
        raise FileNotFoundError(f"No PDB files matched {input_dir / args.pattern}")

    all_contact_rows: List[Dict[str, object]] = []
    all_hotspot_rows: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    worker_kwargs = dict(
        target_chain=args.target_chain,
        binder_chain=args.binder_chain,
        target_hotspots=hotspots,
        contact_cutoff=args.contact_cutoff,
        polar_cutoff=args.polar_cutoff,
        salt_cutoff=args.salt_cutoff,
    )

    if args.n_workers == 1:
        for pdb in pdb_files:
            try:
                contacts, hotspot_contacts, summary = analyze_pdb(str(pdb), **worker_kwargs)
                all_contact_rows.extend(contacts)
                all_hotspot_rows.extend(hotspot_contacts)
                all_summary_rows.append(summary)
            except Exception as exc:  # keep batch running
                errors.append({"pdb_file": pdb.name, "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            future_to_pdb = {
                ex.submit(analyze_pdb, str(pdb), **worker_kwargs): pdb for pdb in pdb_files
            }
            for fut in as_completed(future_to_pdb):
                pdb = future_to_pdb[fut]
                try:
                    contacts, hotspot_contacts, summary = fut.result()
                    all_contact_rows.extend(contacts)
                    all_hotspot_rows.extend(hotspot_contacts)
                    all_summary_rows.append(summary)
                except Exception as exc:  # keep batch running
                    errors.append({"pdb_file": pdb.name, "error": repr(exc)})

    contacts_df = pd.DataFrame(all_contact_rows)
    hotspot_df = pd.DataFrame(all_hotspot_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    errors_df = pd.DataFrame(errors)

    # Stable, useful sorting.
    if not contacts_df.empty:
        contacts_df = contacts_df.sort_values(
            ["pdb_file", "target_is_hotspot", "target_resid", "binder_resid"],
            ascending=[True, False, True, True],
        )
    if not hotspot_df.empty:
        hotspot_df = hotspot_df.sort_values(
            ["pdb_file", "target_resid", "min_heavy_atom_distance_A", "binder_resid"]
        )
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["pdb_file"])

    contacts_df.to_csv(outdir / "interface_contacts.csv", index=False)
    hotspot_df.to_csv(outdir / "hotspot_contacts.csv", index=False)
    summary_df.to_csv(outdir / "model_summary.csv", index=False)
    errors_df.to_csv(outdir / "errors.csv", index=False)

    print(f"Analyzed PDB files: {len(pdb_files)}")
    print(f"Successful models:  {len(summary_df)}")
    print(f"Failed models:      {len(errors_df)}")
    print(f"Output directory:   {outdir.resolve()}")
    print("Wrote:")
    print(f"  - {outdir / 'interface_contacts.csv'}")
    print(f"  - {outdir / 'hotspot_contacts.csv'}")
    print(f"  - {outdir / 'model_summary.csv'}")
    print(f"  - {outdir / 'errors.csv'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch residue-contact analysis for AlphaFold binder:target PDB complexes."
    )
    p.add_argument("--input-dir", default=".", help="Directory containing PDB files.")
    p.add_argument(
        "--pattern",
        default="config_myostatin_*_model_*_id*_model.pdb",
        help="Glob pattern for PDB files.",
    )
    p.add_argument("--outdir", default="interface_results", help="Output directory.")
    p.add_argument("--target-chain", default="A", help="Target chain ID. Default: A")
    p.add_argument("--binder-chain", default="B", help="Binder chain ID. Default: B")
    p.add_argument(
        "--target-hotspots",
        default="",
        help="Comma/space-separated target hotspot residues, e.g. 'A:45,A:46,A:99' or '45,46,99'.",
    )
    p.add_argument(
        "--target-hotspots-file",
        default=None,
        help="Optional CSV/TSV/text file containing hotspot residues.",
    )
    p.add_argument(
        "--contact-cutoff",
        type=float,
        default=4.5,
        help="Heavy-atom distance cutoff in Angstrom for residue-residue contacts. Default: 4.5",
    )
    p.add_argument(
        "--polar-cutoff",
        type=float,
        default=3.5,
        help="Approximate polar atom-pair cutoff in Angstrom. Default: 3.5",
    )
    p.add_argument(
        "--salt-cutoff",
        type=float,
        default=4.0,
        help="Approximate salt bridge atom-pair cutoff in Angstrom. Default: 4.0",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of parallel worker processes. Use 1 to disable multiprocessing.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of PDB files, useful for testing.",
    )
    return p


if __name__ == "__main__":
    parser = build_argparser()
    run_batch(parser.parse_args())
