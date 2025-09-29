
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit script for S2 output: part_index.for_split.csv

Usage:
    python check_part_index_for_split.py --input part_index.for_split.csv --outdir audit_output

What it does (brief):
1) Basic integrity: file load, row count, memory footprint
2) Column discovery (case-insensitive, with synonyms) and required/optional fields check
3) Uniqueness & null checks for key fields (part_id, family_id; optional: family_major, content_hash, duplicate_canonical)
4) Distribution summaries (families, duplicates, datasets/units/kernels, timestamps if present)
5) Cross-field consistency:
   - duplicate_canonical spanning multiple family_id (flag)
   - content_hash spanning multiple duplicate_canonical (flag)
   - family_major uniqueness within duplicate_canonical (should be 1 if the "major" fix was applied)
6) LODO-readiness hints (are there sufficient families? are duplicate groups concentrated? etc.)
7) Outputs: markdown report, JSON summary, anomaly CSVs, simple charts (matplotlib only)

Author: auto-generated
"""
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import math

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ------------------------------ Helpers ------------------------------

SYNONYMS = {
    "part_id": ["part_id", "partid", "id", "pid"],
    "family_id": ["family_id", "familyid", "fam_id", "family"],
    "family_major": ["family_major", "familymajor", "fam_major", "primary_family", "family_main"],
    "duplicate_canonical": ["duplicate_canonical", "dup_canonical", "dup_canon", "canonical", "canonical_id"],
    "content_hash": ["content_hash", "contenthash", "file_hash", "hash_content"],
    "geom_hash": ["geom_hash", "geometry_hash", "geomh", "geomhash"],
    "source_dataset": ["source_dataset", "dataset", "source", "repo", "collection"],
    "unit": ["unit", "units", "length_unit"],
    "kernel": ["kernel", "brep_kernel", "cad_kernel"],
    "timestamp": ["timestamp", "time", "created_at", "mtime", "file_timestamp", "date"],
    "path": ["path", "file", "filepath", "file_path", "relpath", "abspath"],
}

REQUIRED = ["part_id", "family_id"]
RECOMMENDED = ["family_major", "duplicate_canonical", "content_hash"]

ALLOWED_UNITS = {
    "mm": ["mm", "millimeter", "millimetre"],
    "in": ["in", "inch", "inches"],
}

def find_column(df_columns_lower, canonical):
    """Return the actual column name in df that matches canonical via synonyms, or None."""
    names = SYNONYMS.get(canonical, [canonical])
    for name in names:
        if name in df_columns_lower:
            return df_columns_lower[name]
    return None

def build_column_map(df):
    """Map canonical -> actual name (or None)."""
    lower_map = {c.lower(): c for c in df.columns}
    colmap = {}
    for canonical in SYNONYMS.keys():
        colmap[canonical] = find_column(lower_map, canonical)
    return colmap

def safe_series(df, colname):
    return df[colname] if colname in df.columns else pd.Series([None]*len(df))

def normalize_unit(u):
    if pd.isna(u):
        return np.nan
    s = str(u).strip().lower()
    for key, vals in ALLOWED_UNITS.items():
        if s in vals:
            return key
    return s  # unknown retained as-is

def describe_series(s):
    s = s.dropna()
    if s.empty:
        return {"count_nonnull": 0}
    desc = {}
    desc["count_nonnull"] = int(s.shape[0])
    if pd.api.types.is_numeric_dtype(s):
        d = s.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        desc.update({k: (float(v) if not pd.isna(v) else None) for k, v in d.to_dict().items()})
    else:
        vc = s.value_counts(dropna=True).head(20)
        desc["top_values"] = vc.to_dict()
    return desc

def save_histogram(data, outpath, title, xlabel):
    fig = plt.figure()
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_bar(categories, counts, outpath, title, xlabel):
    fig = plt.figure(figsize=(10, 5))
    y = np.arange(len(categories))
    plt.barh(y, counts)
    plt.yticks(y, categories)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Category")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def month_bucket(dt):
    if pd.isna(dt):
        return np.nan
    return datetime(dt.year, dt.month, 1)

# ------------------------------ Main ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to part_index.for_split.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for audit files")
    parser.add_argument("--sep", default=",", help="CSV delimiter (default: ',')")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    parser.add_argument("--sample_rows", type=int, default=0, help="Optional: only load first N rows for a quick smoke test")
    args = parser.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    try:
        nrows = args.sample_rows if args.sample_rows and args.sample_rows > 0 else None
        df = pd.read_csv(inp, sep=args.sep, encoding=args.encoding, nrows=nrows, low_memory=False)
    except Exception as e:
        print(f"[FATAL] Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(2)

    row_count = df.shape[0]
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"[INFO] Loaded {row_count} rows, approx memory {mem_mb:.2f} MB")

    # 2) Column discovery
    colmap = build_column_map(df)
    present = {k: (v is not None) for k, v in colmap.items()}

    missing_required = [c for c in REQUIRED if not present.get(c, False)]
    missing_recommended = [c for c in RECOMMENDED if not present.get(c, False)]

    print("[INFO] Column mapping (canonical -> actual):")
    for k, v in colmap.items():
        print(f"  - {k:18s} -> {v}")

    if missing_required:
        print(f"[FAIL] Missing REQUIRED columns: {missing_required}", file=sys.stderr)
    else:
        print("[PASS] All REQUIRED columns present.")

    if missing_recommended:
        print(f"[WARN] Missing RECOMMENDED columns: {missing_recommended}")

    # Extract key columns (actual names or None)
    col_part = colmap["part_id"]
    col_fam = colmap["family_id"]
    col_fam_major = colmap["family_major"]
    col_dup_can = colmap["duplicate_canonical"]
    col_content_hash = colmap["content_hash"]
    col_geom_hash = colmap["geom_hash"]
    col_dataset = colmap["source_dataset"]
    col_unit = colmap["unit"]
    col_kernel = colmap["kernel"]
    col_ts = colmap["timestamp"]

    # 3) Basic integrity checks
    issues = {}

    # Uniqueness of part_id
    if col_part:
        dup_part = df[df[col_part].duplicated(keep=False)]
        dup_part_out = outdir / "anomaly_duplicate_part_id.csv"
        if not dup_part.empty:
            dup_part.to_csv(dup_part_out, index=False)
            issues["duplicate_part_id_count"] = int(dup_part.shape[0])
            print(f"[FAIL] Found duplicate part_id rows: {dup_part.shape[0]} -> {dup_part_out}")
        else:
            print("[PASS] part_id is unique.")
    else:
        issues["missing_part_id"] = True

    # Null checks for key columns
    def null_rate(col):
        if not col:
            return None
        return float(df[col].isna().mean())

    key_cols = [c for c in [col_part, col_fam, col_fam_major, col_dup_can, col_content_hash] if c]
    nulls = {c: null_rate(c) for c in key_cols}
    for c, r in nulls.items():
        if r is None:
            continue
        tag = "[WARN]" if r > 0 else "[PASS]"
        print(f"{tag} Null rate for {c}: {r:.4f}")
        if r and r > 0:
            issues[f"null_rate::{c}"] = r

    # 4) Distributions
    summary = {
        "row_count": int(row_count),
        "memory_mb": float(mem_mb),
        "columns_present": {k: bool(v) for k, v in colmap.items()},
    }

    # Family size distribution
    family_stats = {}
    if col_fam:
        fam_sizes = df.groupby(col_fam).size().sort_values(ascending=False)
        family_stats["n_families"] = int(fam_sizes.shape[0])
        family_stats["size_describe"] = describe_series(fam_sizes.astype(float))
        family_stats["iso_ratio"] = float((fam_sizes == 1).mean()) if fam_sizes.shape[0] > 0 else None

        # save largest families
        top_fam = fam_sizes.head(50).reset_index()
        top_fam.columns = [col_fam, "count"]
        top_fam_path = outdir / "top50_families.csv"
        top_fam.to_csv(top_fam_path, index=False)
        print(f"[INFO] Saved top-50 families to {top_fam_path}")

        # histogram
        try:
            save_histogram(fam_sizes.values, outdir / "family_size_hist.png",
                           "Family Size Distribution", "Family size")
            print(f"[INFO] Wrote family size histogram: {outdir / 'family_size_hist.png'}")
        except Exception as e:
            print(f"[WARN] Could not write family size histogram: {e}")

    summary["family_stats"] = family_stats

    # Duplicate canonical distribution
    dup_stats = {}
    if col_dup_can:
        g = df.groupby(col_dup_can)
        dup_sizes = g.size().sort_values(ascending=False)

        dup_stats["n_canon_groups"] = int(dup_sizes.shape[0])
        dup_stats["size_describe"] = describe_series(dup_sizes.astype(float))
        dup_stats["singleton_ratio"] = float((dup_sizes == 1).mean()) if dup_sizes.shape[0] > 0 else None

        # Check canon -> multiple family_id
        if col_fam:
            fams_per_canon = g[col_fam].nunique(dropna=True)
            multi_fam_canon = fams_per_canon[fams_per_canon > 1]
            if not multi_fam_canon.empty:
                path = outdir / "anomaly_canonical_with_multiple_families.csv"
                multi_fam_canon.to_frame("unique_family_ids").to_csv(path)
                dup_stats["canonical_with_multiple_families_count"] = int(multi_fam_canon.shape[0])
                print(f"[WARN] duplicate_canonical groups spanning multiple family_id: {multi_fam_canon.shape[0]} -> {path}")

        # Check canon -> multiple family_major (should be 1 post-fix, if present)
        if col_fam_major:
            majors_per_canon = g[col_fam_major].nunique(dropna=True)
            multi_major_canon = majors_per_canon[majors_per_canon > 1]
            if not multi_major_canon.empty:
                path = outdir / "anomaly_canonical_with_multiple_family_major.csv"
                multi_major_canon.to_frame("unique_family_major").to_csv(path)
                dup_stats["canonical_with_multiple_family_major_count"] = int(multi_major_canon.shape[0])
                print(f"[FAIL] duplicate_canonical groups with MULTIPLE family_major: {multi_major_canon.shape[0]} -> {path}")
            else:
                print("[PASS] Each duplicate_canonical maps to a single family_major (if present).")

        # histogram
        try:
            save_histogram(dup_sizes.values, outdir / "duplicate_group_size_hist.png",
                           "Duplicate-Canonical Group Size Distribution", "Group size")
            print(f"[INFO] Wrote duplicate group size histogram: {outdir / 'duplicate_group_size_hist.png'}")
        except Exception as e:
            print(f"[WARN] Could not write duplicate group histogram: {e}")

    summary["duplicate_canonical_stats"] = dup_stats

    # content_hash cross-check
    ch_stats = {}
    if col_content_hash:
        ch_g = df.groupby(col_content_hash)
        ch_sizes = ch_g.size().sort_values(ascending=False)
        ch_stats["n_content_hash_groups"] = int(ch_sizes.shape[0])
        ch_stats["size_describe"] = describe_series(ch_sizes.astype(float))
        ch_stats["singleton_ratio"] = float((ch_sizes == 1).mean()) if ch_sizes.shape[0] > 0 else None

        if col_dup_can:
            can_per_ch = ch_g[col_dup_can].nunique(dropna=True)
            bad = can_per_ch[can_per_ch > 1]
            if not bad.empty:
                path = outdir / "anomaly_content_hash_to_multiple_canonical.csv"
                bad.to_frame("unique_duplicate_canonical").to_csv(path)
                ch_stats["content_hash_to_multiple_canonical_count"] = int(bad.shape[0])
                print(f"[FAIL] content_hash mapping to MULTIPLE duplicate_canonical: {bad.shape[0]} -> {path}")
            else:
                print("[PASS] Each content_hash maps to a single duplicate_canonical (if both present).")

        # histogram
        try:
            save_histogram(ch_sizes.values, outdir / "content_hash_group_size_hist.png",
                           "Content-Hash Group Size Distribution", "Group size")
            print(f"[INFO] Wrote content-hash group size histogram: {outdir / 'content_hash_group_size_hist.png'}")
        except Exception as e:
            print(f"[WARN] Could not write content-hash group histogram: {e}")

    summary["content_hash_stats"] = ch_stats

    # Dataset / unit / kernel distributions
    if col_dataset and df[col_dataset].notna().any():
        counts = df[col_dataset].astype(str).value_counts().head(30)
        path = outdir / "dataset_counts.csv"
        counts.to_csv(path, header=["count"])
        print(f"[INFO] Saved dataset top-30 counts to {path}")

        try:
            save_bar(list(counts.index)[::-1], list(counts.values)[::-1],
                     outdir / "dataset_bar.png", "Top-30 source_dataset counts", "Count")
            print(f"[INFO] Wrote dataset bar chart: {outdir / 'dataset_bar.png'}")
        except Exception as e:
            print(f"[WARN] Could not write dataset bar chart: {e}")

    unit_stats = {}
    if col_unit and df[col_unit].notna().any():
        norm_units = df[col_unit].map(normalize_unit)
        df["_unit_norm"] = norm_units
        unit_counts = norm_units.value_counts(dropna=False)
        unit_stats["counts"] = unit_counts.to_dict()
        unknown = unit_counts.drop(labels=["mm", "in"], errors="ignore")
        if unknown.sum() > 0:
            path = outdir / "anomaly_unknown_units.csv"
            unknown.to_frame("count").to_csv(path)
            unit_stats["unknown_unit_total"] = int(unknown.sum())
            print(f"[WARN] Found unknown unit strings -> {path}")
        else:
            print("[PASS] Units are normalized to {'mm','in'} or missing.")
    summary["unit_stats"] = unit_stats

    if col_kernel and df[col_kernel].notna().any():
        kern_counts = df[col_kernel].astype(str).value_counts().head(30)
        path = outdir / "kernel_counts.csv"
        kern_counts.to_csv(path, header=["count"])
        print(f"[INFO] Saved kernel top-30 counts to {path}")

    # Timestamp checks
    ts_stats = {}
    if col_ts and df[col_ts].notna().any():
        # try parsing
        try:
            ts = pd.to_datetime(df[col_ts], errors="coerce", utc=False, infer_datetime_format=True)
        except Exception:
            ts = pd.to_datetime(df[col_ts], errors="coerce")
        ts_stats["parse_success_rate"] = float(ts.notna().mean())
        if ts.notna().any():
            ts_stats["min"] = str(ts.min())
            ts_stats["max"] = str(ts.max())
            monthly = ts.map(month_bucket).value_counts().sort_index()
            p = outdir / "timestamp_monthly_counts.csv"
            monthly.to_csv(p, header=["count"])
            print(f"[INFO] Saved monthly timestamp counts to {p}")
            try:
                fig = plt.figure()
                plt.plot(monthly.index.astype("datetime64[ns]"), monthly.values)
                plt.title("Monthly file counts (parsed timestamp)")
                plt.xlabel("Month")
                plt.ylabel("Count")
                fig.autofmt_xdate()
                fig.tight_layout()
                fig.savefig(outdir / "timestamp_monthly.png")
                plt.close(fig)
                print(f"[INFO] Wrote timestamp monthly plot: {outdir / 'timestamp_monthly.png'}")
            except Exception as e:
                print(f"[WARN] Could not write timestamp monthly plot: {e}")
        else:
            print("[WARN] No parseable timestamps.")
    summary["timestamp_stats"] = ts_stats

    # 5) LODO readiness hints
    lodo = {}
    if col_fam:
        fam_sizes = df.groupby(col_fam).size()
        lodo["families_ge_10"] = int((fam_sizes >= 10).sum())
        lodo["families_ge_50"] = int((fam_sizes >= 50).sum())
        lodo["families_ge_100"] = int((fam_sizes >= 100).sum())
        lodo["suggestion"] = (
            "OK" if lodo["families_ge_10"] >= 5 else
            "Consider merging very small families or using family_major for analysis only."
        )
    summary["lodo_hint"] = lodo

    # 6) Paths presence (optional)
    if colmap["path"] and df[colmap["path"]].notna().any():
        # Just a sample of missing files check would be out of scope (no FS guarantee),
        # we only report presence of path column.
        print("[INFO] Path column present; filesystem existence not checked.")

    # 7) Save report & summary
    # Summarize issues PASS/FAIL
    hard_fail = bool(missing_required) or \
                issues.get("duplicate_part_id_count", 0) > 0 or \
                dup_stats.get("canonical_with_multiple_family_major_count", 0) if dup_stats else False or \
                ch_stats.get("content_hash_to_multiple_canonical_count", 0) if ch_stats else False

    status = "PASS_WITH_WARNINGS" if not hard_fail and (issues or missing_recommended or dup_stats.get("canonical_with_multiple_families_count", 0)) else ("FAIL" if hard_fail else "PASS")

    summary["status"] = status
    summary["issues"] = issues

    summary_path = outdir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote JSON summary -> {summary_path}")

    # Markdown report
    md = []
    md.append(f"# Audit Report for {os.path.basename(args.input)}")
    md.append("")
    md.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"- Rows: {row_count:,}")
    md.append(f"- Approx memory: {mem_mb:.2f} MB")
    md.append(f"- Status: **{status}**")
    md.append("")
    md.append("## Column Mapping")
    for k, v in colmap.items():
        md.append(f"- `{k}` → `{v}`")
    md.append("")
    if missing_required:
        md.append(f"**Missing REQUIRED columns**: {', '.join(missing_required)}")
    if missing_recommended:
        md.append(f"**Missing RECOMMENDED columns**: {', '.join(missing_recommended)}")
    md.append("")
    md.append("## Key Checks")
    if "duplicate_part_id_count" in issues:
        md.append(f"- ❌ Duplicate part_id rows: {issues['duplicate_part_id_count']} (see `anomaly_duplicate_part_id.csv`)")
    else:
        md.append("- ✅ `part_id` unique")
    for c, r in nulls.items():
        if r is None:
            continue
        badge = "⚠️" if r > 0 else "✅"
        md.append(f"- {badge} Null rate `{c}`: {r:.4f}")
    md.append("")
    md.append("## Family Stats")
    if family_stats:
        md.append(f"- # families: {family_stats.get('n_families')}")
        md.append(f"- iso_ratio (size==1): {family_stats.get('iso_ratio')}")
        md.append(f"- See: `top50_families.csv`, `family_size_hist.png`")
    else:
        md.append("- (family stats unavailable)")
    md.append("")
    md.append("## Duplicate-Canonical Stats")
    if dup_stats:
        for k, v in dup_stats.items():
            md.append(f"- {k}: {v}")
        md.append("See: `duplicate_group_size_hist.png`")
    else:
        md.append("- (duplicate-canonical stats unavailable)")
    md.append("")
    md.append("## Content-Hash Stats")
    if ch_stats:
        for k, v in ch_stats.items():
            md.append(f"- {k}: {v}")
        md.append("See: `content_hash_group_size_hist.png`")
    else:
        md.append("- (content-hash stats unavailable)")
    md.append("")
    md.append("## Units / Kernel / Dataset")
    if unit_stats:
        md.append(f"- Units: {json.dumps(unit_stats, ensure_ascii=False)}")
    if col_kernel and df[col_kernel].notna().any():
        md.append(f"- Kernel top-30 saved: `kernel_counts.csv`")
    if col_dataset and df[col_dataset].notna().any():
        md.append(f"- Dataset top-30 saved: `dataset_counts.csv`, and `dataset_bar.png`")
    md.append("")
    md.append("## Timestamp")
    if ts_stats:
        md.append(f"- Timestamp parse success: {ts_stats.get('parse_success_rate')}")
        if "min" in ts_stats:
            md.append(f"- Range: {ts_stats.get('min')} → {ts_stats.get('max')}")
            md.append("- See: `timestamp_monthly_counts.csv`, `timestamp_monthly.png`")
    md.append("")
    md.append("## LODO Readiness Hint")
    md.append(f"- {json.dumps(lodo, ensure_ascii=False)}")
    md.append("")
    md.append("## Overall Verdict")
    md.append(f"- **{status}**")
    md.append("")

    md_path = outdir / "audit_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[INFO] Wrote markdown report -> {md_path}")

    # Final console hint
    print("\n=== DONE ===")
    print(f"Summary: {summary_path}")
    print(f"Report : {md_path}")
    if status.startswith("FAIL"):
        sys.exit(1)


if __name__ == "__main__":
    main()
