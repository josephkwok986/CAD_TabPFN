import os, re, json, math
import pandas as pd
from collections import Counter

# === 路径（按需修改）===
PI  = "/workspace/Gjj Local/data/CAD/step_out/s2_out/part_index.csv"
CFG = "/workspace/Gjj Doc/Code/CAD_TabPFN/config.json"
LOG = "/workspace/Gjj Local/data/CAD/step_out/s2_out/run.log"
HIST = os.path.join(os.path.dirname(PI), "family_hist.csv")

def load_df():
    assert os.path.isfile(PI), f"not found: {PI}"
    return pd.read_csv(PI)

def iso_ratio_by(col, df):
    if col not in df.columns: 
        print(f"[skip] no column: {col}"); return
    t = (df.assign(is_iso=df["family_id"].str.startswith("fam_iso_", na=False))
           .groupby(col)["is_iso"].mean().sort_values(ascending=False))
    print(f"\n=== fam_iso_* ratio by {col} (desc) ===")
    print(t.to_string())

def canonical_purity(df, topn=20):
    if "duplicate_canonical" not in df.columns:
        print("[skip] no duplicate_canonical"); return
    g = df.groupby("duplicate_canonical")["family_id"].nunique()
    print("\n=== canonical → #families summary ===")
    print(g.describe().to_string())
    bad = g[g>1].sort_values(ascending=False)
    print(f"\ncanonical with >1 families: {bad.size} / {g.size} ({bad.size/max(1,g.size):.4f})")
    if bad.size>0:
        print(f"\nTop-{topn} offenders:")
        print(bad.head(topn).to_string())

def family_cross_domain(df, key="source_dataset", topn=20):
    if key not in df.columns:
        print(f"[skip] no column: {key}"); return
    # 只看非 iso/ch 的“有效家族”
    mask_geom = (df["has_points"]==1)
    fam_sizes = df[mask_geom]["family_id"].value_counts()
    fam_keep = [k for k in fam_sizes.index if not (k.startswith("fam_iso_") or k.startswith("fam_ch_"))]
    sub = df[df["family_id"].isin(fam_keep)][["family_id", key]]
    cnt = (sub.groupby("family_id")[key].nunique()
               .sort_values(ascending=False))
    print(f"\n=== family cross-{key} breadth (unique {key} per family) ===")
    print(cnt.head(topn).to_string())
    print("\nsummary stats:", cnt.describe().to_string())

def main():
    df = load_df()
    print("rows:", len(df), "cols:", len(df.columns))
    # 1) iso 分域/内核
    for col in ["source_dataset", "kernel"]:
        iso_ratio_by(col, df)
    # 2) canonical 纯度
    canonical_purity(df, topn=20)
    # 3) 家族跨域扩散
    for key in ["source_dataset", "kernel"]:
        family_cross_domain(df, key, topn=20)

if __name__=="__main__":
    main()
