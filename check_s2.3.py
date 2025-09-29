import os, re, json, math
import pandas as pd

# === 路径（按需修改为你的实际路径）===
PI  = "/workspace/Gjj Local/data/CAD/step_out/s2_out/part_index.csv"
CFG = "/workspace/Gjj Doc/Code/CAD_TabPFN/config.json"
LOG = "/workspace/Gjj Local/data/CAD/step_out/s2_out/run.log"
HIST = os.path.join(os.path.dirname(PI), "family_hist.csv")
SUMMARY = os.path.join(os.path.dirname(PI), "summary.json")

def load_cfg_thresholds(cfg_path):
    Tc = None; min_samples=None; post=None
    with open(cfg_path,"r",encoding="utf-8") as f:
        cfg=json.load(f)
    s2=cfg.get("s2_dedup_family_occ",{})
    Tc = float(s2.get("family_distance_threshold", 0.012))
    min_samples = int(s2.get("family_min_samples", 6))
    post = s2.get("family_post_split", {"enabled": True, "max_fraction": 0.06, "max_size": 6000})
    auto = s2.get("family_auto_eps", {"enabled": True, "quantile": 0.12, "scale": 1.02, "knn_k": 6})
    return Tc, min_samples, post, auto

def parse_runlog(log_path):
    if not os.path.isfile(log_path): return {}
    txt=open(log_path,"r",encoding="utf-8").read()
    out={}
    m=re.search(r"auto_eps.*?k=(\d+).*?q=([\d.]+).*?scale=([\d.]+).*?sample=(\d+)/(\d+).*?eps=([\d.]+).*?Tc≈([\d.]+)", txt)
    if m:
        out["knn_k"]=int(m.group(1))
        out["q"]=float(m.group(2)); out["scale"]=float(m.group(3))
        out["sample"]=int(m.group(4)); out["N"]=int(m.group(5))
        out["eps"]=float(m.group(6)); out["Tc_from_eps"]=float(m.group(7))
    m2=re.search(r"\[S2\.3\] with_geom=(\d+), eps=([\d.]+), device=(\w+)", txt)
    if m2:
        out["with_geom"]=int(m2.group(1)); out["eps2"]=float(m2.group(2)); out["device"]=m2.group(3)
    return out

def family_hist_from_part_index(df):
    counts=df["family_id"].value_counts().rename_axis("family_id").reset_index(name="count")
    counts["pct"]=counts["count"]/max(1,len(df))
    return counts

def main():
    assert os.path.isfile(PI), f"not found: {PI}"
    df=pd.read_csv(PI)
    print("\n=== part_index.csv 基本信息 ===")
    print("行数, 列数:", df.shape)
    print("列：", list(df.columns))

    Tc, min_samples, post, auto = load_cfg_thresholds(CFG) if os.path.isfile(CFG) else (0.012,6,{"max_fraction":0.06,"max_size":6000},{"quantile":0.12,"scale":1.02,"knn_k":6})
    eps_fb = math.sqrt(2.0*Tc)
    loginfo = parse_runlog(LOG)

    # family 直方图
    if os.path.isfile(HIST):
        hist=pd.read_csv(HIST)
    else:
        hist=family_hist_from_part_index(df)
    print("\n=== 家族规模 Top-10 ===")
    print(hist.head(10).to_string(index=False))

    # 噪声占比（fam_iso_）
    noise_like = hist[hist["family_id"].str.startswith("fam_iso_", na=False)]["count"].sum()
    frac_noise = noise_like / max(1, len(df))
    print(f"\n噪声样本数（fam_iso_*）：{noise_like} / {len(df)}  (ratio={frac_noise:.4f})")

    # 阈值与是否超限
    N = len(df)
    threshold = min(int(post.get("max_size",6000)), int(post.get("max_fraction",0.06)*max(1,N)))
    # 只看 has_points==1 且 非 fam_iso_/fam_ch_ 的家族
    mask_geom = (df["has_points"]==1)
    fam_sizes = df[mask_geom]["family_id"].value_counts()
    not_iso_ch = [k for k in fam_sizes.index if not(k.startswith("fam_iso_") or k.startswith("fam_ch_"))]
    max_fam = fam_sizes.loc[not_iso_ch].max() if len(not_iso_ch)>0 else 0
    offenders = fam_sizes.loc[not_iso_ch][fam_sizes.loc[not_iso_ch] > threshold].sort_values(ascending=False)
    print(f"\npost-split 阈值（max_fraction vs max_size）：{threshold}")
    print("几何家族最大规模（排除 iso/ch）：", int(max_fam))
    if len(offenders)>0:
        print("⚠ 仍超阈的家族：")
        print(offenders.to_string())

    # duplicate_canonical 一致性
    if "duplicate_canonical" in df.columns:
        grp = df.groupby("duplicate_canonical")["family_id"].nunique().sort_values(ascending=False)
        bad = grp[grp>1]
        print(f"\nduplicate_canonical 覆盖数：{grp.size}, 其中 family 不唯一的 canonical 数：{bad.size}")
        if bad.size>0:
            print("⚠ 需要关注的 canonical（family_id>1）：")
            print(bad.head(20).to_string())
    else:
        print("\n未发现 duplicate_canonical 列，跳过一致性检查。")

    # 无几何样本（has_points==0）的 fam_ch_ 规则
    if "has_points" in df.columns:
        df_no = df[df["has_points"]==0]
        if len(df_no)>0:
            ok = df_no["family_id"].str.startswith("fam_ch_", na=False).mean()
            print(f"\nhas_points==0 数量：{len(df_no)}，其中命名为 fam_ch_* 的比例：{ok:.3f}")
        else:
            print("\n无 has_points==0 样本。")

    # eps 合理性与设备
    print("\n=== eps / 设备 ===")
    print(f"fallback eps (sqrt(2*Tc)) = {eps_fb:.6f}  (Tc={Tc})")
    if loginfo:
        eps_auto = loginfo.get("eps", loginfo.get("eps2"))
        if eps_auto:
            ratio = eps_auto / eps_fb if eps_fb>0 else float('nan')
            print(f"auto eps = {eps_auto:.6f} ； 相对 fallback 比例 = {ratio:.3f}")
        dev = loginfo.get("device")
        if dev: print("run.log 设备：", dev)
        wg = loginfo.get("with_geom")
        if wg: print("with_geom（日志报告）:", wg)

if __name__=="__main__":
    main()
