#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2.2: 去重（内容哈希 + 几何近邻），输出 duplicate_groups.json 与 dup2canon.json

保持原逻辑：先按 content_hash 并查集，再对有几何者用 D2 特征（L2规范化，余弦相似 = 1 - cosine distance），
阈值 d2_sim_threshold 与 bbox_tol 一致；kNN 搜索优先使用 FAISS-GPU。
"""

import os, sys, json, time, logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# logging + progress
def _to_level(level: str):
    import logging as _lg
    return {"debug": _lg.DEBUG, "info": _lg.INFO, "warning": _lg.WARNING}.get((level or "info").lower(), _lg.INFO)
def setup_logger(level: str="info", log_file: Optional[str]=None)->logging.Logger:
    logger = logging.getLogger("S2_2"); logger.setLevel(_to_level(level)); logger.propagate=False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s","%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh=logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh=logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

class ProgressPrinter:
    def __init__(self,total:int,prefix:str=""): self.total=max(1,int(total)); self.prefix=prefix; self.last=-1; self.done=False
    def _emit(self,p): print(f"\r{self.prefix} {p:3d}%", end="", flush=True)
    def print_start(self): self._emit(0); self.last=0
    def update(self,cur:int):
        if self.done: return
        p = int(min(max(0,cur), self.total)*100/self.total)
        if p>=self.last+1: self._emit(p); self.last=p
        if cur>=self.total: self.finish()
    def finish(self):
        if not self.done: self._emit(100); print(); self.done=True

# GPU FAISS
faiss=None; _GPU_FAISS=False
def _ensure_faiss():
    global faiss, _GPU_FAISS
    if faiss is not None: return
    try:
        import faiss as _faiss
        faiss=_faiss
        try: _ = _faiss.StandardGpuResources(); _GPU_FAISS=True
        except Exception: _GPU_FAISS=False
    except Exception:
        faiss=None; _GPU_FAISS=False

# sklearn fallback
from sklearn.neighbors import NearestNeighbors

class DSU:
    def __init__(self, items: List[str]): self.p={x:x for x in items}; self.r={x:0 for x in items}
    def find(self,x:str)->str:
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a:str,b:str):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: self.p[ra]=rb
        elif self.r[ra]>self.r[rb]: self.p[rb]=ra
        else: self.p[rb]=ra; self.r[ra]+=1

def _l2_normalize(X: np.ndarray)->np.ndarray:
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)

def pack_feat(row)->np.ndarray:
    d2 = np.array(json.loads(row["d2_hist"]), dtype=np.float32) if pd.notna(row["d2_hist"]) else None
    br = np.array(json.loads(row["bbox_ratio"]), dtype=np.float32) if pd.notna(row["bbox_ratio"]) else None
    surf = np.array(json.loads(row["surf_hist"]), dtype=np.float32) if pd.notna(row["surf_hist"]) else None
    dih = np.array(json.loads(row["dih_hist"]), dtype=np.float32) if pd.notna(row["dih_hist"]) else None
    return np.concatenate([d2, br, surf, dih]).astype(np.float32)

def load_config(path:str)->Tuple[dict,dict]:
    with open(path,"r",encoding="utf-8") as f: allcfg=json.load(f)
    return allcfg.get("s2_dedup_family_occ",{}), allcfg.get("log",{})

def main():
    if len(sys.argv)<2 or sys.argv[1] in ("-h","--help"):
        print("用法: python s2_2_dedup.py <config.json>"); sys.exit(0)
    cfg_path=sys.argv[1]
    s2cfg, log_cfg = load_config(cfg_path)
    out_root  = s2cfg.get("out_root")
    device    = s2cfg.get("device","auto")
    d2_sim_threshold = float(s2cfg.get("d2_sim_threshold",0.995))
    bbox_tol  = float(s2cfg.get("bbox_tol",0.02))
    log_level = s2cfg.get("log_level","info")
    log_file  = s2cfg.get("log_file") or (log_cfg.get("file"))
    sig_csv   = os.path.join(out_root,"signatures.csv")
    logger = setup_logger(log_level, log_file)

    logger.info(f"[S2.2] boot config={os.path.abspath(cfg_path)}")
    logger.info(f"[S2.2] inputs: signatures.csv={sig_csv}")
    if not os.path.isfile(sig_csv):
        logger.error("缺少 signatures.csv，请先运行 S2.1"); sys.exit(1)

    df = pd.read_csv(sig_csv)
    ids = df["part_id"].tolist()
    dsu = DSU(ids)

    # a) 内容哈希去重（完全一致）
    by_ch={}
    for _,r in df.iterrows():
        by_ch.setdefault(r["content_hash"], []).append(r["part_id"])
    for ch, lst in by_ch.items():
        if len(lst)>1:
            canon = lst[0]
            for x in lst[1:]: dsu.union(canon, x)
    logger.info(f"[S2.2] content-hash groups merged={sum(1 for v in by_ch.values() if len(v)>1)}")

    # b) 几何近邻去重（有几何者）
    df_geom = df[(df["has_points"]==1) & df["d2_hist"].notna()]
    merged_pairs=0
    if len(df_geom)>0:
        F = np.stack([pack_feat(r) for _,r in df_geom.iterrows()]).astype(np.float32)
        F = _l2_normalize(F)
        pids = df_geom["part_id"].tolist()

        use_gpu = (device=="gpu")
        prog = ProgressPrinter(len(pids), prefix="[S2.2]")
        prog.print_start()
        if use_gpu:
            _ensure_faiss()
        if use_gpu and _GPU_FAISS and faiss is not None:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, F.shape[1]); index.add(F)
            k = min(10, F.shape[0]); sims, idxs = index.search(F, k)
            for i, pid_i in enumerate(pids):
                for sim, jpos in zip(sims[i], idxs[i]):
                    if jpos==i or jpos<0: continue
                    pid_j = pids[jpos]
                    # bbox 容差（直接从csv读取）
                    br_i = np.array(json.loads(df_geom.iloc[i]["bbox_ratio"]), dtype=np.float32)
                    br_j = np.array(json.loads(df_geom.iloc[jpos]["bbox_ratio"]), dtype=np.float32)
                    br_ok = float(np.max(np.abs(br_i - br_j))) <= bbox_tol
                    if float(sim) >= d2_sim_threshold and br_ok:
                        if dsu.find(pid_i)!=dsu.find(pid_j):
                            dsu.union(pid_i, pid_j); merged_pairs += 1
                prog.update(i+1)
        else:
            nbrs = NearestNeighbors(metric="cosine", n_neighbors=min(10, len(F))).fit(F)
            distances, indices = nbrs.kneighbors(F, return_distance=True)
            for i, pid_i in enumerate(pids):
                for d, jpos in zip(distances[i], indices[i]):
                    if jpos==i: continue
                    pid_j = pids[jpos]
                    sim = 1.0 - float(d)
                    br_i = np.array(json.loads(df_geom.iloc[i]["bbox_ratio"]), dtype=np.float32)
                    br_j = np.array(json.loads(df_geom.iloc[jpos]["bbox_ratio"]), dtype=np.float32)
                    br_ok = float(np.max(np.abs(br_i - br_j))) <= bbox_tol
                    if sim >= d2_sim_threshold and br_ok:
                        if dsu.find(pid_i)!=dsu.find(pid_j):
                            dsu.union(pid_i, pid_j); merged_pairs += 1
                prog.update(i+1)
        prog.finish()
    logger.info(f"[S2.2] geom-merged-pairs={merged_pairs}")

    # 汇总输出
    dup2canon={}; groups={}
    for pid in ids:
        root = dsu.find(pid); dup2canon[pid]=root; groups.setdefault(root,[]).append(pid)

    dup_groups_json = os.path.join(out_root,"duplicate_groups.json")
    with open(dup_groups_json,"w",encoding="utf-8") as f: json.dump(groups, f, indent=2, ensure_ascii=False)
    logger.info(f"[S2.2] Output duplicate_groups.json -> {dup_groups_json}")

    dup2canon_json = os.path.join(out_root,"dup2canon.json")
    with open(dup2canon_json,"w",encoding="utf-8") as f: json.dump(dup2canon, f, indent=2, ensure_ascii=False)
    logger.info(f"[S2.2] Output dup2canon.json -> {dup2canon_json}")

    summary = {
        "num_parts": len(ids),
        "duplicate_groups": sum(1 for v in groups.values() if len(v)>1),
        "geom_merged_pairs": merged_pairs,
        "gpu_available": {"faiss": _GPU_FAISS},
        "out_files": {"duplicate_groups": dup_groups_json, "dup2canon": dup2canon_json}
    }
    out_sum = os.path.join(out_root,"dedup_summary.json")
    with open(out_sum,"w",encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"[S2.2] Output dedup_summary.json -> {out_sum}")
    logger.info("[S2.2] DONE")

if __name__=="__main__":
    main()
