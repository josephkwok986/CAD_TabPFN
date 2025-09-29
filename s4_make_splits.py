#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s4_make_splits.py

改造要点：
1) 所有输入参数从 config.json 读取，且仅读取 config.json 中的 **s4_make_splits** 部分；
   命令行只保留 --config 参数。
2) 支持日志写入：按 config.json 中 logging.log_file 指定的路径写入；关键路径均以 INFO 级别打印“关键日志”。
3) 控制台打印进度（最小粒度 1%，数据量少时按步进增加粒度）。
4) 提升性能但不改变任何原有逻辑：
   - 预先缓存 domain -> dataframe 的映射，避免 LODO 循环中重复 groupby。
   - 并行写文件（仅 I/O，不涉及随机性），不影响逻辑与确定性。
5) 新增功能：把 duplicate_canonical / content_hash 作为“强连边”（强组），保证同组不跨 train/calib/test。
   - 若两列均不存在，则退化为“每个样本自成一组”，行为与旧版完全一致。
"""

import argparse
import json
import logging
import os
import sys
import math
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:
    _CUDA = False


# =========================
# 数据类（与原逻辑保持一致 + 新增强连边列名）
# =========================
@dataclass
class Args:
    input: str
    outdir: str
    protocol: str
    part_col: str
    domain_col: str
    timestamp_col: str
    family_col: str
    calib_frac: float
    train_frac: float
    seed: int
    min_domain_size: int
    per_domain_temporal: bool
    stratify_col: str
    part_index_dir: str
    num_workers: int = 4  # 并行写文件的 worker 数
    duplicate_col: str = "duplicate_canonical"   # 新增
    content_hash_col: str = "content_hash"       # 新增


# =========================
# 工具：日志 & 进度条
# =========================
def setup_logger(log_file: str, level: str = "INFO"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("s4_make_splits")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # 控制台也输出最关键信息
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


class ConsoleProgress:
    """
    控制台进度打印（覆盖行，最小粒度 1%，数据量少时每步打印）。
    """
    def __init__(self, total_steps: int, title: str = "Progress"):
        self.total = max(1, int(total_steps))
        self.title = title
        self.last_pct = -1
        # 若总步数 < 100，则按每步打印；否则按 1% 打印
        self.step_print_every = 1 if self.total < 100 else max(1, self.total // 100)

    def update(self, current_step: int):
        current_step = min(self.total, max(0, int(current_step)))
        if self.total >= 100:
            pct = int(current_step * 100 / self.total)
            if pct > self.last_pct:
                self.last_pct = pct
                bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                sys.stdout.write(f"\r{self.title}: {pct:3d}% |{bar}|")
                sys.stdout.flush()
        else:
            # 总步数 < 100 时，每步打印一次
            if (current_step % self.step_print_every == 0) or (current_step == self.total):
                pct = int(current_step * 100 / self.total)
                bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                sys.stdout.write(f"\r{self.title}: {pct:3d}% |{bar}|")
                sys.stdout.flush()

    def finish(self):
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()


# =========================
# 原有辅助函数（保持逻辑不变）
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def check_required_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

def to_datetime_series(s: pd.Series):
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    return dt

def _write_list_single(path, ids):
    with open(path, "w", encoding="utf-8") as f:
        for x in ids:
            f.write(f"{x}\n")

def write_list_parallel(base_dir: str, parts_train, parts_calib, parts_test, num_workers: int = 4):
    """
    仅对 I/O 使用并行写入，不改变任何结果（只加速落盘）。
    """
    ensure_dir(base_dir)
    tasks = [
        (os.path.join(base_dir, "train.txt"), parts_train),
        (os.path.join(base_dir, "calib.txt"), parts_calib),
        (os.path.join(base_dir, "test.txt"),  parts_test),
    ]
    if max(len(parts_train), len(parts_calib), len(parts_test)) == 0:
        # 极小数据时，直接写（避免线程开销）
        for pth, ids in tasks:
            _write_list_single(pth, ids)
        return

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futs = [ex.submit(_write_list_single, pth, ids) for pth, ids in tasks]
        for _ in as_completed(futs):
            pass

def render_report_md(meta: dict) -> str:
    md = ["# Split Report", ""]
    md.append(f"- Protocol: **{meta['protocol']}**")
    md.append(f"- Seed: **{meta['seed']}**")
    md.append(f"- Input rows: **{meta['n_rows']}**, unique parts: **{meta['n_parts']}**")
    if 'n_groups' in meta:
        md.append(f"- Strong groups: **{meta['n_groups']}** (by duplicate_canonical/content_hash)")
    if 'timestamp_info' in meta:
        tinfo = meta['timestamp_info']
        md.append(f"- Timestamp range: **{tinfo['min']} → {tinfo['max']}**")
    if meta.get('strong_groups_disjoint') is not None:
        md.append(f"- Strong groups disjoint: **{meta['strong_groups_disjoint']}**")
    md.append("")
    if meta['protocol'] == 'lodo':
        md.append("## LODO folds")
        for fold in meta['folds']:
            md.append(f"### Test domain: `{fold['domain']}`")
            md.append(f"- Sizes: train **{fold['n_train']}**, calib **{fold['n_calib']}**, test **{fold['n_test']}**")
            md.append(f"- Families disjoint: **{fold['families_disjoint']}**")
            if 'strong_groups_disjoint' in fold:
                md.append(f"- Strong groups disjoint: **{fold['strong_groups_disjoint']}**")
            md.append("")
    else:
        md.append("## Temporal split")
        md.append(f"- Fractions: train **{meta['train_frac']}**, calib **{meta['calib_frac']}**, test **{meta['test_frac']}**")
        md.append(f"- Sizes: train **{meta['n_train']}**, calib **{meta['n_calib']}**, test **{meta['n_test']}**")
        if meta.get('per_domain_temporal'):
            md.append("- Per-domain temporal slicing: **True**")
        md.append("")
    return "\n".join(md)

def load_part_index(path_or_dir: str):
    if not path_or_dir:
        return None
    if os.path.isdir(path_or_dir):
        candidates = [os.path.join(path_or_dir, "part_index.csv"),
                      os.path.join(path_or_dir, "part_index.csv.gz")]
    else:
        candidates = [path_or_dir]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def merge_cols_from_part_index(dom_df: pd.DataFrame, pi_df: pd.DataFrame, part_col: str, cols_to_merge):
    """
    将缺失的列从 part_index 中补齐（若存在）。
    """
    if pi_df is None:
        return dom_df, {c: False for c in cols_to_merge}
    merged_flags = {}
    out = dom_df
    for c in cols_to_merge:
        if c and ((c not in out.columns) or out[c].isna().all()):
            if c in pi_df.columns:
                out = out.merge(pi_df[[part_col, c]], on=part_col, how="left", suffixes=("", "_pi"))
                # 如已存在旧列且全空，则用 _pi 覆盖
                if f"{c}_pi" in out.columns:
                    need_fill = out[c].isna()
                    out.loc[need_fill, c] = out.loc[need_fill, f"{c}_pi"]
                    out = out.drop(columns=[col for col in out.columns if col.endswith("_pi")])
                merged_flags[c] = True
            else:
                merged_flags[c] = False
        else:
            merged_flags[c] = False
    return out, merged_flags

def merge_family_id(dom_df: pd.DataFrame, pi_df: pd.DataFrame, part_col: str, family_col: str):
    if pi_df is None or not family_col:
        return dom_df, False
    if family_col in dom_df.columns and dom_df[family_col].notna().sum() > 0:
        if family_col in pi_df.columns:
            merged = dom_df.merge(pi_df[[part_col, family_col]], on=part_col, how="left", suffixes=("", "_pi"))
            need_fill = merged[family_col].isna()
            merged.loc[need_fill, family_col] = merged.loc[need_fill, f"{family_col}_pi"]
            merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_pi")])
            return merged, True
        else:
            return dom_df, False
    else:
        if family_col in pi_df.columns:
            merged = dom_df.merge(pi_df[[part_col, family_col]], on=part_col, how="left")
            return merged, True
        else:
            return dom_df, False

def enforce_family_disjoint(train_ids, calib_ids, test_ids, df, part_col, family_col):
    def fams_for(ids):
        return set(df.loc[df[part_col].isin(ids), family_col].dropna().unique().tolist())
    ok = True
    if family_col and family_col in df.columns:
        train_f = fams_for(train_ids)
        calib_f = fams_for(calib_ids)
        test_f  = fams_for(test_ids)
        if train_f & calib_f or train_f & test_f or calib_f & test_f:
            ok = False
    return ok

def enforce_strong_disjoint(train_ids, calib_ids, test_ids, part_to_group):
    """
    检查强组是否被拆跨侧。
    """
    def groups_for(ids):
        return set(part_to_group.get(x) for x in ids if x in part_to_group)
    g_train = groups_for(train_ids)
    g_calib = groups_for(calib_ids)
    g_test  = groups_for(test_ids)
    if (g_train & g_calib) or (g_train & g_test) or (g_calib & g_test):
        return False
    return True


# =========================
# 新增：强连边（并查集构组）
# =========================
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

def build_strong_groups(df: pd.DataFrame, part_col: str, duplicate_col: str, content_hash_col: str) -> pd.Series:
    """
    基于 duplicate_canonical / content_hash 构建强组（并查集）。
    若两列都不存在或全空，则每个样本自成一组（组ID=本身 part_id）。
    返回：Series(index=part_id, value=group_id)
    """
    parts = df[part_col].astype(str).tolist()
    uf = UnionFind()
    # 先把每个 part 初始化
    for pid in parts:
        uf.find(pid)

    # 以 duplicate_canonical 连边
    if duplicate_col and (duplicate_col in df.columns):
        for val, sub in df.loc[df[duplicate_col].notna(), [part_col, duplicate_col]].groupby(duplicate_col):
            ids = sub[part_col].astype(str).tolist()
            if len(ids) > 1:
                base = ids[0]
                for x in ids[1:]:
                    uf.union(base, x)

    # 以 content_hash 连边
    if content_hash_col and (content_hash_col in df.columns):
        for val, sub in df.loc[df[content_hash_col].notna(), [part_col, content_hash_col]].groupby(content_hash_col):
            ids = sub[part_col].astype(str).tolist()
            if len(ids) > 1:
                base = ids[0]
                for x in ids[1:]:
                    uf.union(base, x)

    # 输出 root 作为 group_id
    group_ids = {pid: uf.find(pid) for pid in parts}
    return pd.Series(group_ids, name="_strong_group")

def groups_index(df: pd.DataFrame, part_col: str, group_col: str):
    """
    返回：
    - part_to_group: dict(part_id -> group_id)
    - group_to_parts: dict(group_id -> [part_ids])
    """
    part_to_group = dict(df[[part_col, group_col]].astype(str).values)
    g2p = defaultdict(list)
    for pid, gid in part_to_group.items():
        g2p[gid].append(pid)
    return part_to_group, g2p


# =========================
# 主要逻辑（保持不变 + 关键日志 + 加速 + 强连边）
# =========================
def lodo_split(df: pd.DataFrame, args: Args, logger: logging.Logger):
    """
    LODO 切分（留一域）：
    - 维持原有逻辑与输出格式；
    - 保持强组（_strong_group）不跨侧（若该列存在）；
    - family 强约束：测试域中的 family 不得出现在 train/calib；
    - 额外：将 family_id 以 'fam_iso_' 开头的家族（DBSCAN 噪声单件）从 train/calib 剔除，仅在其所在域为 test 时出现。
    """
    ISO_PREFIX = "fam_iso_"  # 仅留在 test 的 family 前缀

    rng = np.random.default_rng(args.seed)
    domains = df[args.domain_col].dropna().unique().tolist()
    folds_meta = []
    out_base = os.path.join(args.outdir, "lodo")
    ensure_dir(out_base)

    # 预缓存每个 domain 的子表
    domain_groups = {dom: df.loc[df[args.domain_col] == dom, :].copy() for dom in domains}

    # 强组相关（若没有 _strong_group 列则自动退化）
    has_strong = ("_strong_group" in df.columns)
    if has_strong:
        part_to_group, group_to_parts = groups_index(df, args.part_col, "_strong_group")
    else:
        part_to_group, group_to_parts = {}, {}

    logger.info(f"[LODO] 开始 LODO 切分。domains={len(domains)} 输出目录={out_base}")
    logger.info(f"[LODO] seed={args.seed} calib_frac={args.calib_frac} min_domain_size={args.min_domain_size}")

    total_steps = len(domains) + 1
    pb = ConsoleProgress(total_steps, title="LODO splitting")

    k = 0
    processed = 0
    for dom in sorted(domains):
        test_df = domain_groups.get(dom, pd.DataFrame())
        test_ids = test_df[args.part_col].astype(str)

        if len(test_ids) < args.min_domain_size:
            processed += 1
            pb.update(processed)
            continue

        # 强组集合
        test_groups = set(test_df["_strong_group"].astype(str).unique().tolist()) if has_strong else set()

        # 测试域 family 集合
        test_fams = set()
        if args.family_col and (args.family_col in df.columns):
            test_fams = set(test_df[args.family_col].dropna().astype(str).unique().tolist())

        # 训练池：其它 domains 的并集
        train_domains = [d for d in sorted(domains) if d != dom]
        train_pool = pd.concat([domain_groups[d] for d in train_domains], axis=0, ignore_index=True)

        # 先按强组剔除与 test 相交的组
        if has_strong and test_groups:
            train_pool = train_pool.loc[~train_pool["_strong_group"].astype(str).isin(test_groups)].reset_index(drop=True)

        # 再按 family 剔除与 test 同 family 的样本
        if test_fams and (args.family_col in train_pool.columns):
            train_pool = train_pool.loc[~train_pool[args.family_col].astype(str).isin(test_fams)].reset_index(drop=True)

        # 额外：将 fam_iso_* 从 train_pool 剔除（仅留在 test）
        if args.family_col in train_pool.columns:
            train_pool = train_pool.loc[~train_pool[args.family_col].astype(str).str.startswith(ISO_PREFIX)].reset_index(drop=True)

        # —— 校准抽样（保持原逻辑），但同样应用两类过滤 + ISO 剔除 —— #
        calib_ids = []
        if args.family_col and (args.family_col in df.columns):
            for d in train_domains:
                g = domain_groups[d]

                if has_strong and test_groups:
                    g = g.loc[~g["_strong_group"].astype(str).isin(test_groups)]
                if test_fams and (args.family_col in g.columns):
                    g = g.loc[~g[args.family_col].astype(str).isin(test_fams)]
                if args.family_col in g.columns:
                    g = g.loc[~g[args.family_col].astype(str).str.startswith(ISO_PREFIX)]

                if g.empty:
                    continue

                fams = g[args.family_col].dropna().unique().tolist()
                if len(fams) == 0:
                    n = max(1, int(math.ceil(len(g) * args.calib_frac)))
                    sel = g.sample(n=n, random_state=args.seed, replace=False)[args.part_col].astype(str).tolist()
                else:
                    n_families = max(1, int(math.ceil(len(fams) * args.calib_frac)))
                    n_families = min(n_families, len(fams))
                    sel_fams = rng.choice(fams, size=n_families, replace=False).tolist()
                    sel = g[g[args.family_col].isin(sel_fams)][args.part_col].astype(str).tolist()
                calib_ids.extend(sel)
        else:
            for d in train_domains:
                g = domain_groups[d]
                if has_strong and test_groups:
                    g = g.loc[~g["_strong_group"].astype(str).isin(test_groups)]
                if test_fams and (args.family_col in g.columns):
                    g = g.loc[~g[args.family_col].astype(str).isin(test_fams)]
                # 无 family 列时无法按 ISO 剔除（忽略）

                if g.empty:
                    continue
                n = max(1, int(math.ceil(len(g) * args.calib_frac)))
                sel = g.sample(n=n, random_state=args.seed, replace=False)[args.part_col].astype(str).tolist()
                calib_ids.extend(sel)

        # 去重保持首次出现
        seen = set()
        calib_ids = [x for x in calib_ids if not (x in seen or seen.add(x))]

        # 强组扩张：把被选中的样本所在强组的全部成员并入 calib，再次过滤 ISO 与 test_fams
        if has_strong:
            calib_groups = set(part_to_group.get(x) for x in calib_ids if x in part_to_group)
            expanded = []
            for gid in calib_groups:
                expanded.extend(group_to_parts.get(gid, []))
            if args.family_col in df.columns:
                fam_map = dict(zip(df[args.part_col].astype(str), df[args.family_col].astype(str)))
                expanded = [pid for pid in expanded if fam_map.get(pid) not in test_fams
                            and not str(fam_map.get(pid, "")).startswith(ISO_PREFIX)]
            seen = set()
            calib_ids = [x for x in expanded if not (x in seen or seen.add(x))]

        # 训练集 = 训练池中去掉 calib
        calib_set = set(calib_ids)
        train_ids = train_pool.loc[~train_pool[args.part_col].astype(str).isin(calib_set), args.part_col].astype(str).tolist()

        split_dir = os.path.join(out_base, f"split_{k}")
        ensure_dir(split_dir)
        write_list_parallel(split_dir, train_ids, calib_ids, test_ids.astype(str).tolist(), num_workers=args.num_workers)

        families_disjoint = enforce_family_disjoint(train_ids, calib_ids, test_ids.tolist(), df, args.part_col, args.family_col)
        strong_groups_disjoint = True
        if has_strong:
            strong_groups_disjoint = enforce_strong_disjoint(train_ids, calib_ids, test_ids.astype(str).tolist(), part_to_group)

        folds_meta.append({
            "domain": dom,
            "n_train": int(len(train_ids)),
            "n_calib": int(len(calib_ids)),
            "n_test": int(test_ids.shape[0]),
            "families_disjoint": families_disjoint,
            "strong_groups_disjoint": strong_groups_disjoint
        })
        logger.info(f"[LODO][fold={k}] test_domain={dom} | train={len(train_ids)} calib={len(calib_ids)} test={int(test_ids.shape[0])}")
        logger.info(f"[LODO][fold={k}] families_disjoint={families_disjoint} strong_groups_disjoint={strong_groups_disjoint} | 输出目录={split_dir}")

        k += 1
        processed += 1
        pb.update(processed)

    meta = {
        "protocol": "lodo",
        "seed": args.seed,
        "n_rows": int(df.shape[0]),
        "n_parts": int(df[args.part_col].nunique()),
        "folds": folds_meta
    }
    if has_strong:
        meta["n_groups"] = int(df["_strong_group"].nunique())

    report_path = os.path.join(out_base, "split_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report_md(meta))

    pb.update(total_steps)
    pb.finish()

    logger.info(f"[LODO] 共写出 {k} 个 folds。报告：{report_path}")
    print(f"[LODO] Wrote {k} folds to: {out_base}")


def temporal_split(df: pd.DataFrame, args: Args, logger: logging.Logger):
    """
    时间外切分（旧→新）：
    - 以不可拆单元切分：优先 family_id，其次 _strong_group，再次 part_id
    - fam_iso_* 默认全部进 test；但可通过 iso_relax_frac 放宽：按最早时间抽一小部分到 calib，再到 train
      * iso_relax_frac：从 iso 单元中最多抽走的比例（默认 0.10）
      * iso_relax_to_calib_frac：抽走的 iso 中分配到 calib 的比例（默认 0.60），其余进 train
    - per_domain_temporal=True 时逐域切，同一不可拆单元跨域保持同侧
    """
    ISO_PREFIX = "fam_iso_"
    iso_relax_frac = float(getattr(args, "iso_relax_frac", 0.10))            # 可在 config 里配置
    iso_relax_to_calib_frac = float(getattr(args, "iso_relax_to_calib_frac", 0.60))

    if args.timestamp_col not in df.columns:
        raise ValueError(f"Temporal split requires timestamp column: {args.timestamp_col}")

    logger.info(f"[Temporal] 开始时间外切分。timestamp_col={args.timestamp_col} "
                f"train_frac={args.train_frac} calib_frac={args.calib_frac} "
                f"per_domain_temporal={args.per_domain_temporal} "
                f"iso_relax_frac={iso_relax_frac} iso_relax_to_calib_frac={iso_relax_to_calib_frac}")

    # 1) 解析时间
    ts = to_datetime_series(df[args.timestamp_col])
    if ts.isna().all():
        raise ValueError("All timestamps are NaT. Please provide a parseable timestamp column.")
    df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

    # 2) 不可拆单元：family -> _strong_group -> part_id
    if args.family_col and (args.family_col in df.columns) and df[args.family_col].notna().any():
        unit_col = args.family_col
        logger.info(f"[Temporal] 使用 family 作为不可拆单元：{args.family_col}")
    elif "_strong_group" in df.columns and df["_strong_group"].notna().any():
        unit_col = "_strong_group"
        logger.info("[Temporal] 未提供/全空 family，退化为 _strong_group 作为不可拆单元")
    else:
        unit_col = args.part_col
        logger.info("[Temporal] 未提供 _strong_group，退化为按 part_id 作为不可拆单元")

    df["_unit"] = df[unit_col].astype(str)
    mask_na = df["_unit"].isin(["nan", "None", "NaT"])
    if mask_na.any():
        df.loc[mask_na, "_unit"] = df.loc[mask_na, args.part_col].astype(str)

    # 3) unit 的最早时间
    unit_ts = df.groupby("_unit")["_ts"].min().reset_index().rename(columns={"_ts": "_unit_ts"})
    n_units = unit_ts.shape[0]
    logger.info(f"[Temporal] 不可拆单元数量 n_units={n_units}")

    # 4) fam_iso_* 单元（仅当 unit=family 时有效）
    iso_units = set()
    if unit_col == args.family_col:
        fam_vals = df[[args.family_col, "_unit"]].dropna()
        iso_units = set(
            fam_vals.loc[fam_vals[args.family_col].astype(str).str.startswith(ISO_PREFIX),
                         "_unit"].astype(str).unique().tolist()
        )
        logger.info(f"[Temporal] fam_iso_* 单元数={len(iso_units)}（默认全部进 test，可按比例放宽）")

    # 5) 分配函数：按时间把 unit 切到 train/calib/test
    def assign_units_by_time(units_sorted: list[str]):
        m = len(units_sorted)
        n_train = max(1, int(np.floor(m * args.train_frac)))
        n_calib = max(1, int(np.floor(m * args.calib_frac)))
        n_test  = m - n_train - n_calib
        if n_test < 1:
            n_test = 1
            if n_calib > 1: n_calib -= 1
            elif n_train > 1: n_train -= 1
        return (units_sorted[:n_train],
                units_sorted[n_train:n_train+n_calib],
                units_sorted[n_train+n_calib:])

    pb = ConsoleProgress(8, title="Temporal splitting")
    step = 0

    # 映射 unit -> 所有 part_id
    unit_to_parts = df.groupby("_unit")[args.part_col].apply(lambda s: s.astype(str).tolist()).to_dict()

    if args.per_domain_temporal and (args.domain_col in df.columns):
        # 逐域；先把 iso_units 预设为 test；之后按放宽参数改派部分 iso 到 calib/train
        assigned = {}  # unit -> side {"train","calib","test"}

        domains = sorted(df[args.domain_col].dropna().astype(str).unique().tolist())
        for dom in domains:
            g = df[df[args.domain_col].astype(str) == dom]
            dom_units = sorted(g["_unit"].astype(str).unique().tolist())

            # 先把 iso_units 标为 test（若适用）
            if iso_units:
                for u in dom_units:
                    if (u in iso_units) and (u not in assigned):
                        assigned[u] = "test"

            # 未分配的单元按时间切
            to_assign = [u for u in dom_units if u not in assigned]
            if to_assign:
                sub = unit_ts[unit_ts["_unit"].isin(to_assign)].sort_values("_unit_ts")
                u_sorted = sub["_unit"].astype(str).tolist()
                u_tr, u_ca, u_te = assign_units_by_time(u_sorted)
                for u in u_tr: assigned[u] = "train"
                for u in u_ca: assigned[u] = "calib"
                for u in u_te: assigned[u] = "test"
            step += 1; pb.update(step)

        # —— 放宽：把最早的部分 iso_units 改派到 calib/train —— #
        if iso_units and iso_relax_frac > 0:
            iso_sorted = (
                unit_ts[unit_ts["_unit"].isin(iso_units)]
                .sort_values("_unit_ts")["_unit"].astype(str).tolist()
            )
            k_total = int(np.ceil(len(iso_sorted) * max(0.0, min(1.0, iso_relax_frac))))
            k_calib = int(np.ceil(k_total * max(0.0, min(1.0, iso_relax_to_calib_frac))))
            move_calib = set(iso_sorted[:k_calib])
            move_train = set(iso_sorted[k_calib:k_total])
            for u in move_calib: assigned[u] = "calib"
            for u in move_train: assigned[u] = "train"
            logger.info(f"[Temporal] 放宽 iso：total={len(iso_sorted)} moved={k_total} "
                        f"(calib={len(move_calib)}, train={len(move_train)})")

        # 展开为 part 列表
        parts_train, parts_calib, parts_test = [], [], []
        for u, side in assigned.items():
            if side == "train": parts_train.extend(unit_to_parts.get(u, []))
            elif side == "calib": parts_calib.extend(unit_to_parts.get(u, []))
            else: parts_test.extend(unit_to_parts.get(u, []))
    else:
        # 全局：先把非 iso 单元按时间切；iso 单元默认并入 test，再按参数放宽一部分到 calib/train
        u_all = unit_ts.sort_values("_unit_ts")["_unit"].astype(str).tolist()
        u_main = [u for u in u_all if u not in iso_units]  # 非 ISO
        u_tr, u_ca, u_te = assign_units_by_time(u_main)
        parts_train = sum((unit_to_parts.get(u, []) for u in u_tr), [])
        step += 1; pb.update(step)
        parts_calib = sum((unit_to_parts.get(u, []) for u in u_ca), [])
        step += 1; pb.update(step)
        parts_test  = sum((unit_to_parts.get(u, []) for u in u_te), [])

        # iso 默认进 test
        for u in sorted(iso_units):
            parts_test.extend(unit_to_parts.get(u, []))

        # —— 放宽：把最早的部分 iso_units 改派到 calib/train —— #
        if iso_units and iso_relax_frac > 0:
            iso_sorted = (
                unit_ts[unit_ts["_unit"].isin(iso_units)]
                .sort_values("_unit_ts")["_unit"].astype(str).tolist()
            )
            k_total = int(np.ceil(len(iso_sorted) * max(0.0, min(1.0, iso_relax_frac))))
            k_calib = int(np.ceil(k_total * max(0.0, min(1.0, iso_relax_to_calib_frac))))
            move_calib = iso_sorted[:k_calib]
            move_train = iso_sorted[k_calib:k_total]

            # 从 test 移除这些单元，再追加到 calib/train
            to_remove = set(sum((unit_to_parts.get(u, []) for u in move_calib + move_train), []))
            parts_test = [p for p in parts_test if p not in to_remove]
            for u in move_calib:
                parts_calib.extend(unit_to_parts.get(u, []))
            for u in move_train:
                parts_train.extend(unit_to_parts.get(u, []))

            logger.info(f"[Temporal] 放宽 iso：total={len(iso_sorted)} moved={k_total} "
                        f"(calib={len(move_calib)}, train={len(move_train)})")
        step += 1; pb.update(step)

    # 去重（稳妥起见）
    parts_train = list(dict.fromkeys(parts_train))
    parts_calib = list(dict.fromkeys(parts_calib))
    parts_test  = list(dict.fromkeys(parts_test))

    # 6) 落盘 & 报告
    out_base = os.path.join(args.outdir, "temporal", "split_0")
    ensure_dir(out_base)
    write_list_parallel(out_base, parts_train, parts_calib, parts_test, num_workers=args.num_workers)
    step += 1; pb.update(step)

    # 强组一致性检查
    has_strong = ("_strong_group" in df.columns)
    strong_groups_disjoint = True
    if has_strong:
        part_to_group = dict(zip(df[args.part_col].astype(str), df["_strong_group"].astype(str)))
        strong_groups_disjoint = enforce_strong_disjoint(parts_train, parts_calib, parts_test, part_to_group)

    meta = {
        "protocol": "temporal",
        "seed": args.seed,
        "n_rows": int(df.shape[0]),
        "n_parts": int(df[args.part_col].nunique()),
        "n_groups": int(df["_strong_group"].nunique()) if has_strong else None,
        "train_frac": args.train_frac,
        "calib_frac": args.calib_frac,
        "test_frac": round(1.0 - args.train_frac - args.calib_frac, 6),
        "n_train": len(parts_train),
        "n_calib": len(parts_calib),
        "n_test": len(parts_test),
        "timestamp_info": {"min": str(df["_ts"].min()), "max": str(df["_ts"].max())},
        "per_domain_temporal": args.per_domain_temporal,
        "strong_groups_disjoint": strong_groups_disjoint,
    }
    report_dir = os.path.join(args.outdir, "temporal")
    ensure_dir(report_dir)
    report_path = os.path.join(report_dir, "split_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report_md(meta))
    step += 1; pb.update(step)

    pb.finish()

    logger.info(f"[Temporal] 写出 split 到：{out_base}；报告：{report_path}；"
                f"strong_groups_disjoint={strong_groups_disjoint}；unit_col={unit_col}；"
                f"iso_units_total={len(iso_units)} iso_relax_frac={iso_relax_frac} iso_relax_to_calib_frac={iso_relax_to_calib_frac}")
    print(f"[Temporal] Wrote split to: {out_base}")


# =========================
# 主流程（仅接收 --config）
# =========================
def main():
    ap = argparse.ArgumentParser(description="S4 splits: LODO and temporal with independent calibration set. (config-driven; strong-group safe)")
    ap.add_argument("--config", required=True, help="Path to config.json (读取 s4_make_splits 部分)")
    args_cli = ap.parse_args()

    # 读取配置
    with open(args_cli.config, "r", encoding="utf-8") as f:
        conf = json.load(f)

    s4_conf = conf.get("s4_make_splits", {})
    if not s4_conf:
        raise ValueError("config.json 中缺少 's4_make_splits' 配置块。")

    log_conf = conf.get("logging", {})
    log_file = log_conf.get("log_file", "./logs/s4_make_splits.log")
    log_level = log_conf.get("level", "INFO")

    logger = setup_logger(log_file, log_level)

    # 启动信息
    logger.info("===== S4 Make Splits 任务启动 =====")
    logger.info(f"CUDA 可用: {_CUDA}")
    logger.info(f"配置文件: {os.path.abspath(args_cli.config)}")
    logger.info(f"日志文件: {os.path.abspath(log_file)}")

    # 组装 Args
    args = Args(
        input=s4_conf["input"],
        outdir=s4_conf["outdir"],
        protocol=s4_conf["protocol"],
        part_col=s4_conf.get("part_col", "part_id"),
        domain_col=s4_conf.get("domain_col", "domain"),
        timestamp_col=s4_conf.get("timestamp_col", "timestamp"),
        family_col=s4_conf.get("family_col", "family_id"),
        calib_frac=float(s4_conf.get("calib_frac", 0.1)),
        train_frac=float(s4_conf.get("train_frac", 0.8)),
        seed=int(s4_conf.get("seed", 42)),
        min_domain_size=int(s4_conf.get("min_domain_size", 5)),
        per_domain_temporal=bool(s4_conf.get("per_domain_temporal", False)),
        stratify_col=s4_conf.get("stratify_col", ""),
        part_index_dir=s4_conf.get("part_index_dir", ""),
        num_workers=int(s4_conf.get("num_workers", 4)),
        duplicate_col=s4_conf.get("duplicate_col", "duplicate_canonical"),
        content_hash_col=s4_conf.get("content_hash_col", "content_hash"),
    )

    # 入参回显
    logger.info(f"入参：protocol={args.protocol} input={args.input} outdir={args.outdir}")
    logger.info(f"列名：part_col={args.part_col} domain_col={args.domain_col} timestamp_col={args.timestamp_col} family_col={args.family_col}")
    logger.info(f"强连边列：duplicate_col={args.duplicate_col} content_hash_col={args.content_hash_col}")
    logger.info(f"比例：train_frac={args.train_frac} calib_frac={args.calib_frac} min_domain_size={args.min_domain_size} per_domain_temporal={args.per_domain_temporal}")
    logger.info(f"其它：seed={args.seed} stratify_col={args.stratify_col} part_index_dir={args.part_index_dir} num_workers={args.num_workers}")

    # 读入 CSV
    if not os.path.exists(args.input):
        logger.error(f"输入 CSV 不存在: {args.input}")
        raise FileNotFoundError(args.input)
    df = pd.read_csv(args.input)
    logger.info(f"读取输入 CSV 完成：rows={df.shape[0]} cols={list(df.columns)}")

    # 合并 family_id（如提供 part_index）
    pi_df = load_part_index(args.part_index_dir)
    if pi_df is not None:
        df, merged_family = merge_family_id(df, pi_df, args.part_col, args.family_col)
        # 也尝试补齐 duplicate/content_hash
        df, merged_flags = merge_cols_from_part_index(
            df, pi_df, args.part_col,
            [args.duplicate_col, args.content_hash_col]
        )
        logger.info(f"尝试从 part_index 合并列：family_id merged={merged_family}；"
                    f"{args.duplicate_col} merged={merged_flags.get(args.duplicate_col, False)}；"
                    f"{args.content_hash_col} merged={merged_flags.get(args.content_hash_col, False)}")
    else:
        logger.info("未提供 part_index 或文件不存在，跳过 family_id/duplicate/content_hash 合并。")

    # 必要列检查（按协议）
    if args.protocol == "lodo":
        check_required_cols(df, [args.part_col, args.domain_col])
    else:
        check_required_cols(df, [args.part_col, args.timestamp_col])
    logger.info("必需列检查通过。")

    # 按 part_id 去重
    before = len(df)
    df = df.drop_duplicates(subset=[args.part_col]).reset_index(drop=True)
    after = len(df)
    logger.info(f"按照 {args.part_col} 去重：{before} → {after}")

    os.makedirs(args.outdir, exist_ok=True)

    # ===== 新增：构建强组并记录在 df['_strong_group'] =====
    strong_series = build_strong_groups(df, args.part_col, args.duplicate_col, args.content_hash_col)
    # 将 Series（index=part_id）合回 df
    df[args.part_col] = df[args.part_col].astype(str)
    strong_series = strong_series.rename(index=str)
    df = df.merge(strong_series.rename("_strong_group").reset_index().rename(columns={"index": args.part_col}),
                  on=args.part_col, how="left")
    # 对于完全缺失的情况（防御）
    if df["_strong_group"].isna().any():
        df["_strong_group"] = df["_strong_group"].fillna(df[args.part_col].astype(str))
    logger.info(f"强组构建完成：n_groups={df['_strong_group'].nunique()}")

    # 执行
    if args.protocol == "lodo":
        lodo_split(df, args, logger)
    else:
        if not (0 < args.train_frac < 1) or not (0 < args.calib_frac < 1) or args.train_frac + args.calib_frac < 0.05 or args.train_frac + args.calib_frac >= 1.0:
            raise ValueError("Invalid train/calib fractions. Ensure 0<train<1, 0<calib<1, and train+calib in (0.05, 1).")
        temporal_split(df, args, logger)

    logger.info("===== S4 Make Splits 任务完成 =====")


if __name__ == "__main__":
    main()
