#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, date
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from collections import Counter

# ========== 原有逻辑函数（未改变业务逻辑与输出） ==========

def try_import_cudf(prefer_cudf: bool):
    if not prefer_cudf:
        return None
    try:
        import cudf  # type: ignore
        return cudf
    except Exception:
        return None

def import_pandas():
    import pandas as pd
    return pd

def sanitize_token(x):
    if x is None:
        return "NA"
    x = str(x)
    if x.strip() == "":
        return "NA"
    x = re.sub(r"\s+", "_", x.strip().lower())
    x = re.sub(r"[^a-z0-9_\-\.]", "-", x)
    return x

def parse_date_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    patterns = [
        r"(?P<y>20\d{2})[-_/\.](?P<m>0?[1-9]|1[0-2])[-_/\.](?P<d>0?[1-9]|[12]\d|3[01])",
        r"(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])",
        r"(?P<d>0?[1-9]|[12]\d|3[01])[-_/\.](?P<m>0?[1-9]|1[0-2])[-_/\.](?P<y>20\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                y = int(m.group("y")); mth = int(m.group("m")); d = int(m.group("d"))
                dt = datetime(y, mth, d)
                return dt.date().isoformat()
            except Exception:
                continue
    return None

def _coalesce_timestamp_with_source(row, ts_cols: List[str], path_col: Optional[str],
                                    dataset_col: Optional[str], dataset_release_map: dict) -> Tuple[Optional[str], str]:
    for col in ts_cols:
        if col in row and row[col] is not None:
            val = row[col]
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S",
                        "%Y/%m/%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    ts = datetime.strptime(str(val).split(".")[0], fmt).date().isoformat()
                    return ts, "explicit"
                except Exception:
                    pass
            try:
                import pandas as pd
                dt = pd.to_datetime(val, errors="coerce", utc=False, dayfirst=False)
                if pd.notnull(dt):
                    return dt.date().isoformat(), "explicit"
            except Exception:
                pass
    if path_col and path_col in row and row[path_col]:
        ts = parse_date_from_text(str(row[path_col]))
        if ts:
            return ts, "path"
    if dataset_col and dataset_col in row and row[dataset_col] in dataset_release_map:
        return dataset_release_map[row[dataset_col]], "dataset_map"
    return None, "none"

def coalesce_timestamp(row, ts_cols: List[str], path_col: Optional[str], dataset_col: Optional[str], dataset_release_map: dict):
    ts, _ = _coalesce_timestamp_with_source(row, ts_cols, path_col, dataset_col, dataset_release_map)
    return ts

def build_domain(row, domain_keys: List[str]):
    items = []
    for k in domain_keys:
        if k in row:
            items.append(f"{k}={sanitize_token(row[k])}")
    if not items:
        return "domain=unknown"
    return "domain:" + "|".join(items)

# ========== 并行（保持输出顺序；仅加速不改结果） ==========

_WORK_CFG = {"domain_keys": None, "ts_cols": None, "path_col": None, "dataset_col": None, "ds_release_map": None}

def _worker_init(domain_keys, ts_cols, path_col, dataset_col, ds_release_map):
    _WORK_CFG["domain_keys"] = domain_keys
    _WORK_CFG["ts_cols"] = ts_cols
    _WORK_CFG["path_col"] = path_col
    _WORK_CFG["dataset_col"] = dataset_col
    _WORK_CFG["ds_release_map"] = ds_release_map

def _process_row(idx_and_row: Tuple[int, Dict[str, Any]]) -> Tuple[int, str, Optional[str], str]:
    idx, row = idx_and_row
    domain = build_domain(row, _WORK_CFG["domain_keys"])
    ts, src = _coalesce_timestamp_with_source(row, _WORK_CFG["ts_cols"], _WORK_CFG["path_col"],
                                              _WORK_CFG["dataset_col"], _WORK_CFG["ds_release_map"])
    return idx, domain, ts, src

# ========== 进度打印（控制台） ==========

def print_progress(completed: int, total: int, last_percent: int, min_percent: int = 1) -> int:
    if total <= 0:
        return last_percent
    if total < 100:
        pct = int((completed * 100.0) / total + 0.0001)
        if pct > last_percent:
            print(f"\rProgress: {pct:3d}% ({completed}/{total})", end="", flush=True)
            return pct
        return last_percent
    pct = int((completed * 100.0) / total + 0.0001)
    if pct - last_percent >= max(1, min_percent):
        print(f"\rProgress: {pct:3d}% ({completed}/{total})", end="", flush=True)
        return pct
    return last_percent

# ========== 配置、日志与小工具 ==========

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = logging.getLogger("s3")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logging.captureWarnings(True)
    return logger

def _file_stat(p: str) -> str:
    try:
        st = os.stat(p)
        mtime = datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ", timespec="seconds")
        return f"size={st.st_size}B mtime={mtime}"
    except Exception as e:
        return f"stat-failed: {e}"

def _timestamp_stats(ts_list: List[Optional[str]]) -> Dict[str, Any]:
    vals: List[date] = []
    for t in ts_list:
        if isinstance(t, str) and t:
            try:
                vals.append(datetime.fromisoformat(t).date())
            except Exception:
                pass
    total = len(ts_list)
    valid = len(vals)
    none_cnt = sum(1 for t in ts_list if not t)
    stats = {
        "total": total,
        "valid": valid,
        "valid_pct": round(100.0 * valid / total, 2) if total else 0.0,
        "none": none_cnt,
        "none_pct": round(100.0 * none_cnt / total, 2) if total else 0.0,
        "min": min(vals).isoformat() if vals else None,
        "max": max(vals).isoformat() if vals else None,
    }
    return stats

# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser(description="S3: Domain & Timestamp - config-driven (reads s3_domain_and_timestamp section)")
    parser.add_argument("--config", default="config.json", help="Path to top-level config.json")
    args = parser.parse_args()

    # STEP 0: 配置
    logger = None
    cfg_all = load_config(args.config)
    cfg = cfg_all.get("s3_domain_and_timestamp")
    if not isinstance(cfg, dict):
        raise ValueError("config.json 缺少 s3_domain_and_timestamp 段，或其不是对象。")

    part_index = cfg.get("part_index")
    out_path = cfg.get("out")
    if not part_index or not out_path:
        raise ValueError("s3_domain_and_timestamp.part_index 与 s3_domain_and_timestamp.out 均为必填。")

    prefer_cudf = bool(cfg.get("prefer_cudf", False))
    domain_keys = cfg.get("domain_keys", ["source_domain","kernel","repo","author","source_dataset"])
    timestamp_cols = cfg.get("timestamp_cols", ["timestamp","created_at","modified_at","time","date","created_at_header"])
    path_col = cfg.get("path_col", "path")
    dataset_col = cfg.get("dataset_col", "source_dataset")
    unit_col = cfg.get("unit_col", "unit")
    ds_release_map_path = cfg.get("dataset_release_map")
    log_file = cfg.get("log_file") or cfg_all.get("logging", {}).get("log_file") \
               or os.path.join(os.path.dirname(out_path) or ".", "run.log")
    parallel_workers = int(cfg.get("parallel_workers", 0))
    progress_min_percent = int(cfg.get("progress_min_percent", 1))

    logger = setup_logger(log_file)
    logger.info("=== [STEP_START] 0.CONFIG ===")
    logger.info(f"[关键日志] config_path={os.path.abspath(args.config)} section=s3_domain_and_timestamp")
    logger.info(f"[关键日志] params: part_index='{part_index}', out='{out_path}', prefer_cudf={prefer_cudf}, "
                f"parallel_workers={parallel_workers}, progress_min_percent={progress_min_percent}")
    logger.info(f"[关键日志] cols: domain_keys={domain_keys}, timestamp_cols={timestamp_cols}, "
                f"path_col='{path_col}', dataset_col='{dataset_col}', unit_col='{unit_col}'")
    logger.info("[STEP_END] 0.CONFIG")

    # STEP 1: 输入与映射
    logger.info("=== [STEP_START] 1.INPUT_CHECK ===")
    if not os.path.exists(part_index):
        logger.info(f"[关键日志] 输入文件缺失: {part_index}")
        raise FileNotFoundError(f"part_index not found: {part_index}")
    logger.info(f"[关键日志] 输入文件OK: {part_index} ({_file_stat(part_index)})")
    if ds_release_map_path:
        if os.path.exists(ds_release_map_path):
            logger.info(f"[关键日志] dataset_release_map OK: {ds_release_map_path} ({_file_stat(ds_release_map_path)})")
        else:
            logger.info(f"[关键日志] dataset_release_map 缺失(将忽略): {ds_release_map_path}")
    else:
        logger.info("[关键日志] 未配置 dataset_release_map（可选）")
    logger.info("[STEP_END] 1.INPUT_CHECK")

    # STEP 2: 读取 CSV
    logger.info("=== [STEP_START] 2.READ_CSV ===")
    cudf = try_import_cudf(prefer_cudf)
    pd = None if cudf is not None else import_pandas()
    backend = "cuDF(GPU)" if cudf is not None else "pandas(CPU)"
    logger.info(f"[关键日志] backend={backend}")
    t0 = time.perf_counter()
    if cudf is not None:
        df = cudf.read_csv(part_index)
        df_pd = df.to_pandas()
    else:
        df_pd = pd.read_csv(part_index)
    t1 = time.perf_counter()
    if "part_id" not in df_pd.columns:
        logger.info("[关键日志] 读取失败: 缺少必需列 part_id")
        raise ValueError("Input must contain 'part_id' column.")
    n_rows = len(df_pd)
    logger.info(f"[关键日志] 读取完成: rows={n_rows}, cols={len(df_pd.columns)}, elapsed={t1 - t0:.3f}s")
    logger.info("[STEP_END] 2.READ_CSV")
    print(f"Loaded rows: {n_rows}")

    # STEP 3: 列准备
    logger.info("=== [STEP_START] 3.PREP_COLUMNS ===")
    dk_hit = [c for c in domain_keys if c in df_pd.columns]
    dk_miss = [c for c in domain_keys if c not in df_pd.columns]
    ts_hit = [c for c in timestamp_cols if c in df_pd.columns]
    ts_miss = [c for c in timestamp_cols if c not in df_pd.columns]
    logger.info(f"[关键日志] domain_keys 命中 {len(dk_hit)}/{len(domain_keys)} 缺失={dk_miss if dk_miss else '无'}")
    logger.info(f"[关键日志] timestamp_cols 命中 {len(ts_hit)}/{len(timestamp_cols)} 缺失={ts_miss if ts_miss else '无'}")

    ds_release_map = {}
    if ds_release_map_path and os.path.exists(ds_release_map_path):
        try:
            with open(ds_release_map_path, "r", encoding="utf-8") as f:
                ds_release_map = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load dataset-release-map: {e}")
            logger.info(f"[关键日志] dataset_release_map 加载失败: {e}")

    needed_cols = set(domain_keys) | set(timestamp_cols) | {path_col, dataset_col, "part_id"}
    if unit_col in df_pd.columns:
        needed_cols.add(unit_col)
    safe_cols = [c for c in needed_cols if c in df_pd.columns]
    drop_cols = sorted(set(df_pd.columns) - set(safe_cols))
    logger.info(f"[关键日志] 参与处理列数={len(safe_cols)}，忽略列数={len(drop_cols)}")
    records: List[Dict[str, Any]] = df_pd[safe_cols].to_dict(orient="records")
    indexed_records = list(enumerate(records))
    logger.info("[STEP_END] 3.PREP_COLUMNS")

    # STEP 4: 处理（含进度与质量统计）
    logger.info("=== [STEP_START] 4.PROCESS ===")
    if parallel_workers == 1:
        use_workers = 1
    elif parallel_workers > 1:
        use_workers = parallel_workers
    else:
        cpu_cnt = max(1, multiprocessing.cpu_count())
        use_workers = cpu_cnt if n_rows >= 2 * cpu_cnt else 1
    logger.info(f"[关键日志] use_workers={use_workers}")

    domains = [None] * n_rows
    timestamps = [None] * n_rows
    ts_src_counter = Counter()

    completed = 0
    last_pct = -1
    next_keylog_mark = 25

    t2 = time.perf_counter()
    if use_workers == 1:
        for idx, row in indexed_records:
            domain = build_domain(row, domain_keys)
            ts, src = _coalesce_timestamp_with_source(row, timestamp_cols, path_col, dataset_col, ds_release_map)
            domains[idx] = domain
            timestamps[idx] = ts
            ts_src_counter[src] += 1

            completed += 1
            last_pct = print_progress(completed, n_rows, last_percent=last_pct, min_percent=progress_min_percent)
            pct_now = int((completed * 100.0) / n_rows + 0.0001) if n_rows else 100
            if pct_now >= next_keylog_mark:
                logger.info(f"[关键日志] 处理进度 {next_keylog_mark}%（{completed}/{n_rows}）")
                if next_keylog_mark == 25:
                    logger.info(f"[关键日志] 样例 domain: {domains[idx]}")
                next_keylog_mark = {25:50, 50:75, 75:100}.get(next_keylog_mark, 101)
        print()
    else:
        with ProcessPoolExecutor(
            max_workers=use_workers,
            initializer=_worker_init,
            initargs=(domain_keys, timestamp_cols, path_col, dataset_col, ds_release_map),
        ) as ex:
            futures = {ex.submit(_process_row, item): item[0] for item in indexed_records}
            for fut in as_completed(futures):
                idx, domain, ts, src = fut.result()
                domains[idx] = domain
                timestamps[idx] = ts
                ts_src_counter[src] += 1

                completed += 1
                last_pct = print_progress(completed, n_rows, last_percent=last_pct, min_percent=progress_min_percent)
                pct_now = int((completed * 100.0) / n_rows + 0.0001) if n_rows else 100
                if pct_now >= next_keylog_mark:
                    logger.info(f"[关键日志] 处理进度 {next_keylog_mark}%（{completed}/{n_rows}）")
                    if next_keylog_mark == 25:
                        logger.info(f"[关键日志] 样例 domain: {domains[idx]}")
                    next_keylog_mark = {25:50, 50:75, 75:100}.get(next_keylog_mark, 101)
        print()
    t3 = time.perf_counter()

    # 本步输出的合理性校验
    ts_stats = _timestamp_stats(timestamps)
    dom_unknown_cnt = sum(1 for d in domains if d == "domain=unknown")
    dom_unknown_pct = round(100.0 * dom_unknown_cnt / n_rows, 2) if n_rows else 0.0
    logger.info(f"[关键日志] 时间戳来源分布: {dict(ts_src_counter)}")
    logger.info(f"[关键日志] 时间戳统计: valid={ts_stats['valid']}/{ts_stats['total']} "
                f"({ts_stats['valid_pct']}%) none={ts_stats['none']}({ts_stats['none_pct']}%) "
                f"range=[{ts_stats['min']} ~ {ts_stats['max']}]")
    logger.info(f"[关键日志] domain=unknown 比例: {dom_unknown_cnt}/{n_rows} ({dom_unknown_pct}%)")
    logger.info(f"[关键日志] PROCESS 耗时: {t3 - t2:.3f}s")
    logger.info("[STEP_END] 4.PROCESS")

    # STEP 5: 装配输出与写入
    logger.info("=== [STEP_START] 5.WRITE_OUTPUT ===")
    unit_has_col = unit_col in df_pd.columns
    dataset_has_col = dataset_col in df_pd.columns
    out_cols = {
        "part_id": df_pd["part_id"],
        "domain": domains,
        "timestamp": timestamps,
        "unit": df_pd[unit_col] if unit_has_col else ["NA"] * len(df_pd),
        "source_dataset": df_pd[dataset_col] if dataset_has_col else ["NA"] * len(df_pd),
    }
    out_df = import_pandas().DataFrame(out_cols)
    unique_domains = out_df["domain"].nunique()
    logger.info(f"[关键日志] 输出DataFrame形状: rows={len(out_df)}, cols={len(out_df.columns)}, unique_domains={unique_domains}")
    try:
        pid_head = str(out_df['part_id'].iloc[0]); pid_tail = str(out_df['part_id'].iloc[-1])
        logger.info(f"[关键日志] part_id 样例: head={pid_head}, tail={pid_tail}")
        logger.info(f"[关键日志] 输出样例: domain='{out_df['domain'].iloc[0]}', timestamp='{out_df['timestamp'].iloc[0]}'")
    except Exception:
        pass

    print(f"[S3] rows: {len(out_df)}  unique domains: {unique_domains}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    logger.info(f"[关键日志] 写出开始: {out_path}")
    out_df.to_csv(out_path, index=False)
    logger.info(f"[关键日志] 写出完成: {out_path} ({_file_stat(out_path)})")
    print(f"[S3] wrote: {out_path}")

    ok = (len(out_df) == n_rows)
    if ok:
        logger.info(f"[关键日志] 行数对齐校验: OK ({len(out_df)}/{n_rows})")
        logger.info("=== [STEP_END] 5.WRITE_OUTPUT ===")
        logger.info(f"=== [TASK_OK] rows={len(out_df)} unique_domains={unique_domains} ===")
    else:
        logger.info(f"[关键日志] 行数对齐校验: FAIL ({len(out_df)}/{n_rows})")
        logger.info("=== [STEP_END] 5.WRITE_OUTPUT ===")
        logger.info("=== [TASK_FAIL] ===")

if __name__ == "__main__":
    main()

'''
python s3_domain_and_timestamp.py --config "/workspace/Gjj Doc/Code/CAD_TabPFN/config.json"
'''
