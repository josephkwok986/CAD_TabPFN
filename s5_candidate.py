#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5 主程序（并发版，GPU 强制 + 子进程静默 + 单一主进度条）
- 目录：/s5_out/{protocol}/split_{k}/{train,calib,test}/*.jsonl
- 不写空文件；区分三类：written / no_cand / failed
- split 级 summary.json：含 subset_no_cand
- 全局 summary_all.json：新增 no_candidate_by_folder 与 no_candidate_total
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time, signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import orjson
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict

from s5_geometry import make_backend, HAS_CUPY, HoleCandidate  # typing only

# -------------------- 工具函数 --------------------

def _read_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_lines_if_exists(p: Path) -> List[str]:
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _iter_protocol_splits(split_root: Path):
    if not split_root.exists():
        return
    for proto_dir in sorted([d for d in split_root.iterdir() if d.is_dir()]):
        protocol = proto_dir.name
        for split_dir in sorted([d for d in proto_dir.iterdir() if d.is_dir() and d.name.startswith("split_")]):
            try:
                split_id = int(split_dir.name.split("_")[-1])
            except Exception:
                continue
            yield protocol, split_id, split_dir

# -------------------- 并发 worker --------------------

_WORKER_CTX: Dict[str, object] = {}

def _silence_worker_io_and_logging():
    try:
        root = logging.getLogger()
        root.handlers = []
        root.setLevel(logging.CRITICAL)
        for name in list(logging.Logger.manager.loggerDict.keys()):
            logging.getLogger(name).disabled = True
        devnull = open(os.devnull, "w")
        sys.stdout = devnull  # type: ignore
        sys.stderr = devnull  # type: ignore
    except Exception:
        pass

def _worker_init(s5_cfg: Dict, split_out_dir: str, sep: str, silence: bool):
    if silence:
        _silence_worker_io_and_logging()
    use_gpu = True and HAS_CUPY  # 强制 GPU；无 CuPy 时几何端回退
    backend = make_backend(s5_cfg, use_gpu=use_gpu)
    _WORKER_CTX["backend"] = backend
    _WORKER_CTX["rel_root_map"] = s5_cfg["input"]["rel_root_map"]
    _WORKER_CTX["split_out_dir"] = split_out_dir  # 实际写入 split_out/<subset>/
    _WORKER_CTX["sep"] = sep
    _WORKER_CTX["per_file_timeout_sec"] = int(s5_cfg.get("resources", {}).get("per_file_timeout_sec", 180))

def _worker_process_one(part_id: str, rel_path: str, source_dataset: str, subset: str
    ) -> Tuple[str, str, str, Optional[str], int, float, Optional[str], Optional[Dict]]:
    """
    返回：(part_id, subset, status, error_msg, num_candidates, elapsed_ms, err_kind, diag)
    status ∈ {"written","no_cand","failed"}
    - written/no_cand/failed 均计为“已处理”；failed 记入 failed_parts
    - no_cand 时 diag 为 quick_probe 的面/边类型计数
    """
    t0 = time.perf_counter()
    try:
        backend = _WORKER_CTX["backend"]
        rel_root_map = _WORKER_CTX["rel_root_map"]
        split_out_dir = _WORKER_CTX["split_out_dir"]
        sep = _WORKER_CTX["sep"]
        timeout_sec = _WORKER_CTX.get("per_file_timeout_sec", 180)

        # 目标路径
        safe_name = part_id.replace("/", sep)
        out_dir = Path(split_out_dir) / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_name}.jsonl"

        # 幂等
        if out_path.exists():
            return part_id, subset, "written", None, -1, (time.perf_counter() - t0) * 1000.0, None, None

        # 软超时
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"per-file timeout after {timeout_sec}s")
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_sec)
        except Exception:
            pass

        abs_root = rel_root_map[source_dataset]
        abs_path = Path(abs_root) / rel_path
        shape = backend.load_shape(str(abs_path))
        cands: List[HoleCandidate] = backend.extract_candidates(shape, part_id)

        if cands:
            with open(out_path, "w", encoding="utf-8") as fw:
                for c in cands:
                    fw.write(orjson.dumps(asdict(c)).decode("utf-8"))
                    fw.write("\n")
            return part_id, subset, "written", None, len(cands), (time.perf_counter() - t0) * 1000.0, None, None

        # 无候选 → 做快速诊断
        diag = backend.quick_probe(shape)
        return part_id, subset, "no_cand", None, 0, (time.perf_counter() - t0) * 1000.0, None, diag

    except Exception as e:
        msg = str(e)
        ml = msg.lower()
        if "read fail" in ml:
            kind = "read_fail"
        elif "timeout" in ml:
            kind = "timeout"
        elif "no such file" in ml or "not found" in ml or "errno 2" in ml:
            kind = "missing_file"
        else:
            kind = "other_error"
        return part_id, subset, "failed", msg, 0, (time.perf_counter() - t0) * 1000.0, kind, None
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass

# -------------------- 主流程 --------------------

def _make_executor(num_workers: int, s5_cfg: Dict, split_out_dir: str, sep: str, silence_workers: bool) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(s5_cfg, str(split_out_dir), sep, silence_workers),
    )

def _process_one_split_parallel(
    s5_cfg: Dict,
    idx: pd.DataFrame,
    split_root: Path,
    protocol: str,
    split_id: int,
) -> Dict:
    split_dir = split_root / protocol / f"split_{split_id}"

    subsets = ("train", "calib", "test")
    parts_by_subset: Dict[str, List[str]] = {}
    for which in subsets:
        pids = _read_lines_if_exists(split_dir / f"{which}.txt")
        parts_by_subset[which] = [p for p in pids if p in idx.index]

    split_out_dir = Path(s5_cfg["output"]["dir"]) / protocol / f"split_{split_id}"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    sep = s5_cfg["output"].get("flatten_sep", "__")

    total_parts = sum(len(v) for v in parts_by_subset.values())
    if total_parts == 0:
        summary = {
            "protocol": protocol, "split_id": split_id,
            "total_parts": 0, "written_parts": 0, "failed_parts": 0,
            "use_gpu": True and HAS_CUPY, "errors_head": [],
            "median_ms": 0.0, "p90_ms": 0.0, "error_breakdown": {},
            "subset_counts": {k: 0 for k in subsets},
            "subset_written": {k: 0 for k in subsets},
            "subset_failed": {k: 0 for k in subsets},
            "subset_no_cand": {k: 0 for k in subsets},
        }
        with open(split_out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    items = []
    has_size = "file_size_bytes" in idx.columns
    for which in subsets:
        for pid in parts_by_subset[which]:
            row = idx.loc[pid]
            size = int(row["file_size_bytes"]) if has_size else 0
            items.append((pid, row["rel_path"], row["source_dataset"], size, which))
    items.sort(key=lambda t: t[3], reverse=True)
    q = deque(items)

    num_workers = int(s5_cfg["resources"].get("num_workers", max(1, (os.cpu_count() or 2) // 2)))
    inflight_mult = int(s5_cfg["resources"].get("max_inflight_multiplier", 2))
    max_inflight = max(num_workers, num_workers * inflight_mult)
    silence_workers = bool(s5_cfg.get("logging", {}).get("silence_workers", True))
    soft_timeout = int(s5_cfg["resources"].get("per_file_timeout_sec", 180))
    hard_timeout = int(s5_cfg["resources"].get("hard_timeout_sec", max(60, soft_timeout * 4)))
    poll_sec = float(s5_cfg["resources"].get("watchdog_poll_sec", 5.0))
    NO_CAND_SAMPLE_CAP = int(s5_cfg.get("logging", {}).get("no_cand_sample_cap", 200))

    logging.info(f"[{protocol}/split_{split_id}] parts={total_parts} | workers={num_workers} | inflight≤{max_inflight} | "
                 f"force_gpu={bool(HAS_CUPY)} | silence_workers={silence_workers}")

    written = 0
    errors: List[Tuple[str, str]] = []
    times: List[float] = []
    err_kinds: Dict[str, int] = {}
    subset_written = {k: 0 for k in subsets}
    subset_failed  = {k: 0 for k in subsets}
    subset_counts  = {k: len(parts_by_subset[k]) for k in subsets}
    subset_no_cand = {k: 0 for k in subsets}
    no_cand_samples: List[Dict] = []

    futures = set()
    fmeta: Dict[object, Tuple[str, str, str, str, float]] = {}

    with _make_executor(num_workers, s5_cfg, str(split_out_dir), sep, silence_workers) as ex, \
         tqdm(total=len(items), unit="part", desc=f"S5 {protocol}/split_{split_id}", leave=True) as pbar:

        while q and len(futures) < max_inflight:
            pid, rel_path, src, _sz, which = q.popleft()
            fut = ex.submit(_worker_process_one, pid, rel_path, src, which)
            futures.add(fut); fmeta[fut] = (pid, rel_path, src, which, time.time())

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=poll_sec)

            if done:
                for fut in list(done):
                    try:
                        res = fut.result()
                        # 兼容 7/8 元组：7 -> (pid, which, status, err, _n, el_ms, err_kind)
                        #               8 -> 另加 diag
                        if isinstance(res, tuple) and len(res) == 8:
                            pid, which, status, err, _n, el_ms, err_kind, diag = res
                        elif isinstance(res, tuple) and len(res) == 7:
                            pid, which, status, err, _n, el_ms, err_kind = res
                            diag = None
                        else:
                            # 意外长度，按失败处理
                            meta = fmeta.get(fut, ("<unknown>", "", "", "unknown", 0.0))
                            pid, which, status, err, el_ms, err_kind, diag = meta[0], meta[3], "failed", f"unexpected result len={len(res)}", 0.0, "other_error", None
                    except Exception as e:
                        meta = fmeta.get(fut, ("<unknown>", "", "", "unknown", 0.0))
                        pid, which, status, err, el_ms, err_kind, diag = meta[0], meta[3], "failed", repr(e), 0.0, "other_error", None

                    times.append(float(el_ms))
                    if status == "written":
                        written += 1
                        subset_written[which] += 1
                    elif status == "no_cand":
                        subset_no_cand[which] += 1
                        if diag is not None and len(no_cand_samples) < NO_CAND_SAMPLE_CAP:
                            no_cand_samples.append({
                                "part_id": pid, "subset": which,
                                "diag": diag, "elapsed_ms": float(el_ms)
                            })
                    else:
                        subset_failed[which] += 1
                        if err:
                            errors.append((pid, err))
                        if err_kind:
                            err_kinds[err_kind] = err_kinds.get(err_kind, 0) + 1

                    pbar.update(1)
                    futures.remove(fut)
                    fmeta.pop(fut, None)

                    if q:
                        npid, nrel, nsrc, _sz, nwhich = q.popleft()
                        nfut = ex.submit(_worker_process_one, npid, nrel, nsrc, nwhich)
                        futures.add(nfut); fmeta[nfut] = (npid, nrel, nsrc, nwhich, time.time())
                continue

            # 看门狗
            now = time.time()
            stale = []
            for fut in list(futures):
                pid, rel_path, src, which, st = fmeta[fut]
                if now - st > hard_timeout:
                    stale.append(fut)

            if stale:
                for fut in stale:
                    pid, rel_path, src, which, st = fmeta[fut]
                    errors.append((pid, f"hard_timeout>{hard_timeout}s"))
                    err_kinds["hard_timeout"] = err_kinds.get("hard_timeout", 0) + 1
                    times.append((now - st) * 1000.0)
                    subset_failed[which] += 1
                    pbar.update(1)
                    futures.remove(fut)
                    fmeta.pop(fut, None)

                pending_tasks = []
                for fut in list(futures):
                    pid, rel_path, src, which, st = fmeta[fut]
                    pending_tasks.append((pid, rel_path, src, which))
                futures.clear(); fmeta.clear()

                try:
                    if hasattr(ex, "_processes"):
                        for p in list(ex._processes.values()):
                            try: os.kill(p.pid, signal.SIGKILL)
                            except Exception: pass
                except Exception:
                    pass
                ex.shutdown(wait=False, cancel_futures=True)

                ex = _make_executor(num_workers, s5_cfg, str(split_out_dir), sep, silence_workers)
                for pid, rel_path, src, which in pending_tasks:
                    if len(futures) >= max_inflight:
                        q.appendleft((pid, rel_path, src, 0, which))
                    else:
                        nfut = ex.submit(_worker_process_one, pid, rel_path, src, which)
                        futures.add(nfut); fmeta[nfut] = (pid, rel_path, src, which, time.time())
                while q and len(futures) < max_inflight:
                    pid, rel_path, src, _sz, which = q.popleft()
                    nfut = ex.submit(_worker_process_one, pid, rel_path, src, which)
                    futures.add(nfut); fmeta[nfut] = (pid, rel_path, src, which, time.time())

    summary = {
        "protocol": protocol,
        "split_id": split_id,
        "total_parts": total_parts,
        "written_parts": written,
        "failed_parts": int(sum(subset_failed.values())),
        "use_gpu": True and HAS_CUPY,
        "errors_head": errors[:min(20, len(errors))],
        "median_ms": float(np.median(times)) if times else 0.0,
        "p90_ms": float(np.percentile(times, 90)) if times else 0.0,
        "error_breakdown": err_kinds,
        "subset_counts": subset_counts,
        "subset_written": subset_written,
        "subset_failed": subset_failed,
        "subset_no_cand": subset_no_cand,
    }
    with open(split_out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if errors:
        with open(split_out_dir / "errors_full.jsonl", "w", encoding="utf-8") as ef:
            for pid, err in errors:
                ef.write(orjson.dumps({"part_id": pid, "error": err}).decode("utf-8") + "\n")

    if no_cand_samples:
        out_probe = split_out_dir / "no_cand_samples.jsonl"
        with open(out_probe, "w", encoding="utf-8") as fp:
            for rec in no_cand_samples:
                fp.write(orjson.dumps(rec).decode("utf-8") + "\n")

    return summary



def run_s5(config_path: str):
    cfg = _read_config(config_path)
    s5 = cfg.get("S5") or cfg.get("s5")
    if s5 is None:
        raise KeyError("config missing 'S5'/'s5' block")

    # 资源与日志
    s5.setdefault("resources", {})
    s5["resources"].setdefault("device", "gpu")
    s5["resources"].setdefault("per_file_timeout_sec", 180)
    s5["resources"].setdefault("hard_timeout_sec",  max(60, s5["resources"]["per_file_timeout_sec"] * 4))
    s5["resources"].setdefault("watchdog_poll_sec", 5.0)
    s5["resources"].setdefault("max_inflight_multiplier", 2)

    out_base = Path(s5["output"]["dir"])
    out_base.mkdir(parents=True, exist_ok=True)

    log_cfg = s5.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # 索引：尽量带上 file_size_bytes
    idx = pd.read_csv(s5["input"]["part_index_csv"])
    cols = {
        s5["input"]["part_id_column"]: "part_id",
        s5["input"]["path_column"]: "rel_path",
        s5["input"]["source_dataset_column"]: "source_dataset",
    }
    keep_cols = list(cols.keys())
    if "file_size_bytes" in idx.columns:
        keep_cols.append("file_size_bytes")
    idx = idx[keep_cols].copy()
    idx.rename(columns=cols, inplace=True)
    idx.set_index("part_id", inplace=True)

    split_root = Path(s5["input"]["split_root"])

    summaries: List[Dict] = []
    for protocol, split_id, _ in _iter_protocol_splits(split_root):
        summaries.append(
            _process_one_split_parallel(
                s5_cfg=s5, idx=idx, split_root=split_root,
                protocol=protocol, split_id=split_id
            )
        )

    # 汇总到 summary_all.json，并附上每个文件夹的 no_cand 计数
    no_cand_by_folder: Dict[str, int] = {}
    total_no_cand = 0
    for s in summaries:
        proto = s["protocol"]; sid = s["split_id"]
        for subset, cnt in s.get("subset_no_cand", {}).items():
            folder = f"{proto}/split_{sid}/{subset}"
            c = int(cnt)
            no_cand_by_folder[folder] = c
            total_no_cand += c

    agg = {
        "total_protocol_splits": len(summaries),
        "total_parts": int(sum(s["total_parts"] for s in summaries)),
        "written_parts": int(sum(s["written_parts"] for s in summaries)),
        "failed_parts": int(sum(s["failed_parts"] for s in summaries)),   # 真失败合计
        "no_candidate_total": int(total_no_cand),                         # 已处理但无候选合计
        "no_candidate_by_folder": no_cand_by_folder,                      # 按文件夹细分
        "details": summaries                                              # 保留每 split 详情
    }
    with open(out_base / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    logging.info(f"All done. Global summary at {out_base/'summary_all.json'}")

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_s5(args.config)
