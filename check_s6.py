#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_s6.py
S6 输出审计（多进程、安全并行版）
- 仅主进程 tqdm，最小粒度 1%
- 过程日志走 stderr，随时 flush
- 软超时(SIGALRM) + 硬看门狗(池重启并回队列)
- 窗口化提交，inflight=workers*multiplier
- 重活优先：按文件 size 降序
- 幂等：已缓存或已统计过的跳过
"""

import os, re, json, csv, sys, argparse, hashlib, statistics, glob, time, signal
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import defaultdict, deque
import multiprocessing as mp
from tqdm import tqdm

# -------------------- 通用 --------------------
def print_err(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def md5_16(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def read_lines(fp: str) -> List[str]:
    if not os.path.isfile(fp):
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []

def pctl(xs, q):
    if not xs: return 0.0
    xs = sorted(xs)
    k = (len(xs)-1)*q
    i = int(k)
    if i == k: return xs[i]
    j = i+1
    return xs[i] + (xs[j]-xs[i])*(k-i)

def percent_bar(total: int, desc: str):
    """ 只在进度 >=1% 或完成时刷新 tqdm """
    if total <= 0:
        return (lambda n: None, lambda: None)
    step = max(1, total // 100)
    bar = tqdm(total=total, desc=desc, ncols=100)
    state = {"last": 0}
    def update(n_now: int):
        n_now = min(n_now, total)
        if n_now - state["last"] >= step or n_now == total:
            inc = n_now - state["last"]
            if inc > 0:
                bar.update(inc)
                state["last"] = n_now
    def close():
        if state["last"] < total:
            bar.update(total - state["last"])
        bar.close()
    return update, close

# -------------------- 配置/解析 --------------------
def load_config(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_part_index_cache(part_index_csv: str, part_id_col="part_id",
                          rel_path_col="rel_path", dataset_col="source_dataset") -> Dict[str, Tuple[str,str]]:
    cache = {}
    if not part_index_csv or not os.path.isfile(part_index_csv):
        return cache
    with open(part_index_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get(part_id_col)
            relp = row.get(rel_path_col)
            ds = row.get(dataset_col)
            if pid and relp and ds:
                cache[pid] = (ds, relp)
    return cache

def resolve_step_path(raw_line: str, cfg: dict, pi_cache: dict) -> Optional[str]:
    if not raw_line: return None
    s = raw_line.strip().strip('"').strip("'")
    if os.path.isabs(s) and os.path.isfile(s): return s
    if os.path.isfile(s): return os.path.abspath(s)
    s5in = (cfg.get("s5") or {}).get("input") or {}
    rel_roots = s5in.get("rel_root_map", {}) or {}
    norm = s.replace("\\", "/")
    segs = norm.split("/")
    if segs and segs[0] in rel_roots:
        cand = os.path.join(rel_roots[segs[0]], "/".join(segs[1:]))
        if os.path.isfile(cand): return cand
    if s in pi_cache:
        ds, relp = pi_cache[s]
        root = rel_roots.get(ds)
        if root:
            cand = os.path.join(root, relp)
            if os.path.isfile(cand): return cand
    for root in rel_roots.values():
        cand = os.path.join(root, norm)
        if os.path.isfile(cand): return cand
    return None

def parse_stress_report(md_path: str) -> dict:
    out = {}
    if not os.path.isfile(md_path):
        return out
    text = open(md_path, "r", encoding="utf-8").read()
    m = re.search(r"target_recall:\s*([0-9.]+)", text)
    if m: out["target_recall"] = float(m.group(1))
    m = re.search(r"baseline\(stress\):\s*recall=([0-9.]+).*?med_latency=([0-9.]+).*?cand/part=([0-9.]+).*?n=([0-9]+)", text)
    if m:
        out["stress_baseline"] = {
            "recall": float(m.group(1)),
            "lat": float(m.group(2)),
            "cpp": float(m.group(3)),
            "n": int(m.group(4))
        }
    m = re.search(r"selected\(stress\):\s*recall=([0-9.]+)\s*med_latency=([0-9.]+)\s*cand/part=([0-9.]+)", text)
    if m:
        out["stress_selected"] = {
            "recall": float(m.group(1)),
            "lat": float(m.group(2)),
            "cpp": float(m.group(3))
        }
    m = re.search(r"baseline\(real\):\s*med_latency=([0-9.]+)\s*cand/part=([0-9.]+)\s*n=([0-9]+)", text)
    if m:
        out["real_baseline"] = {
            "lat": float(m.group(1)),
            "cpp": float(m.group(2)),
            "n": int(m.group(3))
        }
    m = re.search(r"eval_tolerances:\s*(\{.*\})", text)
    if m:
        try: out["eval_tolerances"] = json.loads(m.group(1))
        except Exception: pass
    return out

# -------------------- 单文件计数（worker） --------------------
def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def _soft_alarm(sec: int):
    class _A:
        def __init__(self, sec): self.sec = int(sec) if sec else 0; self._old=None
        def __enter__(self):
            if self.sec>0:
                self._old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("soft_timeout")))
                signal.alarm(self.sec)
        def __exit__(self, *_):
            if self.sec>0:
                signal.alarm(0)
                if self._old: signal.signal(signal.SIGALRM, self._old)
    return _A(sec)

def _count_lines_worker(path: str, soft_timeout: int) -> Tuple[str, int]:
    with _soft_alarm(soft_timeout):
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f: n += 1
        return (path, n)

# -------------------- 阶段1：覆盖率与待统计枚举 --------------------
def audit_candidates_layer(cfg: dict, s4_root: str, cand_root: str,
                           pi_cache: dict) -> Tuple[dict, List[Tuple[str,str,str,str]]]:
    """
    返回：(分层统计字典, 待统计的jsonl明细清单)
    清单元素： (proto, split_name, subset, jsonl_path)
    """
    total_lines = 0
    proto_dirs = [d for d in sorted(glob.glob(os.path.join(s4_root, "*"))) if os.path.isdir(d)]
    for proto_dir in proto_dirs:
        split_dirs = [d for d in sorted(glob.glob(os.path.join(proto_dir, "split_*"))) if os.path.isdir(d)]
        for split_dir in split_dirs:
            for subset in ("train","calib","test"):
                subset_txt = os.path.join(split_dir, f"{subset}.txt")
                total_lines += len(read_lines(subset_txt))

    print_err(f"[stage1] proto_dirs={len(proto_dirs)} total_split_lines={total_lines}")
    upd_paths, close_paths = percent_bar(total_lines, desc="audit_paths")

    result = {}
    processed_lines = 0
    to_count_files: List[Tuple[str,str,str,str]] = []  # (proto, split, subset, jsonl)

    for proto_dir in proto_dirs:
        proto = os.path.basename(proto_dir)
        result.setdefault(proto, {})
        split_dirs = [d for d in sorted(glob.glob(os.path.join(proto_dir, "split_*"))) if os.path.isdir(d)]
        for split_dir in split_dirs:
            split_name = os.path.basename(split_dir)
            result[proto].setdefault(split_name, {})
            for subset in ("train","calib","test"):
                subset_txt = os.path.join(split_dir, f"{subset}.txt")
                lines = read_lines(subset_txt)
                n_lines = len(lines)

                cand_dir = os.path.join(cand_root, proto, split_name, subset)

                resolved, want_files = [], []
                for ln in lines:
                    p = resolve_step_path(ln, cfg, pi_cache)
                    if p:
                        resolved.append(p)
                        want_files.append(os.path.join(cand_dir, f"{md5_16(p)}.jsonl"))
                    processed_lines += 1
                    upd_paths(processed_lines)

                # 一次性扫描目录，集合判断，避免逐文件 isfile I/O
                present = set()
                if os.path.isdir(cand_dir):
                    try:
                        with os.scandir(cand_dir) as it:
                            present = {e.name for e in it if e.is_file()}
                    except FileNotFoundError:
                        present = set()

                existing = [fp for fp in want_files if os.path.basename(fp) in present]
                missing  = [fp for fp in want_files if os.path.basename(fp) not in present]

                for fp in existing:
                    to_count_files.append((proto, split_name, subset, fp))

                result[proto][split_name][subset] = {
                    "split_lines": n_lines,
                    "resolved_paths": len(resolved),
                    "expected_files": len(want_files),
                    "existing_files": len(existing),
                    "missing_files": len(missing),
                    "coverage_existing_over_split": (len(existing) / n_lines) if n_lines else 0.0,
                    "coverage_existing_over_resolved": (len(existing) / len(resolved)) if resolved else 0.0,
                    "empty_file_count": None,
                    "empty_file_rate": None,
                    "cand_per_part_mean": None,
                    "cand_per_part_median": None,
                    "cand_per_part_p90": None,
                    "cand_per_part_max": None,
                }

    close_paths()
    print_err(f"[stage1] existing_jsonl_total={len(to_count_files)}")
    return result, to_count_files

# -------------------- 阶段2：多进程计数 --------------------
def _load_cache(cache_path: Optional[str]) -> Dict[str, dict]:
    if not cache_path or not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache_path: Optional[str], cache: Dict[str, dict]):
    if not cache_path: return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print_err(f"[cache] save failed: {e}")

def _append_err(err_path: Optional[str], rec: dict):
    if not err_path: return
    try:
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print_err(f"[err-log] write failed: {e}")

def _enumerate_and_sort_tasks(to_count_files: List[Tuple[str,str,str,str]], cache: Dict[str,dict]) -> List[Tuple[str,str,str,str,int,float]]:
    """
    返回任务清单(带 size, mtime)：(proto, sp, subset, path, size, mtime)
    已缓存且未变化的跳过。按目录批量 scandir 降低 I/O，并提供进度。
    """
    by_dir = defaultdict(list)
    for proto, sp, subset, fp in to_count_files:
        by_dir[os.path.dirname(fp)].append((proto, sp, subset, fp))

    tasks = []
    cached_hits = 0
    total = len(to_count_files)
    upd, close = percent_bar(total, desc="stat_jsonl")

    processed = 0
    for d, items in by_dir.items():
        stats_map = {}
        if os.path.isdir(d):
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_file():
                            try:
                                st = e.stat()
                                stats_map[os.path.join(d, e.name)] = (int(st.st_size), float(st.st_mtime))
                            except FileNotFoundError:
                                pass
            except FileNotFoundError:
                pass

        for proto, sp, subset, fp in items:
            tup = stats_map.get(fp)
            if not tup:
                processed += 1
                upd(processed)
                continue
            size, mtime = tup
            ent = cache.get(fp)
            if ent and ent.get("size")==size and abs(ent.get("mtime",0.0)-mtime)<1e-6 and isinstance(ent.get("count"), int):
                cached_hits += 1
            else:
                tasks.append((proto, sp, subset, fp, size, mtime))
            processed += 1
            upd(processed)

    close()
    tasks.sort(key=lambda x: x[4], reverse=True)
    return tasks, cached_hits

def _run_pool(tasks: List[Tuple[str,str,str,str,int,float]],
              result_layer: dict,
              workers: int, multiplier: int,
              soft_timeout: int, hard_timeout: int, watchdog_sec: int,
              err_log: Optional[str], cache: Dict[str,dict]):
    total = len(tasks)
    if total == 0:
        print_err("[stage2] nothing to count (all cached or none existing)")
        return

    inflight_max = max(1, workers*max(1, multiplier))
    print_err(f"[stage2] to_count={total} workers={workers} inflight_max={inflight_max} soft={soft_timeout}s hard={hard_timeout}s wd={watchdog_sec}s")

    agg_vals = defaultdict(list)  # key=(proto,sp,subset) -> [counts]
    upd, close = percent_bar(total, desc="count_jsonl")

    def _new_pool():
        return ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, mp_context=mp.get_context("spawn"))

    pending = deque(tasks)
    inflight = {}
    started = {}
    done_cnt = 0
    last_hb = time.time()

    ex = _new_pool()

    def _submit_some():
        while pending and len(inflight) < inflight_max:
            job = pending.popleft()
            proto, sp, subset, path, size, mtime = job
            fut = ex.submit(_count_lines_worker, path, soft_timeout)
            inflight[fut] = job
            started[fut] = time.time()

    _submit_some()

    while inflight or pending:
        done_futs, _ = wait(list(inflight.keys()), timeout=watchdog_sec, return_when=FIRST_COMPLETED)
        now = time.time()  # 放在 wait 之后，硬超时判断更准确

        if (now - last_hb) >= max(1, watchdog_sec):
            print_err(f"[pool] hb done={done_cnt} inflight={len(inflight)} pending={len(pending)}")
            last_hb = now

        aged = [f for f in list(inflight.keys()) if (now - started.get(f, now)) > hard_timeout]
        if aged:
            print_err(f"[pool] watchdog: {len(aged)} aged futures -> restart pool")
            for f, job in list(inflight.items()):
                if f not in done_futs:
                    pending.appendleft(job)
            try:
                ex.shutdown(cancel_futures=True)
            except Exception:
                pass
            inflight.clear(); started.clear()
            ex = _new_pool()
            _submit_some()
            continue

        for f in done_futs:
            job = inflight.pop(f)
            started.pop(f, None)
            proto, sp, subset, path, size, mtime = job
            try:
                path_ret, cnt = f.result()
                if path_ret != path:
                    _append_err(err_log, {"phase":"count","path":path,"error":"path_mismatch"})
                agg_vals[(proto,sp,subset)].append(int(cnt))
                cache[path] = {"size": size, "mtime": mtime, "count": int(cnt)}
            except TimeoutError:
                _append_err(err_log, {"phase":"count","path":path,"error":"soft_timeout"})
            except Exception as e:
                _append_err(err_log, {"phase":"count","path":path,"error":repr(e)})
            done_cnt += 1
            upd(done_cnt)

        _submit_some()

    close()
    print_err(f"[stage2] finished: done={done_cnt}")

    for (proto, sp, subset), vals in agg_vals.items():
        m = result_layer.get(proto, {}).get(sp, {}).get(subset)
        if not m:
            continue
        if vals:
            empty = sum(1 for c in vals if c == 0)
            m["empty_file_count"] = int(empty)
            m["empty_file_rate"]  = empty / len(vals)
            m["cand_per_part_mean"]    = statistics.mean(vals)
            m["cand_per_part_median"]  = statistics.median(vals)
            m["cand_per_part_p90"]     = pctl(vals, 0.9)
            m["cand_per_part_max"]     = max(vals)
        else:
            m["empty_file_count"] = 0
            m["empty_file_rate"]  = 0.0
            m["cand_per_part_mean"]   = 0.0
            m["cand_per_part_median"] = 0.0
            m["cand_per_part_p90"]    = 0.0
            m["cand_per_part_max"]    = 0


# -------------------- 主流程与输出 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--s6-root")
    ap.add_argument("--s4-root")
    ap.add_argument("--candidates-root")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8)//2))
    ap.add_argument("--multiplier", type=int, default=2)
    ap.add_argument("--soft-timeout", type=int, default=10)
    ap.add_argument("--hard-timeout", type=int, default=60)
    ap.add_argument("--watchdog-sec", type=int, default=5)
    ap.add_argument("--err-log", default=None)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    # spawn，避免 fork 卡死
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg = load_config(args.config)
    s6_root = args.s6_root or (cfg.get("s6", {}) or {}).get("out_root") or ""
    if not s6_root:
        print("ERROR: s6_root not provided and not found in config", file=sys.stderr); sys.exit(2)
    s4_root = args.s4_root or os.path.join(cfg.get("project_root", os.path.dirname(s6_root)), "s4_out")
    cand_root = args.candidates_root or os.path.join(s6_root, "candidates")

    # part_index for resolve
    s5in = (cfg.get("s5") or {}).get("input") or {}
    pi_csv = s5in.get("part_index_csv", "")
    pi_cache = load_part_index_cache(pi_csv,
                                     part_id_col=s5in.get("part_id_column","part_id"),
                                     rel_path_col=s5in.get("path_column","rel_path"),
                                     dataset_col=s5in.get("source_dataset_column","source_dataset"))

    # 压力集报告
    stress_md = os.path.join(s6_root, "recall_stress_report.md")
    stress = parse_stress_report(stress_md)

    # 阶段1：覆盖率与待统计
    per_layer, to_count_files = audit_candidates_layer(cfg, s4_root, cand_root, pi_cache)

    # 阶段2：计数（多进程）
    cache = _load_cache(args.cache)
    tasks, cached_hits = _enumerate_and_sort_tasks(to_count_files, cache)
    print_err(f"[stage2] cached_hits={cached_hits} to_count_after_cache={len(tasks)} total_existing={len(to_count_files)}")
    _run_pool(tasks, per_layer,
              workers=max(1,args.workers), multiplier=max(1,args.multiplier),
              soft_timeout=max(1,args.soft_timeout), hard_timeout=max(2,args.hard_timeout),
              watchdog_sec=max(1,args.watchdog_sec),
              err_log=args.err_log, cache=cache)
    _save_cache(args.cache, cache)

    # 最差子集
    worst = []
    for proto, splits in per_layer.items():
        for sp, subs in splits.items():
            for subset, m in subs.items():
                worst.append((m.get("coverage_existing_over_split",0.0), proto, sp, subset))
    worst_sorted = sorted(worst)[:5]

    metrics = {
        "paths": {
            "s6_root": s6_root,
            "s4_root": s4_root,
            "candidates_root": cand_root,
            "stress_report": stress_md,
        },
        "stress_report": stress,
        "rebuild_layer_metrics": per_layer,
        "worst_coverage_over_split_top5": worst_sorted,
    }

    out_json = os.path.join(s6_root, "s6_audit_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out_md = os.path.join(s6_root, "s6_audit_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# S6 Audit Report\n\n")
        tr = stress.get("target_recall")
        sel = (stress.get("stress_selected") or {}).get("recall")
        f.write(f"- target_recall: {tr if tr is not None else 'n/a'}\n")
        f.write("- selected_recall: n/a\n" if sel is None else f"- selected_recall: {sel:.4f}\n")
        if tr is not None and sel is not None:
            f.write(f"- pass: {bool(sel >= tr)}\n")
        if "stress_baseline" in stress:
            sb = stress["stress_baseline"]
            f.write(f"- stress_baseline: recall={sb['recall']:.4f} lat={sb['lat']:.4f} cpp={sb['cpp']:.2f} n={sb['n']}\n")
        if "stress_selected" in stress:
            ss = stress["stress_selected"]
            f.write(f"- stress_selected: recall={ss['recall']:.4f} lat={ss['lat']:.4f} cpp={ss['cpp']:.2f}\n")
        if "real_baseline" in stress:
            rb = stress["real_baseline"]
            f.write(f"- real_baseline: lat={rb['lat']:.4f} cpp={rb['cpp']:.2f} n={rb['n']}\n")

        f.write("\n## Rebuild coverage (by protocol/split/subset)\n")
        for proto, splits in per_layer.items():
            f.write(f"\n### Protocol: {proto}\n")
            for sp, subs in splits.items():
                f.write(f"- Split: {sp}\n")
                for subset, m in subs.items():
                    f.write(
                        f"  - {subset}: split_lines={m['split_lines']} resolved={m['resolved_paths']} "
                        f"existing={m['existing_files']} missing={m['missing_files']} "
                        f"coverage/split={m['coverage_existing_over_split']:.3f} "
                        f"coverage/resolved={m['coverage_existing_over_resolved']:.3f} "
                        f"empty_rate={m['empty_file_rate'] if m['empty_file_rate'] is not None else 0.0:.3f} "
                        f"cand_mean={m['cand_per_part_mean'] if m['cand_per_part_mean'] is not None else 0.0:.2f} "
                        f"cand_p50={m['cand_per_part_median'] if m['cand_per_part_median'] is not None else 0.0:.2f} "
                        f"cand_p90={m['cand_per_part_p90'] if m['cand_per_part_p90'] is not None else 0.0:.2f} "
                        f"cand_max={m['cand_per_part_max'] if m['cand_per_part_max'] is not None else 0}\n"
                    )
        if worst_sorted:
            f.write("\n## Worst coverage subsets (over split)\n")
            for cov, proto, sp, subset in worst_sorted:
                f.write(f"- {proto}/{sp}/{subset}: coverage_over_split={cov:.3f}\n")
        f.flush()

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")

if __name__ == "__main__":
    main()
