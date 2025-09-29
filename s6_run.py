# -*- coding: utf-8 -*-
"""
s6_run.py
S6：压力测试与阈值回调 + 重建 candidates。
修订：
- 强制 multiprocessing 'spawn'，避免 OCCT 在 fork 下卡死。
- 不再读取 s5.resources；统一用 s6.resources（或安全默认）。
- 路径解析与存在性检查下放到 worker，主进程不做逐文件 stat/解析。
- 枚举后即时记录任务总数并显示 rebuild 进度。
"""
import os, json, time, statistics, random, csv, logging, signal, math
import multiprocessing as mp
from functools import lru_cache
from types import SimpleNamespace
from dataclasses import asdict, is_dataclass
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import deque, defaultdict
from tqdm import tqdm

from s6_core import (
    setup_logging_from_config, generate_stress_set,
    load_s5_params, save_params, backup_file,
    match_recall, find_candidates, write_candidates_jsonl,
    iter_split_dirs, read_lines, hash_path16,
    HoleGT, iter_protocol_dirs, S5Params
)

cfg_cache = {}

# ---------- 统计 ----------
def _wilson_ci(p, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5) / denom
    return max(0.0, center-half), min(1.0, center+half)

def _pctl(xs, q):
    if not xs: return 0.0
    xs = sorted(xs)
    k = (len(xs)-1)*q
    f = math.floor(k); c = math.ceil(k)
    if f == c: return xs[int(k)]
    return xs[f] + (xs[c]-xs[f])*(k-f)

# ---------- s6 资源 ----------
def _get_s6_resources(s6cfg: dict):
    r = (s6cfg.get("resources") or {})
    workers = int(r.get("num_workers", max(1, os.cpu_count() or 4)))
    inflight_mult = int(r.get("max_inflight_multiplier", 2))
    soft = int(r.get("per_file_timeout_sec", 50))
    hard = int(r.get("hard_timeout_sec", 60))
    poll = float(r.get("watchdog_poll_sec", 5.0))
    return workers, inflight_mult, soft, hard, poll

# ---------- 路径解析 ----------
@lru_cache(maxsize=1)
def _load_part_index_cache_tuple(part_index_csv, part_id_col, rel_path_col, dataset_col):
    cache = {}
    if not part_index_csv or not os.path.isfile(part_index_csv):
        return cache
    with open(part_index_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get(part_id_col)
            relp = row.get(rel_path_col)
            ds = row.get(dataset_col)
            if pid and relp and ds:
                cache[pid] = (ds, relp)
    return cache

def _load_part_index_cache(cfg):
    s5in = (cfg.get("s5") or {}).get("input") or {}
    return _load_part_index_cache_tuple(
        s5in.get("part_index_csv"),
        s5in.get("part_id_column", "part_id"),
        s5in.get("path_column", "rel_path"),
        s5in.get("source_dataset_column", "source_dataset"),
    )

def _resolve_step_path(line: str, cfg):
    if not line:
        return None
    s = line.strip().strip('"').strip("'")
    if os.path.isabs(s) and os.path.isfile(s): return s
    if os.path.isfile(s): return os.path.abspath(s)
    s5in = (cfg.get("s5") or {}).get("input") or {}
    rel_roots = s5in.get("rel_root_map", {}) or {}
    norm = s.replace("\\", "/")
    segs = norm.split("/")
    if segs and segs[0] in rel_roots:
        cand = os.path.join(rel_roots[segs[0]], "/".join(segs[1:]))
        if os.path.isfile(cand): return cand
    idx = _load_part_index_cache(cfg)
    if s in idx:
        ds, relp = idx[s]
        root = rel_roots.get(ds)
        if root:
            cand = os.path.join(root, relp)
            if os.path.isfile(cand): return cand
    for root in rel_roots.values():
        cand = os.path.join(root, norm)
        if os.path.isfile(cand): return cand
    return None

# ---------- 并行基础 ----------
def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.ERROR)

class _alarm:
    def __init__(self, sec):
        self.sec = int(sec) if sec else 0
        self._old = None
    def __enter__(self):
        if self.sec > 0:
            self._old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("soft_timeout")))
            signal.alarm(self.sec)
    def __exit__(self, exc_type, exc, tb):
        if self.sec > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)

def _run_windowed(tasks, worker_fn, agg_fn, total, desc,
                  workers, inflight_mult, soft_timeout_sec, hard_timeout_sec, poll_sec):
    if total == 0:
        return {"done":0,"timeouts":0,"errors":0,"median_ms":0.0,"p90_ms":0.0}
    inflight_max = max(1, int(workers * max(1, inflight_mult)))

    def _new_pool():
        return ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)

    pending = deque(tasks)
    inflight = {}
    started = {}
    durations_ms = []
    err_counter = defaultdict(int)
    timeouts = 0

    ex = _new_pool()
    bar = tqdm(total=total, desc=desc, ncols=100)
    step = max(1, total // 100)
    done_cnt = 0

    def _submit_some():
        while pending and len(inflight) < inflight_max:
            job = pending.popleft()
            f = ex.submit(worker_fn, *job, soft_timeout_sec)
            inflight[f] = (job, time.time(), 1)
            started[f] = inflight[f][1]

    _submit_some()

    while inflight or pending:
        done, _ = wait(list(inflight.keys()), timeout=poll_sec, return_when=FIRST_COMPLETED)

        now = time.time()
        aged = [f for f in list(inflight.keys()) if (now - started.get(f, now)) > hard_timeout_sec]
        if not done and aged:
            aged_jobs = [inflight[f][0] for f in aged]
            for f in list(inflight.keys()):
                try: f.cancel()
                except: pass
            ex.shutdown(cancel_futures=True)
            for j in aged_jobs:
                pending.appendleft(j)
            inflight.clear(); started.clear()
            ex = _new_pool()
            _submit_some()
            continue

        for f in done:
            job, st, _ = inflight.pop(f)
            started.pop(f, None)
            try:
                res = f.result()
                lat = None
                if isinstance(res, tuple) and len(res) >= 2:
                    for x in res[1:]:
                        if isinstance(x,(int,float)):
                            lat = float(x); break
                if lat is not None:
                    durations_ms.append(1000.0*lat)
                agg_fn(res)
            except TimeoutError:
                timeouts += 1
                agg_fn(("timeout", job))
            except Exception as e:
                err_counter[type(e).__name__] += 1
                agg_fn(("error", str(e), job))

            done_cnt += 1
            if (done_cnt % step) == 0 or done_cnt == total:
                inc = step if done_cnt != total else (total - bar.n)
                if inc > 0: bar.update(inc)

        _submit_some()

    rem = total - bar.n
    if rem > 0: bar.update(rem)
    bar.close()

    return {
        "done": done_cnt,
        "timeouts": timeouts,
        "errors": sum(err_counter.values()),
        "median_ms": statistics.median(durations_ms) if durations_ms else 0.0,
        "p90_ms": _pctl(durations_ms, 0.9) if durations_ms else 0.0,
        "err_breakdown": dict(err_counter)
    }

# ---------- 并行 worker ----------
def _stress_worker(step_path, params_dict, tol_dict, soft_timeout_sec):
    params = S5Params(**params_dict)
    with _alarm(soft_timeout_sec):
        t0 = time.time()
        preds = find_candidates(step_path, params)
        lat = time.time() - t0
        cpp = len(preds)
        gt = [HoleGT(**h) if isinstance(h, dict) else h for h in tol_dict["gt"]]
        rec = match_recall(
            gt, preds,
            axis_tol_deg=tol_dict["axis_tol_deg"],
            axis_tol_deg_hi=tol_dict["axis_tol_deg_hi"],
            entrance_center_tol_frac=tol_dict["entrance_center_tol_frac"],
            entrance_center_tol_mm=tol_dict["entrance_center_tol_mm"],
            iou_axis_min=tol_dict["iou_axis_min"],
            require_center=tol_dict["require_center"]
        )
    return ("ok", rec, lat, cpp, len(gt))

def _real_worker(abs_step_path, params_dict, soft_timeout_sec):
    params = S5Params(**params_dict)
    with _alarm(soft_timeout_sec):
        t0 = time.time()
        preds = find_candidates(abs_step_path, params)
        lat = time.time() - t0
        cpp = len(preds)
    return ("ok", lat, cpp)

# --- s6_core.py 内：替换 _rebuild_worker_line ---
def _rebuild_worker_line(raw_line, out_dir, params_dict, cfg_dict, soft_timeout_sec):
    # 由 worker 解析真实路径并写结果，避免主进程长时间阻塞
    p = _resolve_step_path(raw_line, cfg_dict)
    if not p:
        return ("skip_unresolved", 0.0, 0, raw_line)
    out_file = os.path.join(out_dir, f"{hash_path16(p)}.jsonl")
    if os.path.exists(out_file):
        return ("skip_exists", 0.0, 0, p)
    params = S5Params(**params_dict)
    with _alarm(soft_timeout_sec):
        t0 = time.time()
        preds = find_candidates(p, params)
        write_candidates_jsonl(out_file, preds)
        lat = time.time() - t0
    return ("ok", lat, len(preds), p)


# ---------- 评测：压力集 ----------
def _eval_on_stress(items, params, s6cfg):
    workers, inflight_mult, soft, hard, poll = _get_s6_resources(s6cfg)

    tasks = []
    tol_base = {
        "axis_tol_deg": float(params.axis_coax_tol_deg),
        "axis_tol_deg_hi": float(params.axis_coax_tol_deg) + 3.0,
        "entrance_center_tol_frac": 0.3,
        "entrance_center_tol_mm": 0.2,
        "iou_axis_min": 0.7,
        "require_center": (not bool(params.missing_ring_fallback))
    }
    for it in items:
        gt_list = []
        for h in it.holes:
            if is_dataclass(h): gt_list.append(asdict(h))
            elif isinstance(h, dict): gt_list.append(h)
            else: gt_list.append(vars(h))
        tol = dict(tol_base); tol["gt"] = gt_list
        tasks.append((it.step_path, params.__dict__, tol))

    # 无排序，直接执行
    hits, total, lats, cpps = 0, 0, [], []

    def _agg(res):
        nonlocal hits, total, lats, cpps
        if not isinstance(res, tuple): return
        tag = res[0]
        if tag == "ok":
            _, rec, lat, cpp, n = res
            hits += rec * max(1, n)
            total += max(1, n)
            lats.append(lat); cpps.append(cpp)
        elif tag == "timeout":
            total += 1

    stats = _run_windowed(tasks, _stress_worker, _agg, len(tasks), "eval_stress",
                          workers, inflight_mult, soft, hard, poll)

    recall = hits / max(1, total)
    med_latency = statistics.median(lats) if lats else 0.0
    cand_pp = statistics.mean(cpps) if cpps else 0.0
    logging.info("eval_stress stats: median_ms=%.1f p90_ms=%.1f timeouts=%d errors=%d",
                 stats["median_ms"], stats["p90_ms"], stats["timeouts"], stats["errors"])
    if stats.get("err_breakdown"):
        logging.info("eval_stress error_breakdown=%s", stats["err_breakdown"])
    return recall, med_latency, cand_pp, total

# ---------- 采样真实切片 ----------
def _sample_real_paths(s4_root, sample_n=300, seed=123, subsets=("train","calib"), apply_protocols=None):
    paths = []
    for proto, _ in iter_protocol_dirs(s4_root, apply_protocols or []):
        for split_dir in iter_split_dirs(s4_root, proto):
            for subset in subsets:
                fp = os.path.join(split_dir, f"{subset}.txt")
                if os.path.exists(fp):
                    paths.extend(read_lines(fp))
    if not paths:
        return []
    rnd = random.Random(seed)
    if len(paths) > sample_n:
        paths = [paths[i] for i in rnd.sample(range(len(paths)), sample_n)]
    return paths

# ---------- 评测真实切片 ----------
def _eval_real_slice(paths, params, cfg, s6cfg):
    if not paths:
        return 0.0, 0.0, 0

    workers, inflight_mult, soft, hard, poll = _get_s6_resources(s6cfg)

    resolved = []
    for raw in paths:
        p = _resolve_step_path(raw, cfg)
        if p: resolved.append(p)
    if not resolved:
        logging.error("no valid STEP resolved in real slice; check rel_root_map and split contents")
        return 0.0, 0.0, 0

    tasks = [(p, params.__dict__) for p in resolved]

    lats, cpps, ok = [], [], 0

    def _agg(res):
        nonlocal ok
        if not isinstance(res, tuple): return
        tag = res[0]
        if tag == "ok":
            _, lat, cpp = res
            lats.append(lat); cpps.append(cpp); ok += 1

    stats = _run_windowed(tasks, _real_worker, _agg, len(tasks), "eval_real",
                          workers, inflight_mult, soft, hard, poll)

    med_latency = statistics.median(lats) if lats else 0.0
    cand_pp = statistics.mean(cpps) if cpps else 0.0
    logging.info("eval_real stats: median_ms=%.1f p90_ms=%.1f timeouts=%d errors=%d",
                 stats["median_ms"], stats["p90_ms"], stats["timeouts"], stats["errors"])
    if stats.get("err_breakdown"):
        logging.info("eval_real error_breakdown=%s", stats["err_breakdown"])
    return med_latency, cand_pp, ok

# ---------- 调参 ----------
def _tune(cfg, s6cfg, base_params, stress_items):
    target = s6cfg.get("target_recall") or (cfg.get("s5", {}).get("recall_guard", {}).get("target_recall", 0.98))
    grid = s6cfg.get("grid_size", 3)
    stages = s6cfg.get("stages", ["A","B","C"])

    logging.info(
        "eval tolerances: axis_tol=%.2f deg (hi %.2f), center_tol=%.2f*D or %.2f mm, IoU_axis>=%.2f, require_center_on_stress=%s",
        base_params.axis_coax_tol_deg, base_params.axis_coax_tol_deg + 3.0, 0.3, 0.2, 0.7, (not bool(base_params.missing_ring_fallback))
    )

    base_recall, base_lat_s, base_cpp_s, n_stress = _eval_on_stress(stress_items, base_params, s6cfg)
    logging.info("baseline(stress): recall=%.4f med_latency=%.4f cand/part=%.2f",
                 base_recall, base_lat_s, base_cpp_s)

    real_cfg = (s6cfg.get("real_slice_check") or {})
    sample_n = int(real_cfg.get("sample_n", 300))
    sample_seed = int(real_cfg.get("sample_seed", 123))
    subsets = tuple(real_cfg.get("subsets", ["train","calib"]))
    real_paths = _sample_real_paths(s6cfg["s4_root"], sample_n=sample_n, seed=sample_seed,
                                    subsets=subsets, apply_protocols=s6cfg.get("apply_protocols", []))
    base_lat_r, base_cpp_r, n_real = _eval_real_slice(real_paths, base_params, cfg, s6cfg)
    logging.info("baseline(real): med_latency=%.4f cand/part=%.2f (n=%d)", base_lat_r, base_cpp_r, n_real)

    best = base_params
    best_score = (base_recall, base_lat_s, base_cpp_s)

    import numpy as np
    def gen_stage(stg, p:S5Params):
        if stg == "A":
            for da in np.linspace(0, 1.0, grid):
                q = S5Params(**p.__dict__)
                q.axis_coax_tol_deg = p.axis_coax_tol_deg + da
                q.entrance_roundness_tol = p.entrance_roundness_tol + 0.01 * da
                q.circle_fit_ransac_it = int(p.circle_fit_ransac_it * (1 + 0.2 * da))
                yield q
        elif stg == "B":
            for dm in np.linspace(0, 1.0, grid):
                q = S5Params(**p.__dict__)
                q.merge_gap_tol = p.merge_gap_tol * (1 + 0.25 * dm)
                q.cyl_len_min = max(0.05, p.cyl_len_min * (1 - 0.5 * dm))
                yield q
        elif stg == "C":
            for dv in np.linspace(0, 1.0, grid):
                q = S5Params(**p.__dict__)
                q.dihedral_var_max = p.dihedral_var_max * (1 + 0.3 * dv)
                q.cone_angle_tol_deg = p.cone_angle_tol_deg + 0.5 * dv
                yield q

    # 资源限制（覆盖率↑，延时/候选开销不超 baseline 的放大系数）
    mr = s6cfg.get("max_candidates_per_part_factor", 1.35)
    ml = s6cfg.get("max_latency_factor", 1.25)

    for stg in stages:
        tried = 0
        for cand in gen_stage(stg, best):
            r_s, lat_s, cpp_s, _ = _eval_on_stress(stress_items, cand, s6cfg)
            lat_r, cpp_r, _ = _eval_real_slice(real_paths, cand, cfg, s6cfg)
            ok = (r_s >= target) and \
                 (lat_r <= base_lat_r * ml if base_lat_r > 0 else True) and \
                 (cpp_r <= base_cpp_r * mr if base_cpp_r > 0 else cpp_r <= 50)
            if tried < 12:
                logging.info("try %s: stress(recall=%.4f lat=%.4f cpp=%.2f) real(lat=%.4f cpp=%.2f) ok=%s",
                             stg, r_s, lat_s, cpp_s, lat_r, cpp_r, ok)
            tried += 1
            if ok and (r_s > best_score[0] or (abs(r_s - best_score[0]) < 1e-6 and lat_s < best_score[1])):
                best, best_score = cand, (r_s, lat_s, cpp_s)
        if best_score[0] >= target:
            break

    ci_lo, ci_hi = _wilson_ci(best_score[0], max(1, n_stress))
    extra = {
        "stress_base": {"recall": base_recall, "lat": base_lat_s, "cpp": base_cpp_s, "n": n_stress},
        "stress_best": {"recall": best_score[0], "lat": best_score[1], "cpp": best_score[2], "ci95": [ci_lo, ci_hi]},
        "real_base": {"lat": base_lat_r, "cpp": base_cpp_r, "n": n_real},
        "eval_tolerances": {
            "axis_tol_deg": base_params.axis_coax_tol_deg,
            "axis_tol_deg_hi": base_params.axis_coax_tol_deg + 3.0,
            "entrance_center_tol_frac": 0.3,
            "entrance_center_tol_mm": 0.2,
            "iou_axis_min": 0.7,
            "require_center_on_stress": (not bool(base_params.missing_ring_fallback))
        }
    }
    return best, best_score, extra, target

# ---------- 重建候选 ----------
# --- s6_run.py 内：替换 _rebuild_candidates ---
def _rebuild_candidates(cfg, s6cfg, params):
    s4_root = s6cfg["s4_root"]
    out_root = s6cfg["out_root"]
    cand_root = os.path.join(out_root, "candidates")
    os.makedirs(cand_root, exist_ok=True)

    workers, inflight_mult, soft, hard, poll = _get_s6_resources(s6cfg)

    # 统计容器：protocol -> split -> subset -> counters
    summary = {}
    jobs = []
    total_lines = 0

    def _bucket(proto, split_name, subset):
        summary.setdefault(proto, {}).setdefault(split_name, {}).setdefault(subset, {
            "split_lines": 0,
            "written_ok": 0,
            "skip_exists": 0,
            "skip_unresolved": 0,
            "timeouts": 0,
            "errors": 0,
            "lat_ms": [],
            "cand_counts": []
        })
        return summary[proto][split_name][subset]

    for proto, _ in iter_protocol_dirs(s4_root, s6cfg.get("apply_protocols", [])):
        for split_dir in iter_split_dirs(s4_root, proto):
            name = os.path.basename(split_dir)
            for subset in ["train", "calib", "test"]:
                in_file = os.path.join(split_dir, f"{subset}.txt")
                if not os.path.exists(in_file):
                    continue
                out_dir = os.path.join(cand_root, proto, name, subset)
                os.makedirs(out_dir, exist_ok=True)
                lines = read_lines(in_file)
                total_lines += len(lines)
                _bucket(proto, name, subset)["split_lines"] += len(lines)
                for raw in lines:
                    jobs.append((raw, out_dir, params.__dict__, cfg))

    logging.info("rebuild enumerate: total_tasks=%d", total_lines)

    def _agg(res):
        if not isinstance(res, tuple):
            return
        tag = res[0]
        # res 结构: (tag, lat, cpp, path_or_line)
        if tag in ("ok", "skip_exists", "skip_unresolved", "timeout", "error"):
            # 从 candidates 目录结构逆推 proto/split/subset 无法直接得出；
            # 因此仅汇总全局延迟与cand统计，同时在末尾再按文件系统核对覆盖率（由审计脚本负责逐层）
            pass

    lat_list, cpp_list = [], []
    err_counter = {"timeouts":0, "errors":0, "ok":0, "skip_exists":0, "skip_unresolved":0}

    def _agg_simple(res):
        nonlocal lat_list, cpp_list, err_counter
        tag = res[0]
        if tag == "ok":
            _, lat, cpp, _ = res
            lat_list.append(lat); cpp_list.append(cpp)
            err_counter["ok"] += 1
        elif tag == "skip_exists":
            err_counter["skip_exists"] += 1
        elif tag == "skip_unresolved":
            err_counter["skip_unresolved"] += 1
        elif tag == "timeout":
            err_counter["timeouts"] += 1
        elif tag == "error":
            err_counter["errors"] += 1

    stats = _run_windowed(jobs, _rebuild_worker_line, _agg_simple, len(jobs), "rebuild",
                          workers, inflight_mult, soft, hard, poll)
    logging.info("rebuild stats: median_ms=%.1f p90_ms=%.1f timeouts=%d errors=%d",
                 stats["median_ms"], stats["p90_ms"], stats["timeouts"], stats["errors"])
    if stats.get("err_breakdown"):
        logging.info("rebuild error_breakdown=%s", stats["err_breakdown"])

    # 全局级别的简要统计落盘
    rebuild_summary = {
        "totals": {
            "tasks": len(jobs),
            "ok": err_counter["ok"],
            "skip_exists": err_counter["skip_exists"],
            "skip_unresolved": err_counter["skip_unresolved"],
            "timeouts": stats["timeouts"],
            "errors": stats["errors"],
            "median_ms": stats["median_ms"],
            "p90_ms": stats["p90_ms"],
            "lat_ms_median_local": (statistics.median([x*1000 for x in lat_list]) if lat_list else 0.0),
            "cand_per_part_mean_local": (statistics.mean(cpp_list) if cpp_list else 0.0),
            "cand_per_part_p90_local": (_pctl(cpp_list, 0.9) if cpp_list else 0.0)
        },
        # 细粒度覆盖率与空文件率建议由外部审计脚本计算（见下文）
    }
    out_json = os.path.join(out_root, "rebuild_summary.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rebuild_summary, f, indent=2)
        logging.info("write rebuild summary: %s", out_json)
    except Exception as e:
        logging.warning("write rebuild_summary.json failed: %s", e)


# ---------- 主入口 ----------
def run(config_path: str):
    # 强制 spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg = setup_logging_from_config(config_path)
    global cfg_cache
    cfg_cache = cfg

    s6cfg = dict(cfg.get("s6", {}))
    out_root = s6cfg.get("out_root")
    if not out_root:
        raise ValueError("s6.out_root is required")
    os.makedirs(out_root, exist_ok=True)
    s6cfg.setdefault("s4_root", os.path.join(cfg.get("project_root", out_root), "s4_out"))
    s6cfg.setdefault("log_file", os.path.join(out_root, "pipeline.log"))
    # s6.resources 默认值（不读取 s5.resources）
    s6cfg.setdefault("resources", {})

    stress_dir = os.path.join(out_root, "stress_step")
    report_file = os.path.join(out_root, "recall_stress_report.md")
    params_out = os.path.join(out_root, "tuned_s5_params.json")
    params_backup = os.path.join(out_root, "tuned_s5_params.prev.json")

    # 压力集
    s5_rg = cfg.get("s5", {}).get("recall_guard", {})
    stress_total = int(s6cfg.get("stress_total", 600))
    stress_seed = s6cfg.get("stress_seed") or s5_rg.get("stress_synth_seed", 20250926)
    manifest = os.path.join(stress_dir, "manifest.json")
    if not os.path.exists(manifest):
        items = generate_stress_set(stress_dir, total=stress_total, seed=int(stress_seed))
    else:
        data = json.load(open(manifest, "r", encoding="utf-8"))
        items = [SimpleNamespace(step=it["step"], holes=[HoleGT(**h) for h in it["holes"]]) for it in data]
        for it in items: it.step_path = it.step

    base = load_s5_params(cfg)
    best, score, extra, target = _tune(cfg, s6cfg, base, items)
    logging.info("best(stress): recall=%.4f med_latency=%.4f cand/part=%.2f (target=%.3f)",
                 score[0], score[1], score[2], target)

    backup_file(params_out, params_backup)
    save_params(params_out, best)

    # 重建
    _rebuild_candidates(cfg, s6cfg, best)

    # 报告
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# S6 Recall-Stress Report\n\n")
        f.write(f"- target_recall: {target}\n")
        f.write(f"- baseline(stress): recall={extra['stress_base']['recall']:.4f} "
                f"med_latency={extra['stress_base']['lat']:.4f} cand/part={extra['stress_base']['cpp']:.2f} "
                f"n={extra['stress_base']['n']}\n")
        f.write(f"- selected(stress): recall={score[0]:.4f} med_latency={score[1]:.4f} cand/part={score[2]:.2f}\n")
        f.write(f"- baseline(real): med_latency={extra['real_base']['lat']:.4f} "
                f"cand/part={extra['real_base']['cpp']:.2f} n={extra['real_base']['n']}\n")
        f.write(f"- eval_tolerances: {json.dumps(extra['eval_tolerances'])}\n")
        f.write(f"- tuned_params: {params_out}\n")
        f.write(f"- candidates_root: {os.path.join(s6cfg['out_root'], 'candidates')}\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
