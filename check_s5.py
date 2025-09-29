#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5 结果自检（新目录结构 + 进度条）
目录：/s5_out/{protocol}/split_{k}/{train,calib,test}/*.jsonl
校验点：
  1) 与 S4 切分对齐：expected == written_files + no_cand + failed
  2) JSONL 可解析、字段完整、数值范围合法
  3) 汇总 no_cand_samples.jsonl 的几何直方图
产物：
  - s5_postcheck.json
  - split_summary.csv
  - diag_summary.csv
  - parse_errors_sample.csv   （最多 200 行）
进度：
  - 单一主进度条，单位=part，最小粒度≈1%（miniters=⌈total_parts/100⌉）
用法：
  python s5_check.py --config /path/to/config.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from math import ceil
from tqdm import tqdm

# ---------- io helpers ----------
try:
    import orjson as _orjson
    def jloads(b: bytes) -> Any: return _orjson.loads(b)
except Exception:
    import json as _json
    def jloads(b: bytes) -> Any: return _json.loads(b.decode("utf-8"))

def read_cfg(p: str) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_lines(p: Path) -> List[str]:
    if not p.exists(): return []
    with open(p, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--diag_cap", type=int, default=200, help="no_cand_samples.jsonl 读取上限")
    ap.add_argument("--parse_err_cap", type=int, default=200, help="解析错误样例上限")
    args = ap.parse_args()

    cfg = read_cfg(args.config)
    s5 = cfg.get("S5") or cfg.get("s5")
    assert s5, "config 缺少 'S5'/'s5' 块"

    out_root = Path(s5["output"]["dir"])                # /.../step_out/s5_out
    split_root_s4 = Path(s5["input"]["split_root"])     # /.../step_out/s4_out
    flatten_sep = s5["output"].get("flatten_sep","__")

    def safe_name(pid: str) -> str:
        return pid.replace("/", flatten_sep)

    # 发现 protocol/split
    protos = [d.name for d in out_root.iterdir() if d.is_dir()]
    subset_names = ("train","calib","test")

    # 预计算总 expected parts 数用于进度条
    total_expected = 0
    tasks = []  # (proto, split_id, subset, expected_list)
    for proto in sorted(protos):
        for split_dir in sorted([d for d in (out_root/proto).iterdir() if d.is_dir() and d.name.startswith("split_")]):
            try: sid = int(split_dir.name.split("_")[-1])
            except: continue
            s4_dir = split_root_s4 / proto / f"split_{sid}"
            for subset in subset_names:
                exp_list = read_lines(s4_dir / f"{subset}.txt")
                if exp_list:
                    tasks.append((proto, sid, subset, exp_list))
                    total_expected += len(exp_list)

    # 结果累积器
    rows_split = []
    rows_diag  = []
    parse_err_rows = []

    global_tot = dict(
        splits=0, subsets=0, expected=0, written_files=0, no_cand=0, failed=0,
        jsonl_lines=0, parse_errors=0, bad_values=0, candidates=0
    )

    # 进度条最小粒度≈1%
    miniters = max(1, ceil(total_expected / 100)) if total_expected else 1
    pbar = tqdm(total=total_expected, unit="part", desc="S5 Check", miniters=miniters)

    seen_split = set()

    # 逐 subset（逐 part 粒度）检查
    for proto, sid, subset, expected_list in tasks:
        split_dir = out_root / proto / f"split_{sid}"
        s4_dir    = split_root_s4 / proto / f"split_{sid}"

        if (proto, sid) not in seen_split:
            seen_split.add((proto, sid))
            global_tot["splits"] += 1

            # 读取 split 级别诊断与 summary
            s5_sum_p = split_dir / "summary.json"
            s5_sum = {}
            if s5_sum_p.exists():
                try: s5_sum = json.loads(s5_sum_p.read_text(encoding="utf-8"))
                except: s5_sum = {}

            diag_p = split_dir / "no_cand_samples.jsonl"
            diag_cnt = 0
            diag_acc = dict(faces_total=0, cyl_faces=0, cone_faces=0, plane_faces=0, nurbs_faces=0, bezier_faces=0, rev_faces=0, circ_edges=0)
            if diag_p.exists():
                with open(diag_p, "rb") as f:
                    for i, b in enumerate(f, 1):
                        if i > args.diag_cap: break
                        try:
                            obj = jloads(b); d = obj.get("diag", {}) or {}
                            for k in diag_acc.keys():
                                if k in d: diag_acc[k] += int(d[k])
                            diag_cnt += 1
                        except Exception:
                            continue
            rows_diag.append({"protocol": proto, "split_id": sid, "samples": diag_cnt, **diag_acc})

        global_tot["subsets"] += 1

        subset_dir = split_dir / subset
        no_cand = 0; failed = 0
        # 来自运行时 summary（若无则置 0）
        s5_sum_p = split_dir / "summary.json"
        s5_sum = {}
        if s5_sum_p.exists():
            try: s5_sum = json.loads(s5_sum_p.read_text(encoding="utf-8"))
            except: s5_sum = {}
        no_cand = int((s5_sum.get("subset_no_cand", {}) or {}).get(subset, 0))
        failed  = int((s5_sum.get("subset_failed",   {}) or {}).get(subset, 0))

        written_files = 0
        jsonl_lines = 0
        parse_errors = 0
        bad_values   = 0
        total_cands  = 0
        cyl_segs     = 0
        cone_segs    = 0

        # 逐 part 检查并更新进度（每 part +1）
        for pid in expected_list:
            fp = subset_dir / f"{safe_name(pid)}.jsonl"
            if fp.exists():
                written_files += 1
                try:
                    with open(fp, "rb") as f:
                        for ln, b in enumerate(f, 1):
                            jsonl_lines += 1
                            try:
                                obj = jloads(b)
                            except Exception as e:
                                parse_errors += 1
                                if len(parse_err_rows) < args.parse_err_cap:
                                    parse_err_rows.append({"protocol": proto, "split_id": sid, "subset": subset, "file": str(fp), "line": ln, "err": str(e)})
                                continue
                            miss = [k for k in ("hole_id","primitives","D0","H0","alpha0","q","flags") if k not in obj]
                            if miss:
                                bad_values += 1
                                if len(parse_err_rows) < args.parse_err_cap:
                                    parse_err_rows.append({"protocol": proto, "split_id": sid, "subset": subset, "file": str(fp), "line": ln, "err": f"missing:{miss}"})
                                continue
                            try:
                                D0 = obj["D0"]; H0 = obj["H0"]; q = float(obj["q"])
                                if (D0 is not None and D0 < 0) or (H0 is not None and H0 < 0) or not (0.0 <= q <= 1.0):
                                    bad_values += 1
                                    if len(parse_err_rows) < args.parse_err_cap:
                                        parse_err_rows.append({"protocol": proto, "split_id": sid, "subset": subset, "file": str(fp), "line": ln, "err": f"out_of_range D0={D0} H0={H0} q={q}"})
                                    continue
                                prims = obj.get("primitives", []) or []
                                for seg in prims:
                                    k = seg.get("kind")
                                    if k == "cyl": cyl_segs += 1
                                    elif k == "cone": cone_segs += 1
                                total_cands += 1
                            except Exception as e:
                                bad_values += 1
                                if len(parse_err_rows) < args.parse_err_cap:
                                    parse_err_rows.append({"protocol": proto, "split_id": sid, "subset": subset, "file": str(fp), "line": ln, "err": f"post_parse:{str(e)}"})
                except Exception as e:
                    parse_errors += 1
                    if len(parse_err_rows) < args.parse_err_cap:
                        parse_err_rows.append({"protocol": proto, "split_id": sid, "subset": subset, "file": str(fp), "line": 0, "err": f"open:{str(e)}"})
            # 进度 +1
            pbar.update(1)

        expected = len(expected_list)
        ok_balance = (expected == written_files + no_cand + failed)

        rows_split.append({
            "protocol": proto, "split_id": sid, "subset": subset,
            "expected_parts": expected,
            "written_files": written_files,
            "no_cand_reported": no_cand,
            "failed_reported": failed,
            "balance_ok": bool(ok_balance),
            "jsonl_lines": jsonl_lines,
            "parse_errors": parse_errors,
            "bad_values": bad_values,
            "total_candidates": total_cands,
            "avg_cands_per_written_file": round(total_cands/ max(written_files,1), 6),
            "cyl_segs": cyl_segs,
            "cone_segs": cone_segs,
        })

        # 全局累计
        global_tot["expected"]      += expected
        global_tot["written_files"] += written_files
        global_tot["no_cand"]       += no_cand
        global_tot["failed"]        += failed
        global_tot["jsonl_lines"]   += jsonl_lines
        global_tot["parse_errors"]  += parse_errors
        global_tot["bad_values"]    += bad_values
        global_tot["candidates"]    += total_cands

    pbar.close()

    # 输出
    out_dir = out_root
    pd.DataFrame(rows_split).to_csv(out_dir / "split_summary.csv", index=False)
    pd.DataFrame(rows_diag).to_csv(out_dir / "diag_summary.csv", index=False)
    if parse_err_rows:
        pd.DataFrame(parse_err_rows).to_csv(out_dir / "parse_errors_sample.csv", index=False)

    # 每 split 的 balance_ok 汇总
    df_split = pd.DataFrame(rows_split)
    split_ok = {}
    for (proto, sid), g in df_split.groupby(["protocol","split_id"]):
        split_ok[(proto,sid)] = bool((g["balance_ok"]).all())

    post = {
        "global": {
            **global_tot,
            "balance_ok_all_splits": all(split_ok.values()) if split_ok else True
        },
        "splits": [
            {"protocol": proto, "split_id": int(sid), "balance_ok_all_subsets": ok}
            for (proto, sid), ok in sorted(split_ok.items())
        ]
    }
    with open(out_dir / "s5_postcheck.json", "w", encoding="utf-8") as f:
        json.dump(post, f, ensure_ascii=False, indent=2)

    # 控制台简报
    print(f"[S5 Postcheck] splits={global_tot['splits']} subsets={global_tot['subsets']}")
    print(f"  expected={global_tot['expected']} | written_files={global_tot['written_files']} | no_cand={global_tot['no_cand']} | failed={global_tot['failed']}")
    print(f"  jsonl_lines={global_tot['jsonl_lines']} | candidates={global_tot['candidates']}")
    print(f"  parse_errors={global_tot['parse_errors']} | bad_values={global_tot['bad_values']}")
    print(f"  balance_ok_all_splits={post['global']['balance_ok_all_splits']}")

if __name__ == "__main__":
    main()
