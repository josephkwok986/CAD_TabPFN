#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 train/calib/test 之间是否存在 family/content_hash/duplicate_canonical 泄漏。
适配目录结构：
/workspace/Gjj Local/data/CAD/step_out/s4_out
├── lodo
│   ├── split_0/{train.txt,calib.txt,test.txt}
│   ├── split_1/{...}
│   ├── split_2/{...}
│   └── split_report.md
└── temporal
    ├── split_0/{train.txt,calib.txt,test.txt}
    └── split_report.md
"""

import os
import sys
import glob
import pandas as pd

# ========= 配置区（按需修改） =========
PI = "/workspace/Gjj Local/data/CAD/step_out/s2_out/part_index.for_split.csv"
S4_ROOT = "/workspace/Gjj Local/data/CAD/step_out/s4_out"
DOMAIN_COL = "source_dataset"          # 你的域列名；若没有可改成 "domain" 或其他
PART_COL = "part_id"
FAMILY_COL = "family_id"
CH_COL = "content_hash"
CAN_COL = "duplicate_canonical"
# ====================================

def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def require(p):
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p

def _non_null_set(iterable):
    out = set()
    for v in iterable:
        if pd.isna(v) or v is None:
            continue
        out.add(v)
    return out

def check_one_split(split_dir, maps, split_name_for_log):
    """对单个 split 目录做泄漏检查。"""
    id2dom, id2fam, id2ch, id2can = maps
    if not os.path.isdir(split_dir):
        return
    tr = set(read_ids(require(os.path.join(split_dir, "train.txt"))))
    ca = set(read_ids(require(os.path.join(split_dir, "calib.txt"))))
    te = set(read_ids(require(os.path.join(split_dir, "test.txt"))))

    # 0) 基本规模
    print(f"\n[{split_name_for_log}] train={len(tr)}, calib={len(ca)}, test={len(te)}")

    # 1) 基本不重叠
    assert tr.isdisjoint(ca),  "train ∩ calib 非空"
    assert tr.isdisjoint(te),  "train ∩ test 非空"
    assert ca.isdisjoint(te),  "calib ∩ test 非空"

    # 2) 若为 LODO：test 域必须单一且自洽（temporal 可跳过）
    if "lodo" in split_name_for_log.lower():
        test_domains = _non_null_set(id2dom.get(pid) for pid in te)
        print("  test 域集合:", test_domains)
        assert len(test_domains) == 1, "LODO：test 内出现多个 domain"
        bad_dom = [pid for pid in te if id2dom.get(pid) not in test_domains]
        assert len(bad_dom) == 0, "LODO：test 出现未知/缺失 domain"

    # 3) family/content_hash 零泄漏（train∪calib vs test）
    trc = tr | ca
    fam_trc = _non_null_set(id2fam.get(pid) for pid in trc)
    fam_te  = _non_null_set(id2fam.get(pid) for pid in te)
    ch_trc  = _non_null_set(id2ch.get(pid)  for pid in trc)
    ch_te   = _non_null_set(id2ch.get(pid)  for pid in te)

    leak_fam = fam_trc & fam_te
    leak_ch  = ch_trc  & ch_te
    print("  family 泄漏数:", len(leak_fam))
    print("  content_hash 泄漏数:", len(leak_ch))
    if len(leak_fam) > 0:
        print("  >>> family 泄漏明细（前若干条）")
    for fam in list(leak_fam)[:20]:
        trc_ids = [pid for pid in trc if id2fam.get(pid) == fam][:10]
        te_ids  = [pid for pid in te  if id2fam.get(pid) == fam][:10]
        print(f"    fam={fam} | train+calib例子={trc_ids} | test例子={te_ids}")
    assert len(leak_fam) == 0, "family 泄漏"
    assert len(leak_ch)  == 0, "content_hash 泄漏"

    # 4) duplicate_canonical 跨侧的“真泄漏”检查（同名且 content_hash 也重叠才算真泄漏）
    can_trc = _non_null_set(id2can.get(pid) for pid in trc)
    can_te  = _non_null_set(id2can.get(pid) for pid in te)
    cross_can = can_trc & can_te

    real_can_leak = []
    if cross_can:
        trc_by_can, te_by_can = {}, {}
        for pid in trc:
            c = id2can.get(pid)
            if c in cross_can:
                trc_by_can.setdefault(c, set()).add(id2ch.get(pid))
        for pid in te:
            c = id2can.get(pid)
            if c in cross_can:
                te_by_can.setdefault(c, set()).add(id2ch.get(pid))
        for c in sorted(cross_can):
            if (_non_null_set(trc_by_can.get(c, set()))
                & _non_null_set(te_by_can.get(c, set()))):
                real_can_leak.append(c)

    print("  duplicate_canonical 跨侧（同名）数:", len(cross_can),
          "/ 其中同名且同 content_hash（真泄漏）数:", len(real_can_leak))
    assert len(real_can_leak) == 0, "duplicate_canonical 真泄漏"

def main():
    # 读索引
    df = pd.read_csv(require(PI))
    need_cols = [PART_COL, FAMILY_COL, CH_COL, CAN_COL]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"索引缺少列：{c}")
    if DOMAIN_COL not in df.columns:
        print(f"[警告] 索引缺少 DOMAIN_COL='{DOMAIN_COL}'，仅跳过 LODO 的域一致性检查。")
        df[DOMAIN_COL] = None

    id2fam = dict(zip(df[PART_COL].astype(str), df[FAMILY_COL]))
    id2ch  = dict(zip(df[PART_COL].astype(str), df[CH_COL]))
    id2can = dict(zip(df[PART_COL].astype(str), df[CAN_COL]))
    id2dom = dict(zip(df[PART_COL].astype(str), df[DOMAIN_COL]))
    maps = (id2dom, id2fam, id2ch, id2can)

    ok_count = 0

    # —— LODO —— #
    lodo_root = os.path.join(S4_ROOT, "lodo")
    if os.path.isdir(lodo_root):
        split_dirs = sorted(glob.glob(os.path.join(lodo_root, "split_*")))
        if not split_dirs:
            print("[提示] lodo 目录存在但未发现 split_* 子目录。")
        for d in split_dirs:
            check_one_split(d, maps, f"LODO:{os.path.basename(d)}")
            ok_count += 1
    else:
        print("[提示] 未发现 lodo 目录，跳过 LODO 检查。")

    # —— Temporal —— #
    temp_root = os.path.join(S4_ROOT, "temporal")
    if os.path.isdir(temp_root):
        split_dirs = sorted(glob.glob(os.path.join(temp_root, "split_*")))
        if not split_dirs:
            print("[提示] temporal 目录存在但未发现 split_* 子目录。")
        for d in split_dirs:
            # temporal 不做域一致性断言，其它泄漏断言一致
            check_one_split(d, maps, f"TEMPORAL:{os.path.basename(d)}")
            ok_count += 1
    else:
        print("[提示] 未发现 temporal 目录，跳过 Temporal 检查。")

    if ok_count == 0:
        print("\n[错误] 没有检查到任何 split_* 目录，请确认 S4_ROOT 配置是否正确。")
        sys.exit(2)

    print("\n[OK] 全部 split_* 通过零泄漏校验。")

if __name__ == "__main__":
    main()
