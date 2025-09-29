# -*- coding: utf-8 -*-
"""
s6_core.py
日志、压力集生成、候选提取、评测、通用工具。读取 S5 参数映射。
改动要点：
- 仅使用 logging_config/s6.log_file 进行日志初始化。
- eval 阶段默认资源来自 s6.resources（由调用方传入或用默认），不再读取 s5.resources。
- 入口圆心提取 _entry_center_from_shape/_entry_center_from_face。
- match_recall 支持 require_center 开关。
"""
import os, sys, json, math, time, shutil, glob, hashlib, statistics, random, contextlib
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Tuple, Iterable
import numpy as np
import logging
from tqdm import tqdm

# ---------- 静默 C/C++ 层 ----------
@contextlib.contextmanager
def silence_cpp_stdio():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)

# ---------- 日志 ----------
def setup_logging_from_config(config_path: str):
    with open(config_path, 'r', encoding="utf-8") as f:
        cfg = json.load(f)

    # 以 logging_config 为主，s6.log_file 覆盖其 log_file；若均无，则落到 s6.out_root/pipeline.log，再退到 ./pipeline.log
    log_cfg = dict(cfg.get("logging_config", {}))
    s6_override = (cfg.get("s6") or {}).get("log_file")
    if s6_override:
        log_cfg["log_file"] = s6_override
    if not log_cfg.get("log_file"):
        out_root = (cfg.get("s6") or {}).get("out_root")
        if out_root:
            log_cfg["log_file"] = os.path.join(out_root, "pipeline.log")

    log_file = log_cfg.get("log_file", "./pipeline.log")
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    console_level = getattr(logging, str(log_cfg.get("console_level", "WARNING")).upper(), logging.WARNING)

    # 确保目录与文件存在，避免 FileHandler 报错
    dirpath = os.path.dirname(log_file) or "."
    os.makedirs(dirpath, exist_ok=True)
    try:
        open(log_file, "a", encoding="utf-8").close()
    except Exception:
        # 如果因编码或权限问题失败，回退到当前目录
        log_file = "./pipeline.log"
        os.makedirs(".", exist_ok=True)
        open(log_file, "a", encoding="utf-8").close()

    # 初始化 root logger
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(
        "logging initialized. file=%s level=%s console=%s",
        log_file, logging.getLevelName(level), logging.getLevelName(console_level)
    )
    return cfg


# ---------- 数据类 ----------
@dataclass
class HoleGT:
    kind: str
    D: float
    H: float
    alpha_deg: float
    axis: Tuple[float,float,float]
    base: Tuple[float,float,float]

@dataclass
class SynthItem:
    step_path: str
    holes: List[HoleGT]

@dataclass
class PredHole:
    D: float; H: float; alpha_deg: float
    axis: Tuple[float,float,float]
    base: Tuple[float,float,float]
    q: float

@dataclass
class S5Params:
    cyl_len_min: float = 0.5                 # s5.geometry.hole_min_length_mm
    axis_coax_tol_deg: float = 3.0           # from s5.geometry.coaxial_dot_min
    missing_ring_fallback: bool = True       # s5.geometry.allow_missing_entry_ring
    entrance_roundness_tol: float = 0.02
    dihedral_var_max: float = 8.0
    cone_angle_tol_deg: float = 3.0
    circle_fit_ransac_it: int = 300
    merge_gap_tol: float = 0.15
    max_candidates_per_part: int = 128

# ---------- OCCT 基础 ----------
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Trsf
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Circle

# ---------- STEP I/O ----------
def _write_step(solid, out_path: str):
    writer = STEPControl_Writer()
    with silence_cpp_stdio():
        writer.Transfer(solid, STEPControl_AsIs)
        status = writer.Write(out_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP: {out_path}")

def load_step_shape_safe(step_path: str):
    p = os.path.expanduser(step_path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    rdr = STEPControl_Reader()
    with silence_cpp_stdio():
        st = rdr.ReadFile(p)
        if st != IFSelect_RetDone:
            raise RuntimeError(f"read step failed: {step_path}")
        nb = rdr.NbRootsForTransfer()
        if nb == 0:
            raise RuntimeError(f"no roots to transfer: {step_path}")
        builder = BRep_Builder()
        comp = TopoDS_Compound(); builder.MakeCompound(comp)
        for i in range(1, nb+1):
            rdr.TransferRoot(i)
            s = rdr.Shape(i)
            try:
                if hasattr(s, "IsNull") and not s.IsNull():
                    builder.Add(comp, s)
            except Exception:
                continue
    if comp.IsNull():
        raise RuntimeError(f"compound null: {step_path}")
    return comp

# ---------- 遍历 ----------
def face_iter_safe(shape) -> Iterable:
    if shape is None:
        return
    try:
        if hasattr(shape, "IsNull") and shape.IsNull():
            return
    except Exception:
        return
    yielded = False
    try:
        exp_solid = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp_solid.More():
            solid = exp_solid.Current()
            exp_face = TopExp_Explorer(solid, TopAbs_FACE)
            while exp_face.More():
                try:
                    f = topods.Face(exp_face.Current())
                    if not f.IsNull():
                        yielded = True
                        yield f
                except Exception:
                    pass
                exp_face.Next()
            exp_solid.Next()
    except Exception as e:
        logging.warning("face_iter_safe solid path err: %s", e)
    if not yielded:
        try:
            exp_face = TopExp_Explorer(shape, TopAbs_FACE)
            while exp_face.More():
                try:
                    f = topods.Face(exp_face.Current())
                    if not f.IsNull():
                        yield f
                except Exception:
                    pass
                exp_face.Next()
        except Exception as e:
            logging.warning("face_iter_safe face path err: %s", e)

# ---------- 压力集合成 ----------
def _mk_box(w=50, d=50, h=15):
    return BRepPrimAPI_MakeBox(w, d, h).Shape()

def _add_hole(box, D, H, theta_deg, with_csk=False):
    from OCC.Core.gp import gp_Ax1
    theta = math.radians(theta_deg)
    ax = gp_Ax2(gp_Pnt(25, 25, 0), gp_Dir(0, 0, 1))
    cyl = BRepPrimAPI_MakeCylinder(ax, D / 2.0, H).Shape()
    rot_axis = gp_Ax1(gp_Pnt(25, 25, 0), gp_Dir(1, 0, 0))
    tr = gp_Trsf(); tr.SetRotation(rot_axis, theta)
    cyl = BRepBuilderAPI_Transform(cyl, tr, True).Shape()
    cut1 = BRepAlgoAPI_Cut(box, cyl).Shape()
    gt_axis = (0.0, -math.sin(theta), math.cos(theta))
    gt = HoleGT(kind="cyl", D=D, H=H, alpha_deg=90.0, axis=gt_axis, base=(25,25,0))
    if with_csk:
        cone = BRepPrimAPI_MakeCone(ax, D, D * 1.8, D * 0.9).Shape()
        cone = BRepBuilderAPI_Transform(cone, tr, True).Shape()
        cut2 = BRepAlgoAPI_Cut(cut1, cone).Shape()
        gt.kind = "cyl+cone"; box = cut2
    else:
        box = cut1
    return box, gt

def generate_stress_set(out_dir: str, total: int, seed: int=0) -> List[SynthItem]:
    os.makedirs(out_dir, exist_ok=True)
    rnd = random.Random(seed)
    items: List[SynthItem] = []
    thetas = [0, 15, 30, 45, 60]
    for i in tqdm(range(total), desc="gen_stress", ncols=100):
        box = _mk_box()
        holes: List[HoleGT] = []
        D = 1.5 * 2**rnd.uniform(0,5)
        H = D * rnd.uniform(0.5, 6.0)
        theta = rnd.choice(thetas)
        with_csk = rnd.random() < 0.5
        box, gt = _add_hole(box, D, H, theta, with_csk=with_csk)
        holes.append(gt)
        sp = os.path.join(out_dir, f"stress_{i:05d}.step")
        _write_step(box, sp)
        items.append(SynthItem(step_path=sp, holes=holes))
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump([{"step":it.step_path, "holes":[asdict(h) for h in it.holes]} for it in items], f, indent=2)
    return items

# ---------- 入口圆心 ----------
def _entry_center_from_face(face) -> Tuple[float,float,float] | None:
    centers = []
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        try:
            e = topods.Edge(exp.Current())
            c = BRepAdaptor_Curve(e)
            if c.GetType() == GeomAbs_Circle:
                circ = c.Circle()
                loc = circ.Location()
                centers.append((loc.X(), loc.Y(), loc.Z()))
        except Exception:
            pass
        exp.Next()
    if centers:
        cx = sum(x for x,_,_ in centers)/len(centers)
        cy = sum(y for _,y,_ in centers)/len(centers)
        cz = sum(z for _,_,z in centers)/len(centers)
        return (cx,cy,cz)
    return None

def _entry_center_from_shape(shape) -> Tuple[float,float,float]:
    centers = []
    for f in face_iter_safe(shape):
        c = _entry_center_from_face(f)
        if c is not None:
            centers.append(c)
    if centers:
        cx = sum(x for x,_,_ in centers)/len(centers)
        cy = sum(y for _,y,_ in centers)/len(centers)
        cz = sum(z for _,_,z in centers)/len(centers)
        return (cx,cy,cz)
    return (0.0,0.0,0.0)

# ---------- 候选 ----------
def _axis_from_surface(surf):
    if surf.GetType() == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        ax = cyl.Axis().Direction()
        return np.array([ax.X(), ax.Y(), ax.Z()]), cyl.Radius()
    elif surf.GetType() == GeomAbs_Cone:
        cone = surf.Cone()
        ax = cone.Axis().Direction()
        semi = math.degrees(cone.SemiAngle())
        return np.array([ax.X(), ax.Y(), ax.Z()]), semi
    return None, None

def find_candidates(step_path: str, params: S5Params) -> List[PredHole]:
    shape = load_step_shape_safe(step_path)
    cyls = []; cones = []
    for f in face_iter_safe(shape):
        try:
            s = BRepAdaptor_Surface(f)
        except Exception:
            continue
        t = s.GetType()
        if t == GeomAbs_Cylinder:
            ax, r = _axis_from_surface(s)
            if ax is None or r is None: 
                continue
            try:
                v0, v1 = s.FirstVParameter(), s.LastVParameter()
                H = abs(float(v1) - float(v0))
            except Exception:
                H = 0.0
            if H < params.cyl_len_min:
                continue
            n = np.linalg.norm(ax)
            if not np.isfinite(n) or n <= 0:
                continue
            cyls.append((ax / n, float(r), H, f))
        elif t == GeomAbs_Cone:
            ax, semi = _axis_from_surface(s)
            if ax is None or semi is None:
                continue
            n = np.linalg.norm(ax)
            if not np.isfinite(n) or n <= 0:
                continue
            cones.append((ax / n, float(semi), f))

    preds: List[PredHole] = []
    if cyls:
        axes = np.array([c[0] for c in cyls], dtype=float)
        ref = axes[0]
        dots = axes @ ref
        aligned = axes.copy()
        aligned[dots < 0] *= -1.0
        mean_ax = aligned.mean(axis=0)
        norm = np.linalg.norm(mean_ax)
        if not np.isfinite(norm) or norm <= 1e-8:
            longest_idx = int(np.argmax([c[2] for c in cyls]))
            mean_ax = cyls[longest_idx][0]
            norm = np.linalg.norm(mean_ax)
        ax_unit = (mean_ax / norm) if norm > 0 else ref

        D = float(np.median([2.0 * c[1] for c in cyls]))
        H = float(np.sum([c[2] for c in cyls]))
        alpha = 90.0
        if cones:
            alpha = 90.0  # 占位

        base = _entry_center_from_shape(shape)
        preds.append(
            PredHole(
                D=D, H=H, alpha_deg=alpha,
                axis=tuple(ax_unit.tolist()),
                base=base, q=0.8
            )
        )

    if len(preds) > params.max_candidates_per_part:
        preds = preds[:params.max_candidates_per_part]
    return preds

def write_candidates_jsonl(out_file: str, preds: List[PredHole]):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for i,p in enumerate(preds):
            rec = {
                "hole_id": i,
                "D0": float(p.D), "H0": float(p.H), "alpha0": float(p.alpha_deg),
                "axis": p.axis, "base": p.base, "q": float(p.q),
                "primitives": ["cyl","cone?"]
            }
            f.write(json.dumps(rec)+"\n")

# ---------- 评测（召回匹配） ----------
def _angle_deg(u, v):
    u = np.array(u, dtype=float); v = np.array(v, dtype=float)
    if np.linalg.norm(u)==0 or np.linalg.norm(v)==0: return 180.0
    u = u/np.linalg.norm(u); v = v/np.linalg.norm(v)
    cosv = float(np.clip(u@v, -1, 1))
    return math.degrees(math.acos(cosv))

def _iou_axis(H_true, H_pred):
    a = [0, max(0.0,float(H_true))]; b = [0, max(0.0,float(H_pred))]
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter/union if union>0 else 0.0

def match_recall(
    gt_list, pred_list,
    axis_tol_deg=5.0, axis_tol_deg_hi=8.0,
    entrance_center_tol_frac=0.3, entrance_center_tol_mm=0.2,
    iou_axis_min=0.7, require_center=True
):
    hit = 0
    for g in gt_list:
        matched = False
        for p in pred_list:
            ang = _angle_deg(g.axis, p.axis)
            tilt = math.degrees(math.acos(max(-1.0,min(1.0,g.axis[2])))) if len(g.axis)>=3 else 0.0
            ang_tol = axis_tol_deg if abs(tilt)<=45 else axis_tol_deg_hi
            if ang > ang_tol: continue
            if require_center:
                cen_tol = max(entrance_center_tol_frac * g.D, entrance_center_tol_mm)
                cen_dist = np.linalg.norm(np.array(g.base) - np.array(p.base))
                if cen_dist > cen_tol: continue
            iou = _iou_axis(g.H, p.H)
            if iou < iou_axis_min: continue
            matched = True
            break
        if matched:
            hit += 1
    recall = hit / max(1, len(gt_list))
    return recall

# ---------- 通用 ----------
def hash_path16(p: str) -> str:
    return hashlib.md5(p.encode("utf-8")).hexdigest()[:16]

def _deg_from_coax_dot(dot_min: float) -> float:
    dot_min = max(-1.0, min(1.0, float(dot_min)))
    return math.degrees(math.acos(dot_min))

def load_s5_params(cfg) -> S5Params:
    s5 = cfg.get("s5", {})
    geom = s5.get("geometry", {})
    cyl_len_min = float(geom.get("hole_min_length_mm", 0.5))
    axis_deg = _deg_from_coax_dot(float(geom.get("coaxial_dot_min", 0.995)))
    allow_missing = bool(geom.get("allow_missing_entry_ring", True))
    max_cand = int(geom.get("max_candidates_per_part", 128))
    p = S5Params(
        cyl_len_min=cyl_len_min,
        axis_coax_tol_deg=axis_deg,
        missing_ring_fallback=allow_missing,
        max_candidates_per_part=max_cand
    )
    return p

def save_params(path: str, params: S5Params):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params.__dict__, f, indent=2)

def backup_file(src: str, dst: str):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        logging.info("backup s5_params: %s -> %s", src, dst)

def read_lines(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def iter_protocol_dirs(s4_root: str, apply_protocols: list):
    if apply_protocols:
        for proto in apply_protocols:
            d = os.path.join(s4_root, proto)
            if os.path.isdir(d):
                yield proto, d
    else:
        if not os.path.isdir(s4_root): return
        for name in os.listdir(s4_root):
            d = os.path.join(s4_root, name)
            if os.path.isdir(d):
                yield name, d

def iter_split_dirs(s4_root: str, protocol: str):
    base = os.path.join(s4_root, protocol)
    if not os.path.isdir(base): return
    for split_dir in sorted(glob.glob(os.path.join(base, "split_*"))):
        yield split_dir
