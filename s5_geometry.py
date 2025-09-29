#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5: 高召回候选孔重建与粗估 —— 几何后端（pythonocc-core 版，修正版）
- 仅依赖 pythonocc-core 7.7.x 与 occt 7.7.x
- 修复：不再调用 BRepAdaptor_Surface.Axis()；轴向统一从 Cylinder().Axis() / Cone().Axis()
- 修复：去除对 gp_Trsf.IsIdentity() 的调用，统一应用位姿变换
- 可选 CuPy 加速半径方差计算
自测：直接运行本文件将生成一个带通孔与沉头的 STEP，随后提取候选并打印结果。
"""
from __future__ import annotations
import math
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp  # optional
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

# pythonocc-core
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Circle
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, topods
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform


@dataclass
class PrimitiveSeg:
    kind: str  # "cyl" | "cone" | "plane" ...
    params: Dict[str, float]  # e.g., {"radius": 2.0, "alpha_deg": 90.0}


@dataclass
class HoleCandidate:
    hole_id: str
    primitives: List[PrimitiveSeg]
    D0: Optional[float]      # 直径粗估 mm
    H0: Optional[float]      # 深度粗估 mm
    alpha0: Optional[float]  # 锥角粗估 deg（如存在）
    q: float                 # 质量分 0-1
    flags: Dict[str, Any]


def _dir_to_np(d) -> np.ndarray:
    return np.array([d.X(), d.Y(), d.Z()], dtype=np.float64)


def _axdir_from_cyl(surf: BRepAdaptor_Surface) -> np.ndarray:
    """从圆柱面适配器获取轴向方向向量。"""
    ax1 = surf.Cylinder().Axis()
    return _dir_to_np(ax1.Direction())


def _axdir_from_cone(surf: BRepAdaptor_Surface) -> np.ndarray:
    """从圆锥面适配器获取轴向方向向量。"""
    ax1 = surf.Cone().Axis()
    return _dir_to_np(ax1.Direction())


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _proj_extent_along(points: np.ndarray, axis_dir: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    axis = _safe_unit(axis_dir)
    t = points @ axis
    return float(t.max() - t.min())


class GeometryBackend:
    def __init__(self, cfg: Dict[str, Any], use_gpu: bool = False):
        self.cfg = cfg
        self.use_gpu = use_gpu and HAS_CUPY

    def load_shape(self, path: str) -> TopoDS_Shape:
        if path.lower().endswith((".step", ".stp")):
            reader = STEPControl_Reader()
            stat = reader.ReadFile(path)
            if stat != IFSelect_RetDone:
                raise RuntimeError(f"STEP read fail: {path}")
            reader.TransferRoots()
            return reader.OneShape()
        else:
            shape = TopoDS_Shape()
            builder = BRep_Builder()
            ok = breptools_Read(shape, path, builder)
            if not ok:
                raise RuntimeError(f"BRep read fail: {path}")
            return shape

    def _mesh_shape_if_needed(self, shape: TopoDS_Shape, deflection: float, angle: float):
        BRepMesh_IncrementalMesh(shape, deflection, True, angle, True)

    def _face_points(self, face: TopoDS_Face) -> np.ndarray:
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            return np.zeros((0, 3), dtype=np.float64)
        trsf: gp_Trsf = loc.Transformation()
        n = tri.NbNodes()
        arr = np.zeros((n, 3), dtype=np.float64)
        for i in range(1, n + 1):
            p = tri.Node(i)
            p = p.Transformed(trsf)
            arr[i - 1] = [p.X(), p.Y(), p.Z()]
        return arr

    def extract_candidates(self, shape: TopoDS_Shape, part_id: str) -> List[HoleCandidate]:
        if "geometry" not in self.cfg:
            raise KeyError("s5.geometry not found in config passed to GeometryBackend")
        geom_cfg = self.cfg["geometry"]
        defl = float(geom_cfg.get("mesh_deflection", 0.1))
        ang = float(geom_cfg.get("mesh_angle", 0.5))
        dot_min = float(geom_cfg.get("coaxial_dot_min", 0.995))
        min_r = float(geom_cfg.get("cylinder_min_radius_mm", 0.25))
        min_h = float(geom_cfg.get("hole_min_length_mm", 0.5))
        max_c = int(geom_cfg.get("max_candidates_per_part", 128))

        self._mesh_shape_if_needed(shape, defl, ang)

        cyl_faces: List[Tuple[TopoDS_Face, BRepAdaptor_Surface]] = []
        cone_faces: List[Tuple[TopoDS_Face, BRepAdaptor_Surface]] = []

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            f = topods.Face(exp.Current())
            try:
                surf = BRepAdaptor_Surface(f, True)
                st = surf.GetType()
                if st == GeomAbs_Cylinder:
                    r = surf.Cylinder().Radius()
                    if r >= min_r:
                        cyl_faces.append((f, surf))
                elif st == GeomAbs_Cone:
                    cone_faces.append((f, surf))
            except Exception:
                pass
            exp.Next()

        cands: List[HoleCandidate] = []
        for idx, (f_cyl, s_cyl) in enumerate(cyl_faces):
            try:
                axis_dir = _axdir_from_cyl(s_cyl)
            except Exception:
                continue
            r = float(s_cyl.Cylinder().Radius())
            pts = self._face_points(f_cyl)
            H = _proj_extent_along(pts, axis_dir)

            # 匹配同轴圆锥（取最大轴向点积）
            best_cone = None
            best_dot = 0.0
            axis_n = _safe_unit(axis_dir)
            for f_cone, s_cone in cone_faces:
                try:
                    cone_dir = _axdir_from_cone(s_cone)
                except Exception:
                    continue
                cone_n = _safe_unit(cone_dir)
                dot = float(abs(np.dot(axis_n, cone_n)))
                if dot > dot_min and dot > best_dot:
                    best_dot = dot
                    best_cone = (f_cone, s_cone)

            alpha0 = None
            prims = [PrimitiveSeg("cyl", {"radius": r})]
            if best_cone is not None:
                _, s_cone = best_cone
                semi = float(s_cone.Cone().SemiAngle())
                alpha0 = math.degrees(2.0 * semi)  # 近似沉头角
                prims.append(PrimitiveSeg("cone", {"semi_angle_deg": math.degrees(semi)}))

            # 入口圆边检测（存在任一圆边即可）
            has_entry = False
            eexp = TopExp_Explorer(f_cyl, TopAbs_EDGE)
            while eexp.More():
                try:
                    e = topods.Edge(eexp.Current())
                    curve = BRepAdaptor_Curve(e)
                    if curve.GetType() == GeomAbs_Circle:
                        has_entry = True
                        break
                except Exception:
                    pass
                eexp.Next()

            D0 = 2.0 * r
            H0 = float(H)

            q_parts = []
            # 圆度/径向稳定性
            if pts.size >= 3:
                axis = _safe_unit(axis_dir)
                use_gpu = self.use_gpu  # 强制 GPU，无视点数
                if use_gpu:
                    xp = cp
                    P = xp.asarray(pts)
                    ax = xp.asarray(axis)
                    rvec = P - (P @ ax)[:, None] * ax
                    rr = xp.sqrt((rvec ** 2).sum(axis=1))
                    rmean = float(rr.mean().get())
                    rstd = float(rr.std().get())
                else:
                    P = pts
                    ax = axis
                    rvec = P - (P @ ax)[:, None] * ax
                    rr = np.sqrt((rvec ** 2).sum(axis=1))
                    rmean = float(rr.mean())
                    rstd = float(rr.std())
                q_round = 1.0 - float(min(1.0, rstd / (rmean + 1e-6)))
                q_parts.append(q_round)

            # 共轴性与入口环
            q_coax = float(min(1.0, max(0.0, best_dot if best_cone else 1.0)))
            q_parts.append(q_coax)
            q_entry = 1.0 if geom_cfg.get("allow_missing_entry_ring", True) else (1.0 if has_entry else 0.3)
            if not geom_cfg.get("allow_missing_entry_ring", True) and not has_entry:
                q_entry = 0.3
            elif has_entry:
                q_entry = 1.0
            else:
                q_entry = 0.7
            q_parts.append(q_entry)

            q = float(max(0.0, min(1.0, sum(q_parts) / len(q_parts)))) if q_parts else 0.5

            # 基本阈值过滤
            if H0 < min_h or D0 < 2 * min_r:
                continue

            cand = HoleCandidate(
                hole_id=f"{part_id}::h{idx:04d}",
                primitives=prims,
                D0=D0, H0=H0, alpha0=alpha0,
                q=q,
                flags={"has_entry_ring": has_entry, "coax_dot": best_dot if best_cone else 1.0}
            )
            cands.append(cand)
            if len(cands) >= max_c:
                break

        return cands
    
    def quick_probe(self, shape) -> Dict[str, int]:
        from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_BSplineSurface, GeomAbs_BezierSurface, GeomAbs_SurfaceOfRevolution
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
        from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Circle
        from OCC.Core.TopoDS import topods

        hist = dict(
            faces_total=0, cyl_faces=0, cone_faces=0,
            plane_faces=0, nurbs_faces=0, bezier_faces=0, rev_faces=0,
            circ_edges=0
        )
        # 面类型计数
        fexp = TopExp_Explorer(shape, TopAbs_FACE)
        while fexp.More():
            f = topods.Face(fexp.Current())
            try:
                st = BRepAdaptor_Surface(f, True).GetType()
                hist["faces_total"] += 1
                if st == GeomAbs_Cylinder: hist["cyl_faces"] += 1
                elif st == GeomAbs_Cone: hist["cone_faces"] += 1
                elif st == GeomAbs_Plane: hist["plane_faces"] += 1
                elif st == GeomAbs_BSplineSurface: hist["nurbs_faces"] += 1
                elif st == GeomAbs_BezierSurface: hist["bezier_faces"] += 1
                elif st == GeomAbs_SurfaceOfRevolution: hist["rev_faces"] += 1
            except Exception:
                pass
            fexp.Next()
        # 圆边计数
        fexp = TopExp_Explorer(shape, TopAbs_FACE)
        while fexp.More():
            f = topods.Face(fexp.Current())
            eexp = TopExp_Explorer(f, TopAbs_EDGE)
            while eexp.More():
                try:
                    e = topods.Edge(eexp.Current())
                    if BRepAdaptor_Curve(e).GetType() == GeomAbs_Circle:
                        hist["circ_edges"] += 1
                except Exception:
                    pass
                eexp.Next()
            fexp.Next()
        return hist



def make_backend(cfg: Dict[str, Any], use_gpu: bool) -> GeometryBackend:
    return GeometryBackend(cfg, use_gpu)


# -------------------- 自测 --------------------
if __name__ == "__main__":
    import tempfile, os
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # 生成：盒子(20×20×10) + 通孔(Ø4, 深10) + 顶部约45°沉头
    box = BRepPrimAPI_MakeBox(20, 20, 10).Shape()
    cyl = BRepPrimAPI_MakeCylinder(2.0, 12).Shape()  # 半径2，略长以保证穿透
    tr1 = gp_Trsf(); tr1.SetTranslation(gp_Vec(10, 10, -1))
    cyl = BRepBuilderAPI_Transform(cyl, tr1, True).Shape()

    cone = BRepPrimAPI_MakeCone(3.0, 0.0, 2.0).Shape()
    tr2 = gp_Trsf(); tr2.SetTranslation(gp_Vec(10, 10, 8))
    cone = BRepBuilderAPI_Transform(cone, tr2, True).Shape()

    cut1 = BRepAlgoAPI_Cut(box, cyl).Shape()
    cut2 = BRepAlgoAPI_Cut(cut1, cone).Shape()

    tmpdir = tempfile.mkdtemp(prefix="s5_occ_")
    stepp = os.path.join(tmpdir, "demo.step")
    writer = STEPControl_Writer()
    writer.Transfer(cut2, STEPControl_AsIs)
    writer.Write(stepp)

    cfg = {
        "geometry": {
            "mesh_deflection": 0.1,
            "mesh_angle": 0.5,
            "cylinder_min_radius_mm": 0.25,
            "hole_min_length_mm": 0.5,
            "coaxial_dot_min": 0.995,
            "allow_missing_entry_ring": True,
            "max_candidates_per_part": 32
        }
    }
    be = GeometryBackend(cfg, use_gpu=False)
    shape = be.load_shape(stepp)
    cands = be.extract_candidates(shape, "DEMO")
    print("Candidates:", json.dumps([asdict(c) for c in cands], ensure_ascii=False, indent=2))
    print("STEP:", stepp)
