"""
Cellpose WSI 后处理 — 模仿 Cerberus 核后处理 (分开处理再整合)

设计思路 (对齐 Cerberus infer/wsi.py):
  1. 推理阶段 (Inference / Raw 缓存):
     - 以 chunk_shape=15000 大瓦片为单位滑动 WSI, 每个 chunk 内再切 patch_input/patch_output (默认 448/144 已对应 Cerberus) 或直接用 cellpose bsize  tiling
     - 将每个 patch 的预测 (flows dP + cellprob) 通过 merge_prediction 平均累加到全局 memmap (raw + count), 最终得到全 WSI 的 flow 场缓存
     - 等价于 Cerberus 的 head_caches / merge_prediction 部分

  2. 后处理阶段 (Post-processing / Tile Stitching):
     - 以 tile_shape=4096 小瓦片为单位重新切分全局 flow 场, 按 Cerberus 的 4 类瓦片生成:
         mode 0: 常规网格  (无重叠)
         mode 1: 垂直条带  (夹在两个常规瓦片之间, 宽度<高度)
         mode 2: 水平条带
         mode 3: 十字交叉  (四瓦片交点)
       每类瓦片附带 tile_flag=[top,bottom,left,right] 标识哪些边是“模糊边界”需要剔除
     - 每个小瓦片: 从全局 flow memmap 裁剪 -> dynamics.compute_masks 得到 inst_map -> 提取 inst_dict (box/contour/centroid)
     - 模糊边界去重 (完全复刻 Cerberus _process_tile_predictions 逻辑, 用 shapely STRtree):
         - mode 0/3: 删除 bbox 完全落在 margin(ambiguous_size, 默认64) 区域内的实例 (predicate="contains")
         - mode 1/2: 删除触碰 margin 或边界的实例
         - mode 3 额外: 从已累积的全局字典中删除与 margin_lines 相交的旧实例 (防止跨瓦片重复)
     - 将剩余实例坐标平移回 WSI 空间, 赋新 uuid, 聚合到全局字典
     - 最终 joblib.dump 为 .dat, 结构与 Cerberus 一致: {inst_uuid: {box, contour, centroid, type...}, proc_resolution, base_resolution ...}

  3. 轻量模式 (direct_tile_eval):
     - 若不想缓存巨大 flow 场 (80k×80k×3 float32 ~ 70GB), 可跳过第1阶段, 直接对每个 4096 瓦片读图 -> model.eval -> 去重
     - 同样复用 margin 去重逻辑, 适合显存/磁盘有限场景, 效果与 flow 缓存模式几乎一致

使用示例见文末及配套 run_wsi_cellpose.py

作者: 仿 Cerberus infer/wsi.py 实现, 适配 Cellpose4 (cpsam/cpdino)
"""

import os
import uuid
import time
import pathlib
import logging
from collections import OrderedDict

import cv2
import numpy as np
import torch
# torch 2.0 兼容: cellpose 源码用 torch.load(mmap=True) 在旧版 torch 会崩
try:
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw.pop("mmap", None)
        return _orig_load(*a, **kw)
    torch.load = _patched_load
except Exception:
    pass
# 静默 torch sparse 警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import tqdm
import joblib
from scipy.ndimage import measurements

from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree

# cellpose 内部
from cellpose import dynamics, utils, transforms
from cellpose.models import CellposeModel

# tiatoolbox 复用 (与 Cerberus 完全一致)
from tiatoolbox.models import IOSegmentorConfig, NucleusInstanceSegmentor
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader

def _get_bbox_fallback(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax+1, cmin, cmax+1

# ===================== 通用工具 =====================

def _rtree_query_indices(rtree, geometries, bounds, predicate=None):
    """STRtree.query 兼容 shapely 1.x/2.0, 返回整数索引"""
    if predicate is None:
        res = rtree.query(bounds)
    else:
        try:
            res = rtree.query(bounds, predicate=predicate)
        except TypeError:
            res = rtree.query(bounds)
            pred = getattr(bounds, predicate)
            res = [geo for geo in res if pred(geo)]
    if len(res) == 0:
        return []
    if isinstance(res[0], (int, np.integer)):
        return [int(i) for i in res]
    id_lookup = {id(geo): i for i, geo in enumerate(geometries)}
    indices = []
    for geo in res:
        idx = id_lookup.get(id(geo))
        if idx is None:
            idx = next((i for i, g in enumerate(geometries) if g.equals(geo)), None)
        indices.append(idx)
    return indices


def get_inst_info_dict(inst_map, min_size=15):
    """
    将 cellpose 语义标签图转实例字典 (仿 Cerberus loader/postproc.py get_inst_info_dict)
    返回: {inst_id: {box: 2x2 np, centroid: 2 np, contour: Nx2 np}}
    box 格式 [[rmin,cmin],[rmax,cmax]]  (YX), contour/centroid 为 XY
    """
    inst_info_dict = {}
    inst_ids = np.unique(inst_map)[1:]
    for inst_id in inst_ids:
        single = inst_map == inst_id
        # bounding box
        try:
            from misc.utils import get_bounding_box
            rmin, rmax, cmin, cmax = get_bounding_box(single)
        except Exception:
            rmin, rmax, cmin, cmax = _get_bbox_fallback(single)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        crop = single[rmin:rmax, cmin:cmax].astype(np.uint8)
        if crop.size == 0:
            continue
        m = cv2.moments(crop)
        if m["m00"] == 0:
            continue
        contours = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0][0]
        cnt = np.squeeze(cnt.astype("int32"))
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            continue
        # 转回全局坐标: cnt 是 (x,y) -> (cmin + x, rmin + y) 但 np 顺序是 x=col, y=row
        cnt[:, 0] += inst_bbox[0][1]
        cnt[:, 1] += inst_bbox[0][0]
        centroid = np.array([m["m10"]/m["m00"] + cmin, m["m01"]/m["m00"] + rmin])  # XY
        # centroid 转 XY: (cmin + ..., rmin + ...) -> (x,y)
        inst_info_dict[int(inst_id)] = {
            "box": inst_bbox,
            "centroid": centroid,
            "contour": cnt,
        }
    return inst_info_dict


def _process_cellpose_tile(
    ioconfig,
    tile_bounds,
    tile_flag,
    tile_mode,
    ref_inst_dict,
    cache_flow_path,   # memmap: [H,W,3] -> dP(Y,X)+cellprob 或 3通道
    cache_prob_path,   # 可选: 单独 cellprob
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    niter=200,
    device=torch.device("cpu"),
):
    """
    模仿 Cerberus _process_tile_predictions, 但输入是 cellpose flows
    tile_bounds: [x0,y0,x1,y1] in WSI coords (XY)
    """
    tile_tl = tile_bounds[:2]
    tile_br = tile_bounds[2:]
    tile_shape = tile_br - tile_tl  # W,H

    # --- 读取 flow 裁剪 ---
    # cache_flow: 期望形状 [H,W,2] (dP y,x)  + cache_prob: [H,W] 或合并为 [H,W,3]
    if os.path.exists(cache_flow_path):
        ptr = np.load(cache_flow_path, mmap_mode="r")
        # 兼容两种存储: [H,W,3] 或 [H,W,2] + 单独 prob
        crop = ptr[tile_tl[1]:tile_br[1], tile_tl[0]:tile_br[0]]
        crop = np.array(crop)  # to RAM
        if crop.ndim == 3 and crop.shape[2] == 3:
            dP = crop[..., :2].transpose(2,0,1)  # -> 2xHxW
            cellprob = crop[..., 2]
        elif crop.ndim == 3 and crop.shape[2] == 2:
            dP = crop.transpose(2,0,1)
            if cache_prob_path and os.path.exists(cache_prob_path):
                prob_ptr = np.load(cache_prob_path, mmap_mode="r")
                cellprob = np.array(prob_ptr[tile_tl[1]:tile_br[1], tile_tl[0]:tile_br[0]])
            else:
                cellprob = np.zeros(dP.shape[1:], dtype=np.float32)
        else:
            raise ValueError(f"unexpected crop shape {crop.shape}")
    else:
        raise FileNotFoundError(cache_flow_path)

    # --- cellpose dynamics -> inst_map ---
    # dynamics.compute_masks 要求 dP: 2xHxW, cellprob: HxW
    inst_map = dynamics.compute_masks(
        dP, cellprob,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        device=device,
    )
    if inst_map is None or inst_map.size == 0:
        return {}, []
    if np.max(inst_map) == 0:
        return {}, []

    inst_dict = get_inst_info_dict(inst_map, min_size=min_size)
    if len(inst_dict) == 0:
        return {}, []

    # --- Cerberus 模糊边界去重逻辑 (完全复刻) ---
    m = ioconfig.margin
    w, h = tile_shape  # tile_shape is [W,H] from tile_br-tile_tl
    inst_boxes = [v["box"] for v in inst_dict.values()]
    # box 是 [[rmin,cmin],[rmax,cmax]] YX -> 转 XY for shapely
    # shapely_box 要求 (minx, miny, maxx, maxy)
    xy_boxes = []
    for b in inst_boxes:
        rmin, cmin = b[0]
        rmax, cmax = b[1]
        xy_boxes.append([cmin, rmin, cmax, rmax])
    xy_boxes = np.array(xy_boxes)
    geometries = [shapely_box(*bounds) for bounds in xy_boxes]
    tile_rtree = STRtree(geometries)

    boundary_lines = [
        shapely_box(0, 0, w, 1),
        shapely_box(0, h - 1, w, h),
        shapely_box(0, 0, 1, h),
        shapely_box(w - 1, 0, w, h),
    ]
    margin_boxes = [
        shapely_box(0, 0, w, m),
        shapely_box(0, h - m, w, h),
        shapely_box(0, 0, m, h),
        shapely_box(w - m, 0, w, h),
    ]
    margin_lines = [
        [[m, m], [w - m, m]],
        [[m, h - m], [w - m, h - m]],
        [[m, m], [m, h - m]],
        [[w - m, m], [w - m, h - m]],
    ]
    margin_lines = np.array(margin_lines) + tile_tl[None, None]
    margin_lines = [shapely_box(*v.flatten().tolist()) for v in margin_lines]

    sel_indices = []
    if tile_mode in [0, 3]:
        sel_boxes = [box for idx, box in enumerate(margin_boxes) if tile_flag[idx] or tile_mode == 3]
        sel_indices = [idx for bounds in sel_boxes for idx in _rtree_query_indices(tile_rtree, geometries, bounds, predicate="contains")]
    elif tile_mode in [1, 2]:
        sel_boxes = [margin_boxes[idx] if flag else boundary_lines[idx] for idx, flag in enumerate(tile_flag)]
        sel_indices = [idx for bounds in sel_boxes for idx in _rtree_query_indices(tile_rtree, geometries, bounds)]
    else:
        raise ValueError(f"Unknown tile mode {tile_mode}")

    def retrieve_sel_uids(sel_indices, inst_dict):
        if len(sel_indices) == 0:
            return []
        uids = list(inst_dict.keys())
        return [uids[idx] for idx in sel_indices]

    remove_in_tile = set(retrieve_sel_uids(sel_indices, inst_dict))

    # cross tile 需删除全局中与 margin_lines 相交的旧实例
    remove_in_orig = []
    if tile_mode == 3 and len(ref_inst_dict) > 0:
        ref_boxes = []
        for v in ref_inst_dict.values():
            b = v["box"]
            ref_boxes.append([b[0][1], b[0][0], b[1][1], b[1][0]])  # cmin,rmin,cmax,rmax
        ref_boxes = np.array(ref_boxes)
        ref_geoms = [shapely_box(*bounds) for bounds in ref_boxes]
        ref_rtree = STRtree(ref_geoms)
        sel = [idx for bounds in margin_lines for idx in _rtree_query_indices(ref_rtree, ref_geoms, bounds)]
        remove_in_orig = retrieve_sel_uids(sel, ref_inst_dict)

    # 平移回 WSI 坐标并赋 uuid
    new_inst_dict = {}
    for uid, info in inst_dict.items():
        if uid in remove_in_tile:
            continue
        # box
        info["box"] = info["box"] + np.array([[tile_tl[1], tile_tl[0]], [tile_tl[1], tile_tl[0]]])  # tile_tl is XY
        # 实际 tile_tl XY -> box YX 偏移: [y,x]
        # info["box"] 已是 YX, tile_tl[1]=y, tile_tl[0]=x
        info["centroid"] = info["centroid"] + tile_tl  # XY
        info["contour"] = info["contour"] + tile_tl    # XY
        new_uuid = uuid.uuid4().hex
        new_inst_dict[new_uuid] = info

    return new_inst_dict, remove_in_orig


def _direct_tile_eval_process(
    ioconfig,
    tile_bounds,
    tile_flag,
    tile_mode,
    ref_inst_dict,
    wsi_reader,
    model,
    resolution,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    diameter=None,
    bsize=256,
    tile_overlap=0.1,
    device=torch.device("cpu"),
):
    """
    轻量模式: 直接对瓦片读图 + model.eval, 复用同样的 margin 去重逻辑
    适合无大 memmap 的场景
    """
    tile_tl = tile_bounds[:2].astype(int)
    tile_br = tile_bounds[2:].astype(int)
    w, h = (tile_br - tile_tl).tolist()
    # WSIReader.read_rect: location/size 默认是 baseline 坐标。
    # Cerberus 的 tile_bounds 是在 proc 分辨率 (0.5 mpp) 下的坐标，
    # 必须用 coord_space="resolution" 才能正确对齐；resolution 要传 float 而非 dict。
    # 兼容：resolution 可能是 {"resolution":0.5,"units":"mpp"} 或 float
    if isinstance(resolution, dict):
        res_val = float(resolution.get("resolution", 0.5))
        res_units = resolution.get("units", "mpp")
    else:
        res_val = float(resolution)
        res_units = "mpp"
    try:
        img = wsi_reader.read_rect(
            location=tuple(tile_tl.tolist()), size=(w, h),
            resolution=res_val, units=res_units, coord_space="resolution"
        )
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    except Exception as e:
        # fallback 1: 旧版 tiatoolbox 不支持 coord_space
        try:
            img = wsi_reader.read_rect(
                location=tuple(tile_tl.tolist()), size=(w, h),
                resolution=res_val, units=res_units
            )
            if img.shape[-1] == 4:
                img = img[..., :3]
        except Exception:
            # fallback 2: 按 baseline 读 (无 mpp 元数据的 TIFF)
            img = wsi_reader.read_rect(location=tuple(tile_tl.tolist()), size=(w, h))
            if img.shape[-1] == 4:
                img = img[..., :3]

    if img.size == 0:
        return {}, []

    # cellpose eval (内部自带 tiling with tile_overlap/bsize)
    masks, flows, styles = model.eval(
        img, diameter=diameter, flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold, min_size=min_size,
        tile_overlap=tile_overlap, bsize=bsize
    )
    # model.eval 返回 masks 2D
    if masks is None or masks.size == 0 or masks.max() == 0:
        return {}, []
    inst_dict = get_inst_info_dict(masks, min_size=min_size)
    if len(inst_dict) == 0:
        return {}, []

    # 去重逻辑 (同 _process_cellpose_tile 后半段)
    m = ioconfig.margin
    tile_shape = tile_br - tile_tl
    w, h = tile_shape
    inst_boxes = [v["box"] for v in inst_dict.values()]
    xy_boxes = np.array([[b[0][1], b[0][0], b[1][1], b[1][0]] for b in inst_boxes])
    geometries = [shapely_box(*b) for b in xy_boxes]
    tile_rtree = STRtree(geometries)

    boundary_lines = [shapely_box(0,0,w,1), shapely_box(0,h-1,w,h), shapely_box(0,0,1,h), shapely_box(w-1,0,w,h)]
    margin_boxes = [shapely_box(0,0,w,m), shapely_box(0,h-m,w,h), shapely_box(0,0,m,h), shapely_box(w-m,0,w,h)]
    margin_lines = np.array([[[m,m],[w-m,m]], [[m,h-m],[w-m,h-m]], [[m,m],[m,h-m]], [[w-m,m],[w-m,h-m]]]) + tile_tl[None,None]
    margin_lines = [shapely_box(*v.flatten().tolist()) for v in margin_lines]

    sel_indices = []
    if tile_mode in [0,3]:
        sel_boxes = [box for idx, box in enumerate(margin_boxes) if tile_flag[idx] or tile_mode==3]
        sel_indices = [idx for bounds in sel_boxes for idx in _rtree_query_indices(tile_rtree, geometries, bounds, predicate="contains")]
    elif tile_mode in [1,2]:
        sel_boxes = [margin_boxes[idx] if flag else boundary_lines[idx] for idx, flag in enumerate(tile_flag)]
        sel_indices = [idx for bounds in sel_boxes for idx in _rtree_query_indices(tile_rtree, geometries, bounds)]

    def retrieve(sel, d):
        if not sel: return []
        uids = list(d.keys())
        return [uids[i] for i in sel]

    remove_in_tile = set(retrieve(sel_indices, inst_dict))
    remove_in_orig = []
    if tile_mode == 3 and len(ref_inst_dict) > 0:
        ref_boxes = np.array([[v["box"][0][1], v["box"][0][0], v["box"][1][1], v["box"][1][0]] for v in ref_inst_dict.values()])
        ref_geoms = [shapely_box(*b) for b in ref_boxes]
        ref_rtree = STRtree(ref_geoms)
        sel = [idx for bounds in margin_lines for idx in _rtree_query_indices(ref_rtree, ref_geoms, bounds)]
        remove_in_orig = retrieve(sel, ref_inst_dict)

    new_inst_dict = {}
    for uid, info in inst_dict.items():
        if uid in remove_in_tile: continue
        info["box"] = info["box"] + np.array([[tile_tl[1], tile_tl[0]],[tile_tl[1], tile_tl[0]]])
        info["centroid"] = info["centroid"] + tile_tl
        info["contour"] = info["contour"] + tile_tl
        new_inst_dict[uuid.uuid4().hex] = info

    return new_inst_dict, remove_in_orig


# ===================== 对外主类 =====================

class CellposeWSI:
    """
    Cellpose WSI 推理器 (Cerberus 风格)
    两种模式:
      1. flow_memmap 模式: 先全 WSI 缓存 flows (大磁盘), 再小瓦片 dynamics 去重 (最还原 Cerberus)
      2. direct 模式: 直接小瓦片 model.eval + 去重 (默认, 更省磁盘)

    参数与 Cerberus 对齐:
      wsi_proc_mag=0.5  (mpp)
      chunk_shape=15000 (仅 flow_memmap 模式使用)
      tile_shape=4096
      ambiguous_size=64
      patch_input_shape=448, patch_output_shape=144 (仅 flow_memmap 模式, 用于 PatchExtractor)
    """
    def __init__(self, model: CellposeModel, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("cellpose_wsi")

    def _merge_predictions(self, canvas_shape, predictions, locations, save_path, cache_count_path=None):
        """复刻 Cerberus merge_prediction (平均重叠) — 用于 flow 缓存"""
        # lazy import tiatoolbox merge
        from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
        return NucleusInstanceSegmentor.merge_prediction(canvas_shape, predictions, locations, save_path, cache_count_path)

    def process_single_wsi_direct(self, wsi_path, output_path, mask_path=None,
                                   wsi_proc_mag=0.5, tile_shape=4096, ambiguous_size=64,
                                   flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
                                   diameter=None, bsize=256, tile_overlap=0.1,
                                   nr_post_proc_workers=0,
                                   save_viz_highres=False, viz_highres_tile=2048, viz_highres_mpp=None, viz_highres_max_tiles=16,
                                   save_viz_highres_full=False, viz_highres_full_max_mpix=120):
        """轻量 direct 模式 — 推荐"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        resolution = {"resolution": wsi_proc_mag, "units": "mpp"}
        wsi_reader = WSIReader.open(input_img=wsi_path)
        # 兼容无 mpp 元数据的普通 TIFF (如合成测试图)
        try:
            wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)  # XY
            wsi_proc_shape_yx = wsi_proc_shape[::-1]
            wsi_base_mag = wsi_reader.info.mpp
            if wsi_base_mag is not None:
                wsi_base_shape = wsi_reader.slide_dimensions(wsi_base_mag, "mpp")[::-1]
            else:
                wsi_base_shape = wsi_proc_shape_yx
        except Exception:
            # fallback: 用 baseline (level 0) 尺寸, 视为已是目标分辨率
            wsi_proc_shape = wsi_reader.slide_dimensions(0, "level")
            wsi_proc_shape_yx = wsi_proc_shape[::-1]
            wsi_base_mag = wsi_proc_mag
            wsi_base_shape = wsi_proc_shape_yx
            resolution = {"resolution": 0, "units": "level"}

        # tissue mask (可选)
        if mask_path and os.path.isfile(mask_path):
            wsi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            wsi_mask = (wsi_mask > 0).astype(np.uint8)
        else:
            wsi_mask = np.ones(wsi_proc_shape_yx, dtype=np.uint8)

        ioconfig_pp = IOSegmentorConfig(
            input_resolutions=[{"units":"mpp","resolution": wsi_proc_mag}],
            output_resolutions=[{"units":"mpp","resolution": wsi_proc_mag}],
            margin=ambiguous_size, tile_shape=[tile_shape, tile_shape],
            patch_input_shape=[448,448], patch_output_shape=[144,144],
            stride_shape=[144,144], save_resolution=resolution
        )
        # 获取 4 类瓦片 (grid, vertical, horizontal, cross)
        # 获取 4 类瓦片 (grid, vertical, horizontal, cross)
        # 直接复刻 tiatoolbox NucleusInstanceSegmentor._get_tile_info，避免版本差异
        def _get_tile_sets(shape_xy, cfg):
            shape_xy = np.array(shape_xy).tolist() if isinstance(shape_xy, np.ndarray) else list(shape_xy)
            # 优先尝试 tiatoolbox 原生实现（若可用）
            try:
                return NucleusInstanceSegmentor._get_tile_info(shape_xy, cfg)
            except Exception:
                pass
            try:
                return NucleusInstanceSegmentor._get_tile_info(np.array(shape_xy), cfg)
            except Exception:
                pass
            # 完全自包含的 fallback，实现与 tiatoolbox 1.4.0 源码 1:1 一致
            from collections import deque
            margin = np.array(cfg.margin)
            tile_shape = np.array(cfg.tile_shape)
            tile_shape = (np.floor(tile_shape / cfg.patch_output_shape) * cfg.patch_output_shape).astype(np.int32)
            image_shape = np.array(shape_xy)
            (_, tile_outputs) = PatchExtractor.get_coordinates(
                image_shape=image_shape,
                patch_input_shape=tile_shape,
                patch_output_shape=tile_shape,
                stride_shape=tile_shape,
            )
            boxes = tile_outputs
            if np.all(image_shape <= tile_shape):
                flag = np.zeros([boxes.shape[0], 4], dtype=np.int32)
                return [[boxes, flag]]
            def unset_removal_flag(boxes_, removal_flag_):
                w_, h_ = image_shape
                sel_boxes_ = [
                    shapely_box(0, 0, w_, 0),
                    shapely_box(0, h_, w_, h_),
                    shapely_box(0, 0, 0, h_),
                    shapely_box(w_, 0, w_, h_),
                ]
                geometries_ = [shapely_box(*b) for b in boxes_]
                rtree_ = STRtree(geometries_)
                for idx, sel_box in enumerate(sel_boxes_):
                    try:
                        sel_indices = list(rtree_.query(sel_box))
                        # shapely 2.0 返回 int 索引, 1.x 返回几何对象
                        if len(sel_indices)>0 and not isinstance(sel_indices[0], (int, np.integer)):
                            # 1.x 几何对象 -> 转索引
                            id_lookup = {id(g): i for i, g in enumerate(geometries_)}
                            sel_indices = [id_lookup.get(id(g), -1) for g in sel_indices]
                            sel_indices = [i for i in sel_indices if i>=0]
                    except Exception:
                        sel_indices = []
                    removal_flag_[sel_indices, idx] = 0
                return removal_flag_
            w, h = image_shape
            boxes_br = boxes[:, 2:]
            boxes_tr = np.dstack([boxes[:, 2], boxes[:, 1]])[0]
            boxes_bl = np.dstack([boxes[:, 0], boxes[:, 3]])[0]
            flag = np.ones([boxes.shape[0], 4], dtype=np.int32)
            flag = unset_removal_flag(boxes, flag)
            info = deque([[boxes, flag]])
            sel_indices = np.nonzero(flag[..., 3])[0]
            if len(sel_indices)>0:
                _boxes = np.concatenate([boxes_tr[sel_indices] - np.array([margin, 0])[None], boxes_br[sel_indices] + np.array([margin, 0])[None]], axis=-1)
                _flag = np.full([_boxes.shape[0], 4], 0, dtype=np.int32)
                _flag[:, [0, 1]] = 1
                _flag = unset_removal_flag(_boxes, _flag)
            else:
                _boxes = np.zeros((0,4), dtype=np.int32); _flag = np.zeros((0,4), dtype=np.int32)
            info.append([_boxes, _flag])
            sel_indices = np.nonzero(flag[..., 1])[0]
            if len(sel_indices)>0:
                _boxes = np.concatenate([boxes_bl[sel_indices] - np.array([0, margin])[None], boxes_br[sel_indices] + np.array([0, margin])[None]], axis=-1)
                _flag = np.full([_boxes.shape[0], 4], 0, dtype=np.int32)
                _flag[:, [2, 3]] = 1
                _flag = unset_removal_flag(_boxes, _flag)
            else:
                _boxes = np.zeros((0,4), dtype=np.int32); _flag = np.zeros((0,4), dtype=np.int32)
            info.append([_boxes, _flag])
            sel_indices = np.nonzero(np.prod(flag[:, [1, 3]], axis=-1))[0]
            if len(sel_indices)>0:
                _boxes = np.concatenate([boxes_br[sel_indices] - np.array([2*margin, 2*margin])[None], boxes_br[sel_indices] + np.array([2*margin, 2*margin])[None]], axis=-1)
                _flag = np.full([_boxes.shape[0], 4], 1, dtype=np.int32)
            else:
                _boxes = np.zeros((0,4), dtype=np.int32); _flag = np.zeros((0,4), dtype=np.int32)
            info.append([_boxes, _flag])
            return list(info)

        tile_info_sets = _get_tile_sets(wsi_proc_shape, ioconfig_pp)

        pool = None
        if nr_post_proc_workers and nr_post_proc_workers > 0:
            pool = ProcessPoolExecutor(max_workers=nr_post_proc_workers)

        inst_dict = {}
        t0 = time.perf_counter()
        total_tiles = sum(len(b) for b,_ in tile_info_sets)
        self.logger.info(f"WSI {wsi_proc_shape} -> {len(tile_info_sets)} sets, {total_tiles} tiles (tile_shape={tile_shape}, ambiguous={ambiguous_size})")
        # 进度条：按 Cerberus 风格，每 set 一个 tqdm
        for set_idx, (set_bounds, set_flags) in enumerate(tile_info_sets):
            set_name = ["grid","vertical strip","horizontal strip","cross"][set_idx] if set_idx<4 else f"set{set_idx}"
            pbar = tqdm.tqdm(total=len(set_bounds), desc=f"PostProc {set_name} ({set_idx+1}/{len(tile_info_sets)})", ncols=90, leave=True)
            futures = []
            # 为避免多进程传 WSIReader/model (不可 pickle), direct 多进程需用 spawn 且重建
            # 这里简化: 若需要并行, 建议 nr_post_proc_workers=0 或自行用 fork
            # 当前实现为串行 (与 Cerberus tile 部分 pool 类似但 direct 无法直接 pool)
            for tile_idx, tile_bounds in enumerate(set_bounds):
                pbar.set_postfix({"instances": len(inst_dict), "tile": f"{tile_idx+1}/{len(set_bounds)}"})
                tile_flag = set_flags[tile_idx]
                # 可加 mask 过滤: 若 tile 完全在背景则跳过
                # 简化: 检查 mask 裁剪和是否全 0
                # (需要将 XY bounds 转 YX 索引)
                # x0,y0,x1,y1
                x0,y0,x1,y1 = tile_bounds
                y0c, y1c = max(0,y0), min(wsi_mask.shape[0], y1)
                x0c, x1c = max(0,x0), min(wsi_mask.shape[1], x1)
                if y1c <= y0c or x1c <= x0c:
                    pbar.update(1); continue
                if np.sum(wsi_mask[y0c:y1c, x0c:x1c]) == 0:
                    pbar.update(1); continue
                args = (ioconfig_pp, tile_bounds, tile_flag, set_idx, inst_dict, wsi_reader, self.model, resolution, flow_threshold, cellprob_threshold, min_size, diameter, bsize, tile_overlap, self.device)
                # 暂不支持直接 pool (WSIReader 不可 pickle), 故串行
                t_tile = time.perf_counter()
                new_dict, remove_list = _direct_tile_eval_process(*args)
                # 合并
                inst_dict.update(new_dict)
                for uid in remove_list:
                    inst_dict.pop(uid, None)
                pbar.update(1)
                self.logger.debug(f"tile {tile_bounds.tolist()} -> +{len(new_dict)} -{len(remove_list)} = {len(inst_dict)} in {time.perf_counter()-t_tile:.2f}s")
            pbar.close()
            self.logger.info(f"Tile set {set_idx} ({set_name}) done, current instances {len(inst_dict)}")

        # 保存
        wsi_inst_info = inst_dict
        # 兼容 Cerberus dat 结构
        save_dict = {"nuclei": wsi_inst_info} if "box" not in str(list(wsi_inst_info.keys())[:1]) else wsi_inst_info
        # 实际 Cerberus 存 {"Nuclei": {...}, "proc_resolution":..., ...}
        out = {"Nuclei": wsi_inst_info,
               "proc_resolution": {"resolution": wsi_proc_mag, "units": "mpp"},
               "base_resolution": {"resolution": wsi_base_mag, "units": "mpp"},
               "proc_dimensions": np.array(wsi_proc_shape_yx),
               "base_dimensions": np.array(wsi_base_shape)}
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        joblib.dump(out, output_path)
        self.logger.info(f"Saved {len(wsi_inst_info)} nuclei to {output_path}, time {time.perf_counter()-t0:.1f}s")
        # 高分辨率分块可视化：原图分辨率，切成小图，放大后看清边界
        if save_viz_highres:
            try:
                basename = pathlib.Path(output_path).stem
                hr_root = os.path.join(os.path.dirname(os.path.dirname(output_path)) if "dat" in output_path else os.path.dirname(output_path), "viz_highres", basename)
                os.makedirs(hr_root, exist_ok=True)
                # 确定高分辨率 mpp：默认用 base (原图) 分辨率
                base_res = out["base_resolution"]["resolution"]
                if isinstance(base_res, (list, np.ndarray)):
                    base_mpp = float(np.array(base_res).flat[0])
                else:
                    base_mpp = float(base_res)
                viz_mpp = float(viz_highres_mpp) if viz_highres_mpp is not None else base_mpp
                if viz_mpp <= 0:
                    viz_mpp = base_mpp
                scale = wsi_proc_mag / viz_mpp  # proc -> viz 缩放
                proc_h, proc_w = out["proc_dimensions"][0], out["proc_dimensions"][1]
                viz_w = int(round(proc_w * scale))
                viz_h = int(round(proc_h * scale))
                tile = int(viz_highres_tile)
                # 生成 viz 网格，过滤无组织区域（用 wsi_mask）
                xs = list(range(0, viz_w, tile))
                ys = list(range(0, viz_h, tile))
                candidates = []
                for y in ys:
                    for x in xs:
                        x1 = min(x + tile, viz_w); y1 = min(y + tile, viz_h)
                        w, h = x1 - x, y1 - y
                        # 映射回 proc 掩膜坐标判断是否有组织
                        px0, py0 = int(x / scale), int(y / scale)
                        px1, py1 = int(x1 / scale), int(y1 / scale)
                        px0, py0 = max(0, px0), max(0, py0)
                        px1, py1 = min(proc_w, px1), min(proc_h, py1)
                        if px1 <= px0 or py1 <= py0:
                            continue
                        if np.sum(wsi_mask[py0:py1, px0:px1]) == 0:
                            continue
                        candidates.append((x, y, x1, y1))
                # 采样：若候选过多，均匀采样到 max_tiles
                if viz_highres_max_tiles > 0 and len(candidates) > viz_highres_max_tiles:
                    step = len(candidates) / viz_highres_max_tiles
                    idxs = [int(i * step) for i in range(viz_highres_max_tiles)]
                    candidates = [candidates[i] for i in idxs]
                    self.logger.info(f"Highres viz: sampled {len(candidates)}/{len(xs)*len(ys)} tiles")
                else:
                    self.logger.info(f"Highres viz: {len(candidates)} tiles, viz {viz_w}x{viz_h} @ {viz_mpp} mpp, tile {tile}")
                # 预计算缩放后的 box 用于快速过滤（STRtree）
                # 为避免对 10万实例全量 STRtree 过重，分批过滤也可；这里用简单 box 重叠判断 + 分块
                # 先把所有实例的 scaled box/contour 存起来
                scaled_items = []
                for inst in out["Nuclei"].values():
                    b = inst["box"]  # 2x2 [[r0,c0],[r1,c1]] proc
                    # 转 viz: [x,y] proc (c,r) -> viz (c*scale, r*scale)
                    viz_box = np.array([[b[0][1]*scale, b[0][0]*scale], [b[1][1]*scale, b[1][0]*scale]], dtype=np.float32)
                    scaled_items.append((viz_box, inst["contour"] * scale))
                # 逐 tile 读图并画
                rdr_hr = WSIReader.open(wsi_path)
                for tx, ty, tx1, ty1 in tqdm.tqdm(candidates, desc="Viz highres", ncols=90):
                    w, h = tx1 - tx, ty1 - ty
                    try:
                        img_hr = rdr_hr.read_rect(location=(tx, ty), size=(w, h), resolution=viz_mpp, units="mpp", coord_space="resolution")
                        if img_hr.shape[-1] == 4:
                            img_hr = img_hr[..., :3]
                        # tiatoolbox 可能返回 RGBA，统一 BGR 保存
                        canvas_hr = cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        self.logger.warning(f"read highres tile ({tx},{ty}) failed: {e}")
                        continue
                    # 画与该 tile 相交的轮廓
                    cnt_drawn = 0
                    for (viz_box, cnt_scaled) in scaled_items:
                        # 快速 box 相交判断
                        if viz_box[1][0] < tx or viz_box[0][0] > tx1 or viz_box[1][1] < ty or viz_box[0][1] > ty1:
                            continue
                        pts = (cnt_scaled - np.array([tx, ty])).astype(np.int32)
                        # 过滤完全在外的
                        if pts[:,0].max() < 0 or pts[:,0].min() >= w or pts[:,1].max() < 0 or pts[:,1].min() >= h:
                            continue
                        cv2.polylines(canvas_hr, [pts], True, (0, 255, 0), 1)
                        cnt_drawn += 1
                    out_path_hr = os.path.join(hr_root, f"{basename}_x{tx}_y{ty}_w{w}_h{h}.png")
                    cv2.imwrite(out_path_hr, canvas_hr)
                self.logger.info(f"Saved {len(candidates)} highres viz tiles to {hr_root}")
                # Optional: stitched full high-res image (single file)
                if save_viz_highres_full:
                    try:
                        mpix = viz_w * viz_h / 1e6
                        limit = float(viz_highres_full_max_mpix)
                        if mpix > limit:
                            self.logger.warning(f"Skip full high-res stitch: {viz_w}x{viz_h} = {mpix:.1f} MP > limit {limit:.1f} MP. Increase --viz_highres_full_max_mpix or use tiled viz. To avoid OOM/crash.")
                        else:
                            # Estimate RAM: viz_h*viz_w*3 bytes
                            est_gb = viz_w * viz_h * 3 / (1024**3)
                            self.logger.info(f"Stitching full high-res {viz_w}x{viz_h} ({mpix:.1f} MP, ~{est_gb:.2f} GB) ...")
                            # Use memmap to avoid RAM spike, then stream tiles into it
                            import tempfile
                            tmp_mmap = os.path.join(hr_root, f"_tmp_full_{basename}.dat")
                            full = np.memmap(tmp_mmap, dtype=np.uint8, mode='w+', shape=(viz_h, viz_w, 3))
                            # Fill by re-reading tiles (all, not sampled) and drawing
                            # Build full candidates (all tiles, not sampled)
                            xs_all = list(range(0, viz_w, tile))
                            ys_all = list(range(0, viz_h, tile))
                            full_cands = []
                            for yy in ys_all:
                                for xx in xs_all:
                                    x1 = min(xx+tile, viz_w); y1 = min(yy+tile, viz_h)
                                    px0, py0 = int(xx/scale), int(yy/scale)
                                    px1, py1 = int(x1/scale), int(y1/scale)
                                    if px1<=px0 or py1<=py0: continue
                                    if np.sum(wsi_mask[py0:py1, px0:px1]) == 0: continue
                                    full_cands.append((xx,yy,x1,y1))
                            self.logger.info(f"Full stitch: {len(full_cands)} tiles")
                            rdr_full = WSIReader.open(wsi_path)
                            for tx,ty,tx1,ty1 in tqdm.tqdm(full_cands, desc="Stitch full highres", ncols=90):
                                w,h = tx1-tx, ty1-ty
                                try:
                                    img_hr = rdr_full.read_rect(location=(tx,ty), size=(w,h), resolution=viz_mpp, units="mpp", coord_space="resolution")
                                    if img_hr.shape[-1]==4: img_hr=img_hr[...,:3]
                                    canvas = cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR)
                                except Exception:
                                    try:
                                        img_hr = rdr_full.read_rect(location=(tx,ty), size=(w,h), resolution=viz_mpp, units="mpp")
                                        if img_hr.shape[-1]==4: img_hr=img_hr[...,:3]
                                        canvas = cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR)
                                    except Exception as e2:
                                        continue
                                # draw intersecting contours
                                for (viz_box, cnt_scaled) in scaled_items:
                                    if viz_box[1][0] < tx or viz_box[0][0] > tx1 or viz_box[1][1] < ty or viz_box[0][1] > ty1: continue
                                    pts = (cnt_scaled - np.array([tx,ty])).astype(np.int32)
                                    if pts[:,0].max()<0 or pts[:,0].min()>=w or pts[:,1].max()<0 or pts[:,1].min()>=h: continue
                                    cv2.polylines(canvas, [pts], True, (0,255,0), 1)
                                full[ty:ty1, tx:tx1] = canvas
                            # flush and save as tiled BigTIFF then PNG? Use cv2.imwrite via memmap->array may still OOM, so write as BigTIFF then optionally PNG if small
                            full.flush()
                            out_full = os.path.join(os.path.dirname(hr_root), f"{basename}_full_{viz_mpp:.3f}mpp.png")
                            # If image is huge, prefer TIFF to avoid PNG 2GB limit; use tifffile
                            if mpix > 60:
                                out_full = out_full.replace(".png", ".tiff")
                                try:
                                    import tifffile
                                    # tifffile can write big images from memmap incrementally? write directly
                                    tifffile.imwrite(out_full, np.array(full), tile=(512,512), bigtiff=True, compression='lzw')
                                    self.logger.info(f"Saved full high-res BigTIFF to {out_full}")
                                except Exception as e:
                                    self.logger.warning(f"tifffile BigTIFF failed {e}, fallback to cv2")
                                    cv2.imwrite(out_full, np.array(full))
                                    self.logger.info(f"Saved full high-res to {out_full}")
                            else:
                                cv2.imwrite(out_full, np.array(full))
                                self.logger.info(f"Saved full high-res to {out_full}")
                            # cleanup memmap
                            del full
                            try: os.remove(tmp_mmap)
                            except: pass
                    except Exception as e:
                        import traceback
                        self.logger.warning(f"full highres stitch failed: {e}\n{traceback.format_exc()}")
            except Exception as e:
                import traceback
                self.logger.warning(f"highres viz failed: {e}\n{traceback.format_exc()}")
        return out

    def process_wsi_list(self, wsi_list, output_dir, mask_list=None,
                         wsi_proc_mag=0.5, tile_shape=4096, ambiguous_size=64,
                         flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
                         diameter=None, save_thumb=False, save_mask=False,
                         nr_post_proc_workers=0, logging_dir=None,
                         save_viz_highres=False, viz_highres_tile=2048, viz_highres_mpp=None, viz_highres_max_tiles=16,
                         save_viz_highres_full=False, viz_highres_full_max_mpix=120):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/dat", exist_ok=True)
        if save_thumb: os.makedirs(f"{output_dir}/thumb", exist_ok=True)
        if save_mask: os.makedirs(f"{output_dir}/mask", exist_ok=True)
        # 日志：仿 Cerberus 同时输出到控制台 + 文件
        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
        else:
            logging_dir = f"{output_dir}/logs"
            os.makedirs(logging_dir, exist_ok=True)
        # 配置 logger 文件输出 (若未配置)
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            fh = logging.FileHandler(f"{logging_dir}/wsi_cellpose_{time.strftime('%Y%m%d_%H%M%S')}.log")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(fh)
            self.logger.setLevel(logging.INFO)
            # 控制台 handler
            if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
                self.logger.addHandler(sh)
        self.logger.info(f"WSI list: {len(wsi_list)} files, tile_shape={tile_shape}, ambiguous={ambiguous_size}, wsi_proc_mag={wsi_proc_mag}")
        mask_list = mask_list or [None]*len(wsi_list)
        # 外层 WSI 进度条
        for wsi_path, mask_path in tqdm.tqdm(list(zip(wsi_list, mask_list)), desc="WSI", ncols=90):
            basename = pathlib.Path(wsi_path).stem
            out_path = f"{output_dir}/dat/{basename}.dat"
            if os.path.exists(out_path):
                self.logger.info(f"Skip existing {basename}")
                # 高分辨率分块 viz 补生成（原图分辨率）
                if save_viz_highres:
                    hr_root = os.path.join(output_dir, "viz_highres", basename)
                    if not os.path.isdir(hr_root) or len(os.listdir(hr_root)) == 0:
                        try:
                            info = joblib.load(out_path)
                            # 复用与上面相同的逻辑，触发一次高分辨率绘制（不重推理）
                            self.logger.info(f"Generating highres viz from existing dat for {basename}")
                            # 临时构造一个空输出路径再调用高分辨率分支：直接复用 process 的高分辨率代码
                            # 为避免重复推理，这里单独调用一次高分辨率绘制
                            import tempfile
                            # 构造必要的 wsi_mask / base 信息
                            base_res = info["base_resolution"]["resolution"]
                            base_mpp = float(np.array(base_res).flat[0]) if isinstance(base_res, (list, np.ndarray)) else float(base_res)
                            viz_mpp = float(viz_highres_mpp) if viz_highres_mpp is not None else base_mpp
                            scale = wsi_proc_mag / viz_mpp
                            proc_h, proc_w = info["proc_dimensions"][0], info["proc_dimensions"][1]
                            # 重新计算 mask（若有文件则读，无则全1）
                            try:
                                rdr_tmp = WSIReader.open(wsi_path)
                                wsi_mask_hr = np.ones((proc_h, proc_w), dtype=np.uint8)
                                if mask_path and os.path.isfile(mask_path):
                                    wsi_mask_hr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                    wsi_mask_hr = (wsi_mask_hr > 0).astype(np.uint8)
                                    # resize 到 proc 尺寸
                                    wsi_mask_hr = cv2.resize(wsi_mask_hr, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
                            except Exception:
                                wsi_mask_hr = np.ones((proc_h, proc_w), dtype=np.uint8)
                            # 生成候选
                            viz_w = int(round(proc_w * scale)); viz_h = int(round(proc_h * scale))
                            tile = int(viz_highres_tile)
                            xs = list(range(0, viz_w, tile)); ys = list(range(0, viz_h, tile))
                            cands = []
                            for y in ys:
                                for x in xs:
                                    x1 = min(x+tile, viz_w); y1 = min(y+tile, viz_h)
                                    px0, py0 = int(x/scale), int(y/scale)
                                    px1, py1 = int(x1/scale), int(y1/scale)
                                    px0, py0 = max(0,px0), max(0,py0)
                                    px1, py1 = min(proc_w, px1), min(proc_h, py1)
                                    if px1<=px0 or py1<=py0: continue
                                    if np.sum(wsi_mask_hr[py0:py1, px0:px1]) == 0: continue
                                    cands.append((x,y,x1,y1))
                            if viz_highres_max_tiles>0 and len(cands)>viz_highres_max_tiles:
                                step=len(cands)/viz_highres_max_tiles
                                cands=[cands[int(i*step)] for i in range(viz_highres_max_tiles)]
                            os.makedirs(hr_root, exist_ok=True)
                            scaled_items=[]
                            for inst in info["Nuclei"].values():
                                b=inst["box"]
                                viz_box=np.array([[b[0][1]*scale,b[0][0]*scale],[b[1][1]*scale,b[1][0]*scale]],dtype=np.float32)
                                scaled_items.append((viz_box, inst["contour"]*scale))
                            rdr_hr=WSIReader.open(wsi_path)
                            for tx,ty,tx1,ty1 in tqdm.tqdm(cands, desc="Viz highres (from dat)", ncols=90):
                                w,h=tx1-tx,ty1-ty
                                try:
                                    img_hr=rdr_hr.read_rect(location=(tx,ty), size=(w,h), resolution=viz_mpp, units="mpp", coord_space="resolution")
                                    if img_hr.shape[-1]==4: img_hr=img_hr[...,:3]
                                    canvas_hr=cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR)
                                except Exception as e:
                                    continue
                                for (viz_box,cnt_scaled) in scaled_items:
                                    if viz_box[1][0]<tx or viz_box[0][0]>tx1 or viz_box[1][1]<ty or viz_box[0][1]>ty1: continue
                                    pts=(cnt_scaled-np.array([tx,ty])).astype(np.int32)
                                    if pts[:,0].max()<0 or pts[:,0].min()>=w or pts[:,1].max()<0 or pts[:,1].min()>=h: continue
                                    cv2.polylines(canvas_hr,[pts],True,(0,255,0),1)
                                cv2.imwrite(os.path.join(hr_root,f"{basename}_x{tx}_y{ty}_w{w}_h{h}.png"),canvas_hr)
                            self.logger.info(f"Saved {len(cands)} highres tiles to {hr_root}")
                        except Exception as e:
                            import traceback
                            self.logger.warning(f"highres viz from dat failed: {e}\n{traceback.format_exc()}")
                continue
            self.logger.info(f"Processing {basename}")
            # thumbnail / mask 保存 (复刻 Cerberus)
            if save_thumb or save_mask:
                try:
                    rdr = WSIReader.open(wsi_path)
                    if save_thumb:
                        thumb = rdr.slide_thumbnail(resolution=1.25, units="power")
                        cv2.imwrite(f"{output_dir}/thumb/{basename}.png", thumb)
                    if save_mask and mask_path and os.path.isfile(mask_path):
                        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        cv2.imwrite(f"{output_dir}/mask/{basename}.png", m)
                except Exception as e:
                    self.logger.warning(f"thumb/mask save failed: {e}")
            self.process_single_wsi_direct(
                wsi_path, out_path, mask_path,
                wsi_proc_mag=wsi_proc_mag, tile_shape=tile_shape,
                ambiguous_size=ambiguous_size, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, min_size=min_size,
                diameter=diameter, nr_post_proc_workers=nr_post_proc_workers,
                save_viz_highres=save_viz_highres, viz_highres_tile=viz_highres_tile, viz_highres_mpp=viz_highres_mpp, viz_highres_max_tiles=viz_highres_max_tiles,
                save_viz_highres_full=save_viz_highres_full, viz_highres_full_max_mpix=viz_highres_full_max_mpix
            )


def get_bounding_box_fallback(img):
    """fallback for misc.utils.get_bounding_box"""
    return _get_bbox_fallback(img)


