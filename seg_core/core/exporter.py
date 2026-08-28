"""统一结果导出与高分辨率分块可视化（原图分辨率切小图）"""
import os
import pathlib
import cv2
import numpy as np
import joblib
import tqdm
from tiatoolbox.wsicore.wsireader import WSIReader


def save_viz_highres_from_dat(dat_path, wsi_path, output_dir,
                               wsi_proc_mag=0.5, viz_highres_tile=2048,
                               viz_highres_mpp=None, viz_highres_max_tiles=16,
                               mask_path=None, logger=None):
    """从已有 .dat 生成高分辨率分块可视化，复用 CellposeWSI 同款逻辑"""
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    info = joblib.load(dat_path)
    basename = pathlib.Path(dat_path).stem
    # 也支持传入的 dat 是 dict 而非路径
    if isinstance(info, dict) and "Nuclei" not in info and "nuclei" in info:
        info["Nuclei"] = info.pop("nuclei")
    base_res = info["base_resolution"]["resolution"]
    base_mpp = float(np.array(base_res).flat[0]) if isinstance(base_res, (list, np.ndarray)) else float(base_res)
    viz_mpp = float(viz_highres_mpp) if viz_highres_mpp is not None else base_mpp
    if viz_mpp <= 0:
        viz_mpp = base_mpp
    scale = wsi_proc_mag / viz_mpp
    proc_h, proc_w = int(info["proc_dimensions"][0]), int(info["proc_dimensions"][1])
    viz_w = int(round(proc_w * scale))
    viz_h = int(round(proc_h * scale))
    tile = int(viz_highres_tile)
    # mask
    try:
        wsi_mask = np.ones((proc_h, proc_w), dtype=np.uint8)
        if mask_path and os.path.isfile(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = (m > 0).astype(np.uint8)
                wsi_mask = cv2.resize(m, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        wsi_mask = np.ones((proc_h, proc_w), dtype=np.uint8)
    xs = list(range(0, viz_w, tile))
    ys = list(range(0, viz_h, tile))
    cands = []
    for y in ys:
        for x in xs:
            x1 = min(x + tile, viz_w); y1 = min(y + tile, viz_h)
            px0, py0 = int(x / scale), int(y / scale)
            px1, py1 = int(x1 / scale), int(y1 / scale)
            px0, py0 = max(0, px0), max(0, py0)
            px1, py1 = min(proc_w, px1), min(proc_h, py1)
            if px1 <= px0 or py1 <= py0:
                continue
            if np.sum(wsi_mask[py0:py1, px0:px1]) == 0:
                continue
            cands.append((x, y, x1, y1))
    if viz_highres_max_tiles > 0 and len(cands) > viz_highres_max_tiles:
        step = len(cands) / viz_highres_max_tiles
        cands = [cands[int(i * step)] for i in range(viz_highres_max_tiles)]
    hr_root = os.path.join(output_dir, "viz_highres", basename)
    os.makedirs(hr_root, exist_ok=True)
    scaled_items = []
    for inst in info.get("Nuclei", {}).values():
        b = inst["box"]
        viz_box = np.array([[b[0][1]*scale, b[0][0]*scale], [b[1][1]*scale, b[1][0]*scale]], dtype=np.float32)
        scaled_items.append((viz_box, inst["contour"] * scale))
    # 也画 gland/lumen 若有
    for key in ("Gland", "Lumen"):
        for inst in info.get(key, {}).values():
            b = inst.get("box")
            cnt = inst.get("contour")
            if b is None or cnt is None:
                continue
            viz_box = np.array([[b[0][1]*scale, b[0][0]*scale], [b[1][1]*scale, b[1][0]*scale]], dtype=np.float32)
            scaled_items.append((viz_box, np.array(cnt) * scale))
    rdr_hr = WSIReader.open(wsi_path)
    for tx, ty, tx1, ty1 in tqdm.tqdm(cands, desc="Viz highres", ncols=90):
        w, h = tx1 - tx, ty1 - ty
        try:
            img_hr = rdr_hr.read_rect(location=(tx, ty), size=(w, h), resolution=viz_mpp, units="mpp", coord_space="resolution")
            if img_hr.shape[-1] == 4:
                img_hr = img_hr[..., :3]
            canvas_hr = cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR)
        except Exception:
            try:
                img_hr = rdr_hr.read_rect(location=(tx, ty), size=(w, h), resolution=viz_mpp, units="mpp")
                if img_hr.shape[-1] == 4:
                    img_hr = img_hr[..., :3]
                canvas_hr = cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR)
            except Exception as e:
                if logger:
                    logger.warning(f"read highres tile ({tx},{ty}) failed: {e}")
                continue
        for viz_box, cnt_scaled in scaled_items:
            if viz_box[1][0] < tx or viz_box[0][0] > tx1 or viz_box[1][1] < ty or viz_box[0][1] > ty1:
                continue
            pts = (cnt_scaled - np.array([tx, ty])).astype(np.int32)
            if pts[:, 0].max() < 0 or pts[:, 0].min() >= w or pts[:, 1].max() < 0 or pts[:, 1].min() >= h:
                continue
            # gland 用蓝，nuclei 用绿
            cv2.polylines(canvas_hr, [pts], True, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(hr_root, f"{basename}_x{tx}_y{ty}_w{w}_h{h}.png"), canvas_hr)
    log(f"Saved {len(cands)} highres viz tiles to {hr_root}")
    # optional full stitch if requested via kwargs in future (kept for API compatibility)
    return hr_root

def save_qupath_geojson_from_dat(dat_path, output_dir, logger=None):
    """Export dat contours to QuPath GeoJSON (base resolution coordinates)."""
    import json, pathlib, numpy as np, joblib, os, cv2
    def log(msg):
        (logger.info(msg) if logger else print(msg))
    info=joblib.load(dat_path)
    basename=pathlib.Path(dat_path).stem
    # ensure base/proc mpp for scaling (contours are at proc)
    try:
        proc_mpp=float(np.array(info["proc_resolution"]["resolution"]).flat[0]) if isinstance(info["proc_resolution"]["resolution"], (list, np.ndarray)) else float(info["proc_resolution"]["resolution"])
        base_mpp=float(np.array(info["base_resolution"]["resolution"]).flat[0]) if isinstance(info["base_resolution"]["resolution"], (list, np.ndarray)) else float(info["base_resolution"]["resolution"])
        scale=proc_mpp/base_mpp if base_mpp else 1.0
    except Exception:
        scale=1.0
    out_dir=os.path.join(output_dir, "qupath")
    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir, f"{basename}.geojson")
    features=[]
    for cls_name in ["Nuclei","Gland","Lumen"]:
        d=info.get(cls_name, {})
        # also handle lowercase
        if not d and cls_name.lower() in info:
            d=info[cls_name.lower()]
        for uid, inst in d.items():
            cnt=np.array(inst.get("contour", []), dtype=np.float64)
            if cnt.size==0 or cnt.shape[0]<3:
                continue
            # scale to base (QuPath level 0)
            if scale!=1.0:
                cnt=cnt*scale
            # GeoJSON Polygon expects closed ring, ensure closed
            pts=cnt.tolist()
            if pts[0]!=pts[-1]:
                pts.append(pts[0])
            # QuPath classification
            feat={"type":"Feature","geometry":{"type":"Polygon","coordinates":[pts]},"properties":{"classification":{"name": cls_name},"objectType":"annotation","isLocked":False}}
            features.append(feat)
    fc={"type":"FeatureCollection","features":features}
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(fc,f)
    log(f"Saved QuPath GeoJSON {len(features)} features to {out_path} (scale {scale:.3f})")
    return out_path

