"""QuPath GeoJSON export (dat -> directly draggable GeoJSON via qp_clean engine)"""
import os
import pathlib
import numpy as np
import joblib

def save_qupath_geojson_from_dat(dat_path, output_dir, logger=None, fix_invalid=True, split_large=None, use_qpath_engine=True):
    """Export dat contours to QuPath GeoJSON (base resolution coordinates).

    fix_invalid: ignored if use_qpath_engine=True (QuPath engine is authoritative).
    split_large: None or 0 = single file; set to 30000 to split into ~15M/part.
    use_qpath_engine: True = export raw GeoJSON then validate per-nucleus via QuPath engine,
                      output is 100% directly draggable (zero Reduction failed). Requires local QuPath install.
                      False = export raw only (no cleaning).
    """
    import json, pathlib, numpy as np, joblib, os, glob
    def log(msg):
        (logger.info(msg) if logger else print(msg))
    info=joblib.load(dat_path)
    basename=pathlib.Path(dat_path).stem
    try:
        proc_mpp=float(np.array(info["proc_resolution"]["resolution"]).flat[0]) if isinstance(info["proc_resolution"]["resolution"], (list, np.ndarray)) else float(info["proc_resolution"]["resolution"])
        base_mpp=float(np.array(info["base_resolution"]["resolution"]).flat[0]) if isinstance(info["base_resolution"]["resolution"], (list, np.ndarray)) else float(info["base_resolution"]["resolution"])
        scale=proc_mpp/base_mpp if base_mpp else 1.0
    except Exception:
        scale=1.0
    out_dir=os.path.join(output_dir, "qupath")
    os.makedirs(out_dir, exist_ok=True)
    features=[]
    for cls_name in ["Nuclei","Gland","Lumen"]:
        d=info.get(cls_name, {})
        if not d and cls_name.lower() in info:
            d=info[cls_name.lower()]
        for uid, inst in d.items():
            cnt=np.array(inst.get("contour", []), dtype=np.float64)
            if cnt.size==0 or cnt.shape[0]<3:
                continue
            if scale!=1.0:
                cnt=cnt*scale
            pts=cnt.tolist()
            if pts[0]!=pts[-1]:
                pts.append(pts[0])
            color_map = {"Nuclei": [0,255,0], "Gland": [0,0,255], "Lumen": [255,255,0]}
            col = color_map.get(cls_name, [128,128,128])
            feat={"type":"Feature","geometry":{"type":"Polygon","coordinates":[pts]},"properties":{"objectType":"annotation","classification":{"name": cls_name, "color": col}}}
            features.append(feat)
    # ---- write raw (single or split) ----
    raw_paths=[]
    if split_large and len(features) > split_large:
        parts = (len(features) + split_large - 1)//split_large
        for i in range(parts):
            part = features[i*split_large:(i+1)*split_large]
            fc={"type":"FeatureCollection","features":part}
            out_path=os.path.join(out_dir, f"{basename}_part{i+1}of{parts}.geojson")
            with open(out_path,"w",encoding="utf-8") as f:
                json.dump(fc,f)
            raw_paths.append(out_path)
    else:
        out_path=os.path.join(out_dir, f"{basename}.geojson")
        fc={"type":"FeatureCollection","features":features}
        with open(out_path,"w",encoding="utf-8") as f:
            json.dump(fc,f)
        raw_paths.append(out_path)
    if not use_qpath_engine or not fix_invalid:
        log(f"Saved QuPath GeoJSON {len(features)} features -> {len(raw_paths)} file(s) to {out_dir} (scale {scale:.3f}, raw, no QuPath clean)")
        return raw_paths[0]
    # ---- QuPath engine clean each file in place ----
    try:
        from seg_core.utils.qupath_drag import clean_file_qpath
        cleaned_keep = 0
        cleaned_drop = 0
        for rp in raw_paths:
            tmp = rp + ".raw"
            os.rename(rp, tmp)
            try:
                dropped, kept = clean_file_qpath(tmp, rp)
                cleaned_drop += dropped
                cleaned_keep += kept
            finally:
                try: os.remove(tmp)
                except: pass
        log(f"Saved QuPath GeoJSON {len(features)} -> {cleaned_keep} kept (dropped {cleaned_drop}) -> {len(raw_paths)} file(s) to {out_dir} (scale {scale:.3f}, QuPath engine cleaned, directly draggable)")
    except Exception as e:
        import traceback
        log(f"[WARN] QuPath engine clean failed, keeping raw: {e}\n{traceback.format_exc()}")
    return raw_paths[0]

