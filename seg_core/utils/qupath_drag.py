"""qupath_drag - thin wrapper around qp_clean_draggable QuPath engine
Original: /media/linjiatai/086399513677BED7/zhang/tool/qp_clean_draggable.py
No shapely/simulation — 100% QuPath GsonTools validation.
"""
import os, json, glob

from seg_core.utils.qp_clean_draggable import (
    QPATH_BIN,
    GROOVY_TEMPLATE,
    run_one,
    split_file,
)

def _find_qpath():
    if os.path.isfile(QPATH_BIN) and os.access(QPATH_BIN, os.X_OK):
        return QPATH_BIN
    import shutil
    w = shutil.which("QuPath")
    if w: return w
    return None

def clean_file_qpath(in_path, out_path, qpath_bin=None):
    """Same as qp_clean_draggable.py: validate per-nucleus via QuPath headless, return (dropped, kept)"""
    # count before cleaning
    try:
        with open(in_path, encoding="utf-8") as f:
            total = len(json.load(f).get("features", []))
    except:
        total = 0
    qpath_bin = qpath_bin or _find_qpath()
    if not qpath_bin:
        raise FileNotFoundError(f"QuPath not found: {QPATH_BIN}")
    if qpath_bin != QPATH_BIN:
        import seg_core.utils.qp_clean_draggable as _m
        old = _m.QPATH_BIN
        _m.QPATH_BIN = qpath_bin
        try:
            run_one(in_path, out_path)
        finally:
            _m.QPATH_BIN = old
    else:
        run_one(in_path, out_path)
    try:
        with open(out_path, encoding="utf-8") as f:
            kept = len(json.load(f).get("features", []))
    except:
        kept = 0
    dropped = max(0, total - kept)
    return dropped, kept

def clean_dir(input_dir, output_dir, split=None):
    """Batch directory cleaning, same logic as qp_clean_draggable.py --input dir"""
    os.makedirs(output_dir, exist_ok=True)
    qbin = _find_qpath()
    if not qbin:
        raise FileNotFoundError(f"QuPath not found: {QPATH_BIN}")
    for p in sorted(glob.glob(os.path.join(input_dir, "*.geojson"))):
        out = os.path.join(output_dir, os.path.basename(p))
        clean_file_qpath(p, out, qbin)
        if split and os.path.isfile(out):
            with open(out, encoding="utf-8") as f:
                feats = json.load(f).get("features", [])
            if len(feats) > split:
                os.remove(out)
                parts = (len(feats) + split - 1) // split
                for i in range(parts):
                    slc = feats[i*split:(i+1)*split]
                    pp = str(out).replace(".geojson", f"_part{i+1}of{parts}.geojson")
                    json.dump({"type":"FeatureCollection","features":slc}, open(pp,"w",encoding="utf-8"))
