#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""inspect_dat.py

Dedicated viewer for Cerberus WSI instance-segmentation output (.dat).

Usage:
  inspect_dat.py [--input=<path>] [--viz] [--output=<path>] \
            [--max_inst=<n>] [--task=<name>] [--scale=<n>] \
            [--region=<x,y,w,h>] [--wsi_dir=<path>] [--mag=<mpp>] [--tif]
  inspect_dat.py (-h | --help)
  inspect_dat.py --version

python /media/linjiatai/086399513677BED7/zhang/tool/inspect_dat.py --input /media/linjiatai/086399513677BED7/zhang/dataset/test_out/dat --viz --output /media/linjiatai/086399513677BED7/zhang/dataset/test_out --max_inst=-1

Options:
  -h --help                 Show this string.
  --version                 Show version.
  --input=<path>            Path to a .dat file or a directory containing .dat files.
  --viz                     Draw instance contours on the thumbnail and save to --output.
  --output=<path>           Output directory for visualizations. [default: viz_dat/]
  --max_inst=<n>            Max instances to draw per task. Use -1 to draw all. [default: 500]
  --task=<name>             Task to draw (Nuclei/Gland/Lumen). Default: all.
  --scale=<n>               Downscale for the overlay canvas when no thumbnail is found. [default: 0.5]
  --region=<x,y,w,h>        Crop a region [x, y, w, h] in proc (0.5 mpp) coords and draw at full detail.
  --wsi_dir=<path>          Directory of input WSIs (required with --region).
  --mag=<mpp>               Resolution (mpp) for reading the region. Smaller = sharper. [default: 0.5]
  --tif                     Save region output as lossless TIFF instead of PNG.

"""

import os
import glob
import joblib
import numpy as np
import cv2

from docopt import docopt

# -------------------------------------------------------------------------------------------------------


def _get_dat_files(input_path):
    """Return a list of .dat file paths from a file or directory input."""
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.dat")))
    raise FileNotFoundError("Input not found: %s" % input_path)


def _find_thumb(dat_path, basename):
    """Locate the matching thumbnail file alongside the .dat output."""
    dat_dir = os.path.dirname(dat_path)
    candidates = [
        os.path.join(dat_dir, "..", "thumb", basename + ".png"),
        os.path.join(dat_dir, "..", "..", "thumb", basename + ".png"),
        os.path.join(dat_dir, "thumb", basename + ".png"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _report_dat(dat_path):
    """Print the structure of a single .dat file."""
    info = joblib.load(dat_path)
    print("=" * 70)
    print("File: %s" % dat_path)
    print("=" * 70)
    for key, value in info.items():
        if isinstance(value, dict):
            print("  %-14s dict, %d instances" % (key, len(value)))
        elif isinstance(value, np.ndarray):
            print("  %-14s ndarray %s = %s" % (key, value.shape, value))
        else:
            print("  %-14s %s = %s" % (key, type(value).__name__, value))

    # show fields of the first instance in the first task dict found
    for key, value in info.items():
        if isinstance(value, dict) and len(value) > 0:
            inst = list(value.values())[0]
            print("  -- first %s instance fields --" % key)
            for k, v in inst.items():
                if hasattr(v, "shape"):
                    print("     %-10s shape=%s dtype=%s" % (k, v.shape, v.dtype))
                else:
                    print("     %-10s %s = %s" % (k, type(v).__name__, v))
            break
    print()
    return info


def _draw_dat(info, thumb_path, output_path, max_inst, tasks, scale):
    """Draw instance contours onto the thumbnail and save the overlay."""
    if thumb_path is not None:
        canvas = cv2.imread(thumb_path)
        sx = canvas.shape[1] / float(info["proc_dimensions"][1])
        sy = canvas.shape[0] / float(info["proc_dimensions"][0])
    else:
        print("  [WARNING] thumbnail not found, using white canvas as background")
        proc_h, proc_w = info["proc_dimensions"]
        canvas = np.full(
            (int(proc_h * scale), int(proc_w * scale), 3), 255, dtype=np.uint8
        )
        sx, sy = scale, scale

    colour_map = {
        "Nuclei": (0, 255, 0),
        "Gland": (0, 0, 255),
        "Lumen": (255, 0, 0),
    }
    for task in tasks:
        if task not in info or not isinstance(info[task], dict):
            print("  [%s] not found, skipped" % task)
            continue
        colour = colour_map.get(task, (255, 255, 0))
        count = 0
        for inst in info[task].values():
            if max_inst >= 0 and count >= max_inst:
                break
            pts = (inst["contour"] * np.array([sx, sy])).astype(np.int32)
            cv2.polylines(canvas, [pts], True, colour, 1)
            count += 1
        print("  [%s] drew %d instances" % (task, count))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print("Saved overlay: %s" % output_path)


def _draw_region(info, wsi_dir, wsi_basename, region, mag, output_path, max_inst, tasks):
    """Crop a high-resolution region from the WSI and overlay instances inside it.

    Region coordinates are given in proc (0.5 mpp) space. The background is read
    directly from the input WSI at resolution `mag` (mpp), so smaller `mag` yields
    a sharper, less blurred result than using the thumbnail.
    """
    from tiatoolbox.wsicore.wsireader import WSIReader

    wsi_candidates = glob.glob(os.path.join(wsi_dir, wsi_basename + ".*"))
    if not wsi_candidates:
        raise FileNotFoundError(
            "WSI not found for %s in %s" % (wsi_basename, wsi_dir)
        )
    wsi_reader = WSIReader.open(input_img=wsi_candidates[0])

    rx, ry, rw, rh = region
    bounds = [rx, ry, rx + rw, ry + rh]
    region_img = wsi_reader.read_bounds(bounds, resolution=mag, units="mpp")
    # some tiatoolbox versions return (img, bounds), others just img
    if isinstance(region_img, tuple):
        region_img, out_bounds = region_img
        origin_x, origin_y = out_bounds[0], out_bounds[1]
    else:
        origin_x, origin_y = rx, ry
    # proc (0.5 mpp) px per read px
    sf = 0.5 / mag

    canvas = cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR)

    colour_map = {
        "Nuclei": (0, 255, 0),
        "Gland": (0, 0, 255),
        "Lumen": (255, 0, 0),
    }
    for task in tasks:
        if task not in info or not isinstance(info[task], dict):
            print("  [%s] not found, skipped" % task)
            continue
        colour = colour_map.get(task, (255, 255, 0))
        count = 0
        for inst in info[task].values():
            if max_inst >= 0 and count >= max_inst:
                break
            contour = inst["contour"]
            # quick reject instances outside the region (proc coords)
            c_min = contour.min(axis=0)
            c_max = contour.max(axis=0)
            if c_max[0] < rx or c_min[0] > rx + rw:
                continue
            if c_max[1] < ry or c_min[1] > ry + rh:
                continue
            pts = ((contour - np.array([origin_x, origin_y])) * sf).astype(np.int32)
            cv2.polylines(canvas, [pts], True, colour, 1)
            count += 1
        print("  [%s] drew %d instances in region" % (task, count))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print("Saved region overlay: %s" % output_path)


# -------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__, version="Cerberus DAT Viewer")

    if args["--input"] is None:
        raise SystemExit("[ERROR] --input is required. Run with --help for usage.")

    dat_files = _get_dat_files(args["--input"])
    tasks = [args["--task"]] if args["--task"] else ["Nuclei", "Gland", "Lumen"]
    max_inst = int(args["--max_inst"])
    scale = float(args["--scale"])

    region = None
    if args["--region"] is not None:
        region = [int(v) for v in args["--region"].split(",")]
        if len(region) != 4:
            raise SystemExit("[ERROR] --region must be x,y,w,h")
        if args["--wsi_dir"] is None:
            raise SystemExit(
                "[ERROR] --region requires --wsi_dir=<directory of input WSIs>"
            )

    region_ext = ".tif" if args["--tif"] else ".png"
    for dat_file in dat_files:
        info = _report_dat(dat_file)
        if args["--viz"]:
            basename = os.path.basename(dat_file)[:-4]
            if region is not None:
                out_file = os.path.join(
                    args["--output"],
                    "%s_region_%d_%d%s"
                    % (basename, region[0], region[1], region_ext),
                )
                _draw_region(
                    info,
                    args["--wsi_dir"],
                    basename,
                    region,
                    float(args["--mag"]),
                    out_file,
                    max_inst,
                    tasks,
                )
            else:
                thumb_path = _find_thumb(dat_file, basename)
                out_file = os.path.join(
                    args["--output"], basename + "_overlay.png"
                )
                _draw_dat(info, thumb_path, out_file, max_inst, tasks, scale)
