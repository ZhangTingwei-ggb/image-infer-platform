#!/usr/bin/env python
"""
seg_platform 统一入口 — 不同命令调用不同模型
  python cli.py --model cerberus --input_dir ... --output_dir ... [cerberus 参数]
  python cli.py --model cellpose --input_dir ... --output_dir ... [cellpose 参数]
"""
import os
import sys
import glob
import argparse
import pathlib

def build_parser():
    p = argparse.ArgumentParser(description="seg_platform — Cerberus / Cellpose WSI 统一平台")
    p.add_argument("--model", required=True, choices=["cerberus", "cellpose"], help="选择模型")
    # 通用 IO
    p.add_argument("--input_dir", required=True, help="WSI 输入目录")
    p.add_argument("--output_dir", required=True, help="输出目录")
    p.add_argument("--msk_dir", default=None, help="组织掩膜目录（同名 png，可选）")
    p.add_argument("--wsi_file_ext", default=".svs", help="WSI extension, comma-separated for multiple e.g. .svs,.tiff or * for all [default: .svs]")
    p.add_argument("--gpu", default="0", help="GPU id(s) for CUDA_VISIBLE_DEVICES, e.g. \"0\" or \"0,1\"; empty string = CPU")
    # 通用 WSI / 后处理（两模型对齐）
    p.add_argument("--wsi_proc_mag", type=float, default=0.5, help="处理 mpp [default: 0.5]")
    p.add_argument("--tile_shape", type=int, default=4096, help="核后处理 tile [default: 4096]")
    p.add_argument("--ambiguous_size", type=int, default=64, help="模糊边界 [default: 64]")
    p.add_argument("--patch_input_shape", type=int, default=448, help="cerberus patch_input [default: 448]")
    p.add_argument("--patch_output_shape", type=int, default=144, help="cerberus patch_output [default: 144]")
    p.add_argument("--chunk_shape", type=int, default=15000, help="cerberus chunk [default: 15000]")
    p.add_argument("--batch_size", type=int, default=30, help="batch size")
    p.add_argument("--nr_inference_workers", type=int, default=0, help="cerberus 推理并行")
    p.add_argument("--nr_post_proc_workers", type=int, default=0, help="后处理并行")
    p.add_argument("--save_thumb", action="store_true", help="保存缩略图")
    p.add_argument("--save_mask", action="store_true", help="保存掩膜")
    p.add_argument("--use_cerberus_infer", action="store_true", help="cellpose use Cerberus-style chunk+flow memmap inference (chunk/patch batching, logs Inference/PostProc Time) instead of direct per-4096")
    p.add_argument("--save_qupath", action="store_true", help="export QuPath-readable GeoJSON (qupath/<basename>.geojson) alongside dat — validated by QuPath engine (GsonTools), directly draggable in QuPath 0.7 (zero Reduction failed, single file by default)")
    p.add_argument("--qupath_split", type=int, default=0, help="QuPath split N (default 0 = single file, 30000 = ~15M/part) [only with --save_qupath]")
    p.add_argument("--qupath_no_clean", action="store_true", help="disable QuPath engine cleaning (raw export, not draggable)")
    # Cellpose 专属
    p.add_argument("--cellpose_model", default="cpsam", help="cellpose 模型名或路径 [default: cpsam]")
    p.add_argument("--diameter", type=float, default=None, help="cellpose diameter，None 自动")
    p.add_argument("--flow_threshold", type=float, default=0.4)
    p.add_argument("--cellprob_threshold", type=float, default=0.0)
    p.add_argument("--min_size", type=int, default=15)
    # Cerberus 专属
    p.add_argument("--cerberus_model", default=None, help="cerberus 权重目录，默认 models/cerberus/pretrained_weights/resnet34_cerberus")
    p.add_argument("--cache_path", default=None, help="cerberus cache 路径，默认 output_dir/cache")
    p.add_argument("--logging_dir", default=None, help="日志目录，默认 output_dir/logs")
    return p

def collect_wsi(input_dir, wsi_file_ext, msk_dir):
    raw = (wsi_file_ext or ".svs").strip()
    # allow comma-separated multi ext: ".svs,.tiff" or "*" for all
    if raw in ("*", "*.*"):
        exts = ["*"]
    else:
        parts = [pp.strip() for pp in raw.replace(";", ",").replace(" ", ",").split(",") if pp.strip()]
        exts = []
        for pp in parts:
            if pp == "*":
                exts.append("*")
            elif pp.startswith("*."):
                exts.append(pp[1:])
            elif pp.startswith("."):
                exts.append(pp)
            elif pp.startswith("*"):
                exts.append(pp[1:])
            else:
                exts.append("." + pp.lstrip("."))
        if not exts:
            exts = [".svs"]
    wsi_files = []
    for ext in exts:
        pat = f"{input_dir}/*{ext}" if ext != "*" else f"{input_dir}/*"
        wsi_files.extend(glob.glob(pat))
    wsi_files = sorted(set(wsi_files))
    if not wsi_files and len(exts)==1 and exts[0] != "*":
        for ext in [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", "*"]:
            wsi_files = sorted(glob.glob(f"{input_dir}/*{ext}"))
            if wsi_files:
                print(f"[INFO] No *{exts[0]} found, fallback to *{ext}")
                break
    wsi_files = [f for f in wsi_files if os.path.isfile(f)]
    if not wsi_files:
        raise FileNotFoundError(f"No WSI found in {input_dir} (ext={wsi_file_ext})")
    mask_list = []
    for wsi in wsi_files:
        stem = pathlib.Path(wsi).stem
        if msk_dir and os.path.isfile(os.path.join(msk_dir, stem + ".png")):
            mask_list.append(os.path.join(msk_dir, stem + ".png"))
        else:
            mask_list.append(None)
    if msk_dir:
        filtered = [(w, m) for w, m in zip(wsi_files, mask_list) if m is not None]
        if filtered:
            wsi_files, mask_list = zip(*filtered)
            wsi_files, mask_list = list(wsi_files), list(mask_list)
        else:
            print(f"[WARN] msk_dir 提供但无匹配 mask，处理全部 {len(wsi_files)} 张")
    return wsi_files, mask_list

def main():
    args = build_parser().parse_args()
    wsi_list, mask_list = collect_wsi(args.input_dir, args.wsi_file_ext, args.msk_dir)
    print(f"Found {len(wsi_list)} WSIs, model={args.model}")

    common_kwargs = dict(
        wsi_proc_mag=args.wsi_proc_mag,
        tile_shape=args.tile_shape,
        ambiguous_size=args.ambiguous_size,
        patch_input_shape=args.patch_input_shape,
        patch_output_shape=args.patch_output_shape,
        chunk_shape=args.chunk_shape,
        batch_size=args.batch_size,
        nr_post_proc_workers=args.nr_post_proc_workers,
        save_thumb=args.save_thumb,
        save_mask=args.save_mask,
        save_qupath=args.save_qupath,
        qupath_split=(None if args.qupath_split==0 else args.qupath_split),
        qupath_no_clean=args.qupath_no_clean,
        msk_dir=args.msk_dir,
    )

    if args.model == "cerberus":
        from seg_core.adapters.cerberus_adapter import CerberusAdapter
        adapter = CerberusAdapter(model_dir=args.cerberus_model, gpu=args.gpu)
        adapter.run(
            wsi_list, mask_list, args.output_dir,
            nr_inference_workers=args.nr_inference_workers,
            cache_path=args.cache_path or os.path.join(args.output_dir, "cache"),
            logging_dir=args.logging_dir or os.path.join(args.output_dir, "logs"),
            **common_kwargs,
        )
    else:
        from seg_core.adapters.cellpose_adapter import CellposeAdapter
        adapter = CellposeAdapter(model_name=args.cellpose_model, gpu=args.gpu)
        adapter.run(
            wsi_list, mask_list, args.output_dir,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            **common_kwargs,
        )
    print("Done.")

if __name__ == "__main__":
    main()
