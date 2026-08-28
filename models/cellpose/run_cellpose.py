#!/usr/bin/env python
"""
run_wsi_cellpose.py — 模仿 cerberus/run_infer_wsi.py 的 Cellpose WSI 入口
                     Cellpose 负责推理，Cerberus 负责拼整块（4类瓦片 + STRtree 去重）

==================== 完整用法 ====================

1) 仅推理（最快，不画图）：
  python run_wsi_cellpose.py --model cpsam --gpu 0 \
    --input_dir /media/linjiatai/086399513677BED7/zhang/dataset/test \
    --output_dir /media/linjiatai/086399513677BED7/zhang/dataset/test_out \
    --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5 --diameter 20

2) 推理 + 高分辨率分块可视化（原图分辨率切小图，放大看清边界）：
  python run_wsi_cellpose.py ... --save_viz_highres --viz_highres_tile 2048 --viz_highres_max_tiles 16

3) 更细腻的高分辨率：
  python run_wsi_cellpose.py ... --save_viz_highres --viz_highres_tile 1024 --viz_highres_mpp 0.20 --viz_highres_max_tiles 36

4) 已有 dat 补画高分辨率图（不重推理，直接从 dat 生成 viz_highres）：
  python run_wsi_cellpose.py --input_dir .../test --output_dir .../test_out \
    --model cpsam --gpu 0 --save_viz_highres --viz_highres_tile 2048 --viz_highres_max_tiles 4

==================== 命令行参数 ====================

基础：
  --model STR              Cellpose 模型：cpsam / cyto / cyto2 / cyto3 / 本地路径  [default: cpsam]
  --gpu STR                GPU id，如 0 或 0,1，传空字符串用 CPU  [default: 0]
  --input_dir PATH         WSI 输入目录（必填）
  --output_dir PATH        输出目录（必填），自动建 dat/thumb/mask/viz_highres/logs
  --msk_dir PATH           组织掩膜目录（同名 png，可选）；若提供则只处理有掩膜的 WSI
  --wsi_file_ext STR       WSI 后缀  [default: .svs]

Cerberus 对齐（后处理拼块）：
  --tile_shape INT         核后处理小瓦片边长  [default: 4096]  显存不足可改 2048
  --chunk_shape INT        已废弃，direct 模式忽略，保留兼容 Cerberus  [default: 15000]
  --ambiguous_size INT     模糊边界宽度 px  [default: 64]  瓦片重叠去重的 margin
  --wsi_proc_mag FLOAT     处理 mpp  [default: 0.5]  0.5=20x，0.25=40x；程序自动处理无 mpp 的 TIFF
  --patch_input_shape INT  保留兼容  [default: 448]
  --patch_output_shape INT 保留兼容  [default: 144]

Cellpose 推理：
  --diameter FLOAT         细胞直径，None 自动估计  [default: None]  H&E 在 0.5mpp 建议 15-20，30偏大易粘连
  --flow_threshold FLOAT   flow 误差阈值  [default: 0.4]  越大检出越多
  --cellprob_threshold FLOAT cellprob 阈值  [default: 0.0]  越小检出越多
  --min_size INT           最小面积过滤  [default: 15]
  --batch_size INT         保留参数，direct 模式内部用 bsize=256  [default: 8]
  --nr_post_proc_workers INT 后处理并行  [default: 0]  direct 模式建议 0（WSIReader 不可 pickle）
  --use_bfloat16           强制 bfloat16，默认 float32；4090 上 bfloat16 会触发 upsample_linear1d 崩溃

可视化控制（默认不画，用参数控制是否生成）：
  --save_thumb             保存缩略图到 output_dir/thumb/<basename>.png
  --save_mask              保存组织掩膜到 output_dir/mask/
  --save_viz_highres       生成高分辨率分块 output_dir/viz_highres/<basename>/xxx.png
    --viz_highres_tile INT   每张小图边长  [default: 2048]  1024 更细，4096 更大视野
    --viz_highres_mpp FLOAT  高分辨率 mpp  [default: None=原图 base mpp 约0.261]  0.20 更清晰，0.5 与处理一致
    --viz_highres_max_tiles INT 最多生成多少小图  [default: 16]  均匀采样，-1 生成全部（大 WSI 会上千张）

输出结构：
  output_dir/dat/<basename>.dat          joblib.dump {Nuclei:{uuid:{box,contour,centroid}}, proc_resolution, base_resolution, proc_dimensions, base_dimensions}
  output_dir/thumb/<basename>.png        缩略图（若 --save_thumb）
  output_dir/mask/<basename>.png         掩膜（若 --save_mask）
  output_dir/viz_highres/<basename>/*.png 高分小图（若 --save_viz_highres）
  output_dir/logs/wsi_cellpose_*.log     日志

==================== 函数说明 ====================

parse_args() -> argparse.Namespace
  解析上述全部命令行参数，返回 Namespace。无入参，内部定义 19 个 --xxx。

main()
  入口函数。流程：
    1) 解析参数，设置 CUDA_VISIBLE_DEVICES
    2) glob 搜索 WSI，配对 msk_list
    3) 初始化 CellposeModel(model_type/pretrained_model, gpu, use_bfloat16)
       - 自动处理 torch.load(mmap) 兼容
       - 默认 float32 规避 BFloat16 崩溃
    4) 创建 CellposeWSI(model) 并调用 process_wsi_list(...)
  无返回值，异常时抛 FileNotFoundError。

CellposeWSI(model)  # 定义在 cellpose/contrib/wsi_cerberus.py
  .process_single_wsi_direct(wsi_path, output_path, mask_path=None,
        wsi_proc_mag=0.5, tile_shape=4096, ambiguous_size=64,
        flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
        diameter=None, bsize=256, tile_overlap=0.1, nr_post_proc_workers=0,
        save_viz_highres=False, viz_highres_tile=2048, viz_highres_mpp=None, viz_highres_max_tiles=16)
    单张 WSI：4类瓦片生成 -> WSIReader.read_rect(coord_space="resolution") 读瓦片
    -> model.eval -> dynamics 去重 -> joblib.dump(.dat) -> 可选高分可视化。
    返回 out dict。

  .process_wsi_list(wsi_list, output_dir, mask_list=None,
        wsi_proc_mag=..., tile_shape=..., ambiguous_size=...,
        flow_threshold=..., cellprob_threshold=..., min_size=..., diameter=...,
        save_thumb=False, save_mask=False, nr_post_proc_workers=0, logging_dir=None,
        save_viz_highres=False, viz_highres_tile=..., viz_highres_mpp=..., viz_highres_max_tiles=...)
    批量 WSI：建目录 + 日志（logs/wsi_cellpose_*.log + tqdm 双进度条）
    -> 遍历 WSI，已存在 dat 则跳过推理但可补画 viz_highres
    -> 调用 process_single_wsi_direct。

依赖：tiatoolbox, openslide, cellpose, torch, opencv, shapely, joblib, tqdm
环境：conda activate linjiatai_4090  已验证可用
"""
import os, glob, argparse, logging, pathlib
import numpy as np
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.pop("mmap", None)
    return _orig(*a, **kw)
torch.load = _patched

def parse_args():
    p = argparse.ArgumentParser(description="Cellpose WSI (Cerberus-style)")
    p.add_argument("--model", type=str, default="cpsam", help="Cellpose 模型：cpsam/cyto/cyto2/cyto3 或本地路径")
    p.add_argument("--gpu", type=str, default="0", help="GPU id，如 0 或 0,1")
    p.add_argument("--input_dir", type=str, required=True, help="WSI 输入目录")
    p.add_argument("--output_dir", type=str, required=True, help="输出目录")
    p.add_argument("--msk_dir", type=str, default=None, help="组织掩膜目录（同名 png，可选）")
    p.add_argument("--wsi_file_ext", type=str, default=".svs", help="WSI extension, comma-separated e.g. .svs,.tiff or * for all")
    p.add_argument("--tile_shape", type=int, default=4096, help="后处理 tile 大小")
    p.add_argument("--chunk_shape", type=int, default=15000, help="兼容 Cerberus, direct 模式忽略")
    p.add_argument("--ambiguous_size", type=int, default=64, help="模糊边界")
    p.add_argument("--wsi_proc_mag", type=float, default=0.5, help="处理 mpp")
    p.add_argument("--patch_input_shape", type=int, default=448, help="兼容保留")
    p.add_argument("--patch_output_shape", type=int, default=144, help="兼容保留")
    p.add_argument("--diameter", type=float, default=None, help="Cellpose diameter, None 自动 (H&E 在 0.5mpp 建议 15-20)")
    p.add_argument("--flow_threshold", type=float, default=0.4)
    p.add_argument("--cellprob_threshold", type=float, default=0.0)
    p.add_argument("--min_size", type=int, default=15)
    p.add_argument("--save_thumb", action="store_true", help="保存缩略图")
    p.add_argument("--save_mask", action="store_true", help="保存组织掩膜")
    p.add_argument("--save_viz_highres", action="store_true", help="是否生成高分辨率分块可视化 (原图分辨率，切成小图，output_dir/viz_highres/)")
    p.add_argument("--viz_highres_tile", type=int, default=2048, help="高分辨率分块边长 (像素)，默认 2048")
    p.add_argument("--viz_highres_mpp", type=float, default=None, help="高分辨率 mpp，默认用原图 base mpp (约 0.25)，越小越清晰")
    p.add_argument("--viz_highres_max_tiles", type=int, default=16, help="最多生成多少高分辨率小图，-1 不限，默认 16 均匀采样")
    p.add_argument("--save_qupath", action="store_true", help="导出 QuPath GeoJSON (qupath/<basename>.geojson)")
    p.add_argument("--batch_size", type=int, default=8, help="cellpose eval batch")
    p.add_argument("--nr_post_proc_workers", type=int, default=0, help="后处理并行 (direct 模式建议 0)")
    p.add_argument("--use_bfloat16", action="store_true", help="强制 bfloat16 (4090 上会触发 upsample bug，默认 float32)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # support comma-separated multi ext
    raw = (args.wsi_file_ext or ".svs").strip()
    if raw in ("*", "*.*"):
        exts = ["*"]
    else:
        parts = [pp.strip() for pp in raw.replace(";", ",").replace(" ", ",").split(",") if pp.strip()]
        exts=[]
        for pp in parts:
            if pp=="*": exts.append("*")
            elif pp.startswith("*."): exts.append(pp[1:])
            elif pp.startswith("."): exts.append(pp)
            else: exts.append("."+pp.lstrip("."))
        if not exts: exts=[".svs"]
    wsi_files=[]
    for ext in exts:
        pat=f"{args.input_dir}/*{ext}" if ext!="*" else f"{args.input_dir}/*"
        wsi_files.extend(glob.glob(pat))
    wsi_files=sorted(set(wsi_files))
    if not wsi_files and len(exts)==1 and exts[0]!="*":
        for ext in [".svs",".tif",".tiff",".ndpi",".mrxs",".scn","*"]:
            wsi_files = sorted(glob.glob(f"{args.input_dir}/*{ext}"))
            if wsi_files: break
    wsi_files = [f for f in wsi_files if os.path.isfile(f)]
    if not wsi_files:
        raise FileNotFoundError(f"No WSI found in {args.input_dir}")

    mask_list = []
    for wsi in wsi_files:
        stem = pathlib.Path(wsi).stem
        if args.msk_dir and os.path.isfile(os.path.join(args.msk_dir, stem+".png")):
            mask_list.append(os.path.join(args.msk_dir, stem+".png"))
        else:
            mask_list.append(None)
    if args.msk_dir:
        filtered = [(w,m) for w,m in zip(wsi_files, mask_list) if m is not None]
        if filtered:
            wsi_files, mask_list = zip(*filtered)
            wsi_files, mask_list = list(wsi_files), list(mask_list)
        else:
            print(f"[WARN] msk_dir 提供但无匹配 mask, 将处理全部 {len(wsi_files)} 张 WSI")

    print(f"Found {len(wsi_files)} WSIs")

    import torch
    from cellpose.models import CellposeModel
    use_bf16 = bool(args.use_bfloat16)
    if os.path.isdir(args.model) or os.path.isfile(args.model):
        model = CellposeModel(pretrained_model=args.model, gpu=torch.cuda.is_available(), use_bfloat16=use_bf16)
    else:
        try:
            model = CellposeModel(model_type=args.model, gpu=torch.cuda.is_available(), use_bfloat16=use_bf16)
        except TypeError:
            model = CellposeModel(pretrained_model=args.model, gpu=torch.cuda.is_available(), use_bfloat16=use_bf16)

    from cellpose.contrib.wsi_cerberus import CellposeWSI
    wsi_engine = CellposeWSI(model)

    wsi_engine.process_wsi_list(
        wsi_list=wsi_files,
        output_dir=args.output_dir,
        mask_list=mask_list,
        wsi_proc_mag=args.wsi_proc_mag,
        tile_shape=args.tile_shape,
        ambiguous_size=args.ambiguous_size,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        diameter=args.diameter,
        save_thumb=args.save_thumb,
        save_mask=args.save_mask,
        nr_post_proc_workers=args.nr_post_proc_workers,
        save_viz_highres=args.save_viz_highres,
        viz_highres_tile=args.viz_highres_tile,
        viz_highres_mpp=args.viz_highres_mpp,
        viz_highres_max_tiles=args.viz_highres_max_tiles,
        save_qupath=args.save_qupath,
    )
    print("Done.")

if __name__ == "__main__":
    main()
