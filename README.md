# seg_platform — Cerberus + Cellpose 统一 WSI 分割平台

---

## 1. 平台定位与设计原则

| 原则 | 说明 |
|------|------|
| 复制不改源 | `models/cerberus/*` 与 `models/cellpose/*` 均为拷贝，不修改原 `zhang/model/*` 任何文件 |
| 推理后处理全保留 | Cerberus 保留 4 任务推理（Gland/Lumen/Nuclei/Patch-Class）与 WSI 后处理；Cellpose 保留 `CellposeWSI` 的 Direct WSI 后处理（4 类瓦片 + STRtree 去重） |
| 权重保留 | Cerberus `pretrained_weights/resnet34_cerberus/weights.tar + settings.yml` 已拷贝到 `models/cerberus/pretrained_weights/` |
| 统一可视化 | 高分辨率分块 `viz_highres` 四件套在两模型上都生效（Cerberus 原生无此功能，平台新增） |
| 单入口分流 | `python cli.py --model cerberus|cellpose` 选模型，其余参数透传 |

---

## 2. 目录结构

```
seg_platform/
├── cli.py                         # 统一入口，--model 分流
├── README.md                      # 本文档
├── requirements.txt               # 轻量依赖说明
├── bio_seg_platform_setup.md      # 原方案草案（参考）
├── platform/
│   ├── core/
│   │   ├── loader.py              # 组织掩膜加载（可选）
│   │   └── exporter.py            # save_viz_highres_from_dat：原图分辨率切小图
│   └── adapters/
│       ├── base.py                # BaseAdapter 抽象
│       ├── cerberus_adapter.py    # 封装 InferManager，追加高分可视化
│       └── cellpose_adapter.py    # 封装 CellposeWSI
├── models/
│   ├── cerberus/                  # 拷贝自 model/cerberus（不含 dataset/cache/log）
│   │   ├── infer/{base,patch,tile,wsi}.py
│   │   ├── loader/{augs,infer_loader,postproc,targets}.py
│   │   ├── models/{net_desc,run_desc,backbone,utils}
│   │   ├── misc/{utils,wsi_handler,viz_utils}
│   │   ├── pretrained_weights/resnet34_cerberus/{weights.tar,settings.yml}
│   │   ├── dataset.yml / environment.yml / run_infer_wsi.py
│   │   └── README.md
│   └── cellpose/                  # 拷贝自 model/cellpose
│       ├── run_cellpose.py        # 原 run_wsi_cellpose.py
│       ├── wsi_cerberus.py        # 原 cellpose/contrib/wsi_cerberus.py
│       └── environment.yml
├── scripts/
│   └── inspect_dat.py             # 单张 dat 可视化复核（缩略图/区域）
└── tests/                         # 预留
```

---

## 3. 环境

推荐 `linjiatai_4090`（已验证 `tiatoolbox 1.4 + openslide + cellpose 3.x + torch 2.x` 共存）：

```bash
conda activate linjiatai_4090
python -c "import tiatoolbox, openslide, cellpose, torch; print('ok')"
```

`requirements.txt` 仅作说明，不强制新建 env；Cerberus 官方 `environment.yml`（python 3.7 + torch 1.10）已过时，单 env 更省事，后续再 Docker 化也不迟。权重遵守非商业许可，需本地已有。

---

## 4. 参数总表（是否为全部参数？）

### 4.1 结论

| 模型 | 原脚本全部参数 | 平台 cli 是否覆盖 | 备注 |
|------|---------------|------------------|------|
| Cerberus `run_infer_wsi.py` | 15 个 | 覆盖 13 个 | 未暴露 `wsi_bulk_idx / wsi_proc_step`（多机分片专用，单机不需要；如需可追加） |
| Cellpose `run_wsi_cellpose.py` | 19 个 | 全部 19 个 | 含 `chunk_shape/patch_*` 兼容保留 |

平台另新增统一的 4 个高分可视化参数，两模型通用。

### 4.2 通用参数（两模型对齐）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--model` | `cerberus\|cellpose` | 必填 | 选模型 |
| `--input_dir` | PATH | 必填 | WSI 目录，非递归 |
| `--output_dir` | PATH | 必填 | 输出根，自动建 `dat/thumb/mask/viz_highres/logs[/cache]` |
| `--msk_dir` | PATH | `None` | 组织掩膜目录（同名 `.png`）；若提供则只处理有掩膜的 WSI |
| `--wsi_file_ext` | STR | `.svs` | WSI 后缀；找不到时自动遍历 `.tif/.tiff/.ndpi/.mrxs/.scn` |
| `--gpu` | STR | `0` | `0` / `0,1` / `""`（CPU）；写入 `CUDA_VISIBLE_DEVICES` |
| `--wsi_proc_mag` | FLOAT | `0.5` | 处理 mpp：`0.5=20x, 0.25=40x`；无 mpp 的 TIFF 自动 fallback 到 level 0 |
| `--tile_shape` | INT | `4096` | 核后处理 tile 边长；显存紧张改 `2048` |
| `--ambiguous_size` | INT | `64` | 模糊带宽（margin），4 类瓦片去重用 |
| `--patch_input_shape` | INT | `448` | 兼容 Cerberus，Cellpose direct 模式忽略 |
| `--patch_output_shape` | INT | `144` | 同上 |
| `--chunk_shape` | INT | `15000` | Cerberus 推理大瓦片；Cellpose direct 模式忽略 |
| `--batch_size` | INT | `30` | Cerberus 默认 30，Cellpose 原默认 8，平台统一 30 |
| `--nr_post_proc_workers` | INT | `0` | 后处理并行；direct 模式建议 `0`（WSIReader 不可 pickle） |
| `--save_thumb` | flag | - | 存缩略图 `output_dir/thumb/<basename>.png` |
| `--save_mask` | flag | - | 存掩膜 `output_dir/mask/<basename>.png` |

### 4.3 高分辨率分块可视化（两模型通用，平台新增）

> 默认不画，用参数控制；原图分辨率切小图，放大看边界。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--save_viz_highres` | - | 开启高分分块 |
| `--viz_highres_tile` | `2048` | 每张小图边长；`1024` 更细，`4096` 更大视野 |
| `--viz_highres_mpp` | `None=base mpp(~0.261)` | 高分 mpp，越小越清晰；`0.20` 更细，`0.5` 与处理一致 |
| `--viz_highres_max_tiles` | `16` | 最多几张，均匀采样；`-1` 生成全部（大 WSI 会上千张） |

实现（`exporter.save_viz_highres_from_dat`）：
`scale = wsi_proc_mag / viz_mpp` 将 `contour` 从 0.5 mpp 缩放到 `viz_mpp`；`WSIReader.read_rect(..., resolution=viz_mpp, units='mpp', coord_space='resolution')` 读原图块；跳过无组织块（`wsi_mask` 全 0）；绿框画 `Nuclei`（Cerberus 额外画 Gland/Lumen 蓝/绿区分）；输出 `viz_highres/<basename>/<basename>_x<x>_y<y>_w<w>_h<h>.png`；`Skip existing` 时可补画（见下）。

### 4.4 Cellpose 专属

| 参数 | 默认 | 说明 |
|------|------|------|
| `--cellpose_model` | `cpsam` | `cpsam/cyto/cyto2/cyto3` 或本地路径 |
| `--diameter` | `None` | 自动估计；H&E 在 0.5 mpp 建议 `15-20`，`30` 偏大易粘连 |
| `--flow_threshold` | `0.4` | 越大检出越多 |
| `--cellprob_threshold` | `0.0` | 越小检出越多 |
| `--min_size` | `15` | 最小面积过滤 |
| `--use_bfloat16` | - | 强制 bfloat16；4090 上会触发 `upsample_linear1d BFloat16` 崩溃，默认 `float32` |

### 4.5 Cerberus 专属

| 参数 | 默认 | 说明 |
|------|------|------|
| `--cerberus_model` | `models/cerberus/pretrained_weights/resnet34_cerberus` | 权重目录 |
| `--cache_path` | `output_dir/cache` | 推理缓存（raw memmap），需 SSD 100GB+ |
| `--logging_dir` | `output_dir/logs` | 日志目录 |
| `--nr_inference_workers` | `0` | 推理 DataLoader 并行 |

未暴露但可按需加回：`--wsi_bulk_idx` / `--wsi_proc_step`（Cerberus 原用于多机分片，`proc_list = wsi_list[(idx-1)*step : idx*step]`）；单机批量已够用。

---

## 5. 日志与进度条（有吗？）

**有，且两模型都双输出（控制台 + 文件）+ 双层 tqdm。**

### Cellpose 路径

- 代码：`models/cellpose/wsi_cerberus.py` 的 `CellposeWSI.process_wsi_list`
- 日志文件：`output_dir/logs/wsi_cellpose_YYYYMMDD_HHMMSS.log`（`FileHandler` + `StreamHandler`，格式 `%(asctime)s | %(levelname)s | %(message)s`）
- 内容：`WSI X/Y 1/5 40%` 外层；内层每 set `PostProc grid (1/4): 4/16 [instances=1234, tile=4/16]`；`Tile set 0 (grid) done, current instances 12345`；`Saved N nuclei to dat/... time Xs`；高分 `Viz highres: 16 tiles, viz WxH @ mpp, tile T` / `Saved 16 highres viz tiles to ...`
- 屏蔽：`torch sparse` 警告已 `warnings.filterwarnings("ignore")`，`read_rect` 的 `dict→float` 与 `coord_space` 兼容已修复（错位读背景的 248 vs 243 问题已解决）

### Cerberus 路径

- 代码：`models/cerberus/infer/wsi.py` 的 `InferManager.process_wsi_list`
- 日志文件：`logging_dir/<basename>_<dt>_std.log`（每 WSI 一个，`FileHandler` DEBUG 级）
- 进度：`tqdm Process Batch: X/Y`；平台适配器在后处追加高分可视化的 `tqdm Viz highres` 与 `Saved ... highres viz tiles`
- 额外：`output_dir/cache/` 下的 `raw.npy/count.npy` memmap（大 WSI 可能 70GB+）

### 日志查看

```bash
ls -lh output_dir/logs/
tail -f output_dir/logs/wsi_cellpose_*.log
# 或
tail -f output_dir/logs/*_std.log
```

---

## 6. 输出结构（1:1 兼容 Cerberus，便于 inspect 通用）

```
output_dir/
├── dat/<basename>.dat   # joblib.dump {Nuclei:{uuid:{box,contour,centroid}}, Gland?, Lumen?, proc_resolution, base_resolution, proc_dimensions, base_dimensions}
├── thumb/<basename>.png # --save_thumb
├── mask/<basename>.png  # --save_mask
├── viz_highres/<basename>/*.png  # --save_viz_highres，高分小图
├── logs/                # Cellpose: wsi_cellpose_*.log；Cerberus: *_std.log
└── cache/               # 仅 Cerberus
```

`dat` 结构示例（`joblib.load`）：

```python
{
  "Nuclei": {"a3f...": {"box": np(2,2), "contour": np(N,2), "centroid": np(2)}, ...}, # XY contour, YX box
  "Gland":  {...},  # 仅 Cerberus
  "Lumen":  {...},  # 仅 Cerberus
  "proc_resolution": {"resolution": 0.5, "units": "mpp"},
  "base_resolution": {"resolution": 0.261, "units": "mpp"},
  "proc_dimensions": np([H,W]),   # YX
  "base_dimensions": np([H,W]),
}
```

`contour` 为 XY，`box` 为 `[[rmin,cmin],[rmax,cmax]]` YX，`centroid` 为 XY；`inspect_dat.py` / `exporter` 均按此解析。

---

## 7. WSI 处理流程（分开处理再整合）

```
WSI (.svs 95k×71k) ── WSIReader.slide_dimensions(mpp) ─┬─ Cerberus: chunk 15000 ─ PatchExtractor(448/144) ─ merge_prediction ─ memmap raw.npy
                                                       └─ Cellpose: tile 4096 直接读图 ─ model.eval(dP+cellprob) ─ dynamics

4 类瓦片生成（_get_tile_info 1:1 复刻 tiatoolbox 1.4）：
  grid（常规）→ vertical strip（夹缝）→ horizontal strip → cross（十字） + flag[top,bottom,left,right]
  │
  ├─ margin 去重（shapely STRtree）：
  │    mode 0/3: 删除完全落在 margin 内的 bbox（predicate="contains"）
  │    mode 1/2: 删除触碰 margin/boundary 的
  │    mode 3: 额外删除已累积全局字典中与 margin_lines 相交的旧实例
  └─ 坐标平移 + uuid → 全局 dict → joblib.dump .dat

可选：save_viz_highres → scale=0.5/viz_mpp → read_rect 原图分辨率 → 绿框 viz_highres/
```

---

## 8. 快速开始

```bash
conda activate linjiatai_4090
cd /media/linjiatai/086399513677BED7/zhang/seg_platform
python cli.py --help
```

### 8.1 Cellpose

```bash
# 仅推理
python cli.py --model cellpose --input_dir /media/linjiatai/086399513677BED7/zhang/dataset/test \
  --output_dir /media/linjiatai/086399513677BED7/zhang/dataset/test_out \
  --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5 --diameter 20 --gpu 0

# 推理 + 高分分块（推荐）
python cli.py --model cellpose --input_dir .../test --output_dir .../test_out \
  --tile_shape 4096 --diameter 20 --gpu 0 \
  --save_thumb --save_mask --save_viz_highres --viz_highres_tile 2048 --viz_highres_max_tiles 16

# 更细腻
python cli.py --model cellpose --input_dir ... --output_dir ... --diameter 15 \
  --save_viz_highres --viz_highres_tile 1024 --viz_highres_mpp 0.20 --viz_highres_max_tiles 36

# 已有 dat 补画高分图（不重推理）
python cli.py --model cellpose --input_dir .../test --output_dir .../test_out \
  --save_viz_highres --viz_highres_tile 2048 --viz_highres_max_tiles 4
```

### 8.2 Cerberus

```bash
python cli.py --model cerberus --input_dir /path/wsi --output_dir /path/out_cerberus \
  --tile_shape 4096 --chunk_shape 15000 --wsi_proc_mag 0.5 --batch_size 30 --gpu 0 \
  --save_thumb --save_mask

# Cerberus + 高分可视化（平台新增）
python cli.py --model cerberus --input_dir ... --output_dir ... \
  --save_viz_highres --viz_highres_tile 2048 --viz_highres_max_tiles 16

# 指定权重 / cache / 日志
python cli.py --model cerberus --input_dir ... --output_dir ... \
  --cerberus_model models/cerberus/pretrained_weights/resnet34_cerberus \
  --cache_path /ssd/cache --logging_dir /path/out/logs --nr_inference_workers 4
```

### 8.3 单张复核（inspect_dat）

```bash
python scripts/inspect_dat.py --input /path/out/dat --wsi_dir /path/wsi --output /tmp/viz --viz --max_inst 200
# 区域高清
python scripts/inspect_dat.py --input .../dat --wsi_dir ... --output /tmp/viz --viz --region 10000,20000,2048,2048 --max_inst -1
```

---

## 9. 常见问题

- **显存不足**：`--tile_shape 4096 → 2048`；Cerberus 另可 `--batch_size 30 → 16`；Cellpose `diameter` 过大（30）易粘连，建议 `15-20` 或 `None` 自动。
- **4090 bfloat16 崩溃**：`RuntimeError: upsample_linear1d BFloat16` 为已知 bug，平台默认 `float32`，勿加 `--use_bfloat16`。
- **读图错位（白背景 248 均值）**：已修复为 `read_rect(..., resolution=float, coord_space="resolution")`；旧 `dat` 需 `rm -rf output_dir/dat/<basename>.dat` 后重跑。
- **二次运行不补画**：`Skip existing` 时若 `viz_highres/<basename>/` 不存在会自动补画高分图，无需删 `dat`。
- **空格文件名**：`2519980 ER.svs` 这类需引号或转义，平台内部已用 `pathlib.Path.stem` 处理。
- **cache 很大**：Cerberus `cache/raw.npy` 可能 70GB+，务必放 SSD；Cellpose direct 模式无此开销。

---

## 10. 依赖与引用

- Cerberus：`infer/wsi.py 1036 行 + tile.py + base.py + loader/postproc.py + models/* + misc/*`（已拷贝）
- Cellpose：`run_wsi_cellpose.py 211 行 + cellpose/contrib/wsi_cerberus.py 850 行`（已拷贝，含 `read_rect` 与 `bfloat16` 修复）
- 工具：`tiatoolbox 1.4 / openslide-python / shapely / opencv / joblib / tqdm / PyYAML`

---

## 11. 图形界面（GUI）

> `gui.py` 为 Tkinter 桌面界面，复制不改源，底层调用 `cli.py`；无需额外依赖（`tkinter 8.6` 已验证）。

```bash
conda activate linjiatai_4090
cd /media/linjiatai/086399513677BED7/zhang/seg_platform
python gui.py
```

界面：
- 顶部：模型切换 `Cellpose / Cerberus`，参数显式透传
- 左侧卡片（可滚动）：输入/输出/掩膜目录（浏览按钮）→ WSI/后处理通用 → 高分可视化四件套 → Cellpose/Cerberus 专属参数（随模型切换显隐）
- 右侧：实时日志（黑底等宽）+ 不定进度条 + 状态；`▶ 开始推理` 后台线程跑 `cli.py`，`■ 停止` 可终止；`打开输出 / 预览高分图 / 清空日志`

运行本质等价于手写 `cli.py` 命令，日志与进度与命令行一致。

---

*最后更新：2026-08-27；平台路径 `/media/linjiatai/086399513677BED7/zhang/seg_platform`；原模型路径 `zhang/model/cerberus` 与 `zhang/model/cellpose` 未改动；GUI `gui.py 496 行`。*
