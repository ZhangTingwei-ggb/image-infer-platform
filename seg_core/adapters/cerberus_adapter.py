"""Cerberus 适配器 — 复用 seg_platform/models/cerberus 内的 InferManager，增加高分可视化"""
import os
import sys
import glob
import pathlib
import yaml
import numpy as np
import joblib

# 将 models/cerberus 加入 path 以便 import infer.* / models.* / loader.* / misc.*
_CERBERUS_ROOT = os.path.join(os.path.dirname(__file__), "../../models/cerberus")
_CERBERUS_ROOT = os.path.abspath(_CERBERUS_ROOT)
if _CERBERUS_ROOT not in sys.path:
    sys.path.insert(0, _CERBERUS_ROOT)

from .base import BaseAdapter

class CerberusAdapter(BaseAdapter):
    def __init__(self, model_dir=None, gpu="0"):
        self.model_dir = model_dir or os.path.join(_CERBERUS_ROOT, "pretrained_weights/resnet34_cerberus")
        self.gpu = gpu
        self._infer = None

    def _prepare_infer(self):
        if self.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        checkpoint_path = os.path.join(self.model_dir, "weights.tar")
        settings_path = os.path.join(self.model_dir, "settings.yml")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"cerberus weights not found: {checkpoint_path}")
        with open(settings_path) as f:
            run_paramset = yaml.safe_load(f)
        from infer.wsi import InferManager
        self._infer = InferManager(
            checkpoint_path=checkpoint_path,
            decoder_dict=run_paramset["dataset_kwargs"]["req_target_code"],
            model_args=run_paramset["model_kwargs"],
        )
        self._run_paramset = run_paramset

    def run(self, wsi_list, mask_list, output_dir, **kwargs):
        if self._infer is None:
            self._prepare_infer()

        # 兼容部分参数缺省，使用 cerberus 默认
        wsi_proc_mag = float(kwargs.get("wsi_proc_mag", 0.5))
        tile_shape = int(kwargs.get("tile_shape", 4096))
        chunk_shape = int(kwargs.get("chunk_shape", 15000))
        ambiguous_size = int(kwargs.get("ambiguous_size", 64))
        patch_input_shape = int(kwargs.get("patch_input_shape", 448))
        patch_output_shape = int(kwargs.get("patch_output_shape", 144))
        nr_inference_workers = int(kwargs.get("nr_inference_workers", 0))
        nr_post_proc_workers = int(kwargs.get("nr_post_proc_workers", 0))
        batch_size = int(kwargs.get("batch_size", 30))
        cache_path = kwargs.get("cache_path", os.path.join(output_dir, "cache"))
        save_thumb = bool(kwargs.get("save_thumb", False))
        save_mask = bool(kwargs.get("save_mask", False))
        # 高分可视化（cerberus 原生无此功能，平台新增）
        save_viz_highres = bool(kwargs.get("save_viz_highres", False))
        viz_highres_tile = int(kwargs.get("viz_highres_tile", 2048))
        viz_highres_mpp = kwargs.get("viz_highres_mpp", None)
        viz_highres_max_tiles = int(kwargs.get("viz_highres_max_tiles", 16))

        cache_path = f"{cache_path.rstrip('/')}/"
        os.makedirs(cache_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        target_list = ["gland", "lumen", "nuclei", "patch-class"]
        run_args = {
            "nr_inference_workers": nr_inference_workers,
            "nr_post_proc_workers": nr_post_proc_workers,
            "batch_size": batch_size,
            "input_list": wsi_list,
            "mask_list": mask_list,
            "output_dir": output_dir,
            "patch_input_shape": patch_input_shape,
            "patch_output_shape": patch_output_shape,
            "save_thumb": save_thumb,
            "save_mask": save_mask,
            "mask_dir": kwargs.get("msk_dir"),
            "postproc_list": target_list,
            "msk_dir": kwargs.get("msk_dir"),
            "tile_shape": tile_shape,
            "chunk_shape": chunk_shape,
            "ambiguous_size": ambiguous_size,
            "cache_path": cache_path,
            "logging_dir": kwargs.get("logging_dir", os.path.join(output_dir, "logs")),
            "wsi_proc_mag": wsi_proc_mag,
        }
        self._infer.process_wsi_list(run_args)

        # 后处：高分可视化（已有的 dat 补画，不重推理）
        if save_viz_highres:
            from seg_core.core.exporter import save_viz_highres_from_dat
            # cerberus 的 logger 在 InferManager 内部，这里复用 print
            for wsi_path, mask_path in zip(wsi_list, mask_list or [None]*len(wsi_list)):
                basename = pathlib.Path(wsi_path).stem
                dat_path = os.path.join(output_dir, "dat", f"{basename}.dat")
                if not os.path.isfile(dat_path):
                    # 兼容 cerberus 旧版可能直接以 stem 命名
                    alt = glob.glob(os.path.join(output_dir, "dat", basename + ".*"))
                    if alt:
                        dat_path = alt[0]
                    else:
                        continue
                # 若已存在 viz_highres 则跳过，避免重复生成
                hr_root = os.path.join(output_dir, "viz_highres", basename)
                if os.path.isdir(hr_root) and len(os.listdir(hr_root)) > 0:
                    continue
                try:
                    save_viz_highres_from_dat(
                        dat_path, wsi_path, output_dir,
                        wsi_proc_mag=wsi_proc_mag,
                        viz_highres_tile=viz_highres_tile,
                        viz_highres_mpp=viz_highres_mpp,
                        viz_highres_max_tiles=viz_highres_max_tiles,
                        mask_path=mask_path,
                    )
                except Exception as e:
                    print(f"[cerberus viz_highres] {basename} failed: {e}")

    def get_info(self):
        return {"name": "Cerberus", "model_dir": self.model_dir, "supports_wsi": True}
