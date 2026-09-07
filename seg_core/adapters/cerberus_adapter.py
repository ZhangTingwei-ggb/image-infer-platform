"""Cerberus 适配器 — 复用 seg_platform/models/cerberus 内的 InferManager"""
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

        # QuPath GeoJSON — validated by QuPath engine, directly draggable
        if bool(kwargs.get("save_qupath", False)):
            from seg_core.core.exporter import save_qupath_geojson_from_dat
            qupath_split = kwargs.get("qupath_split", None)
            if qupath_split == 0: qupath_split = None
            qupath_no_clean = bool(kwargs.get("qupath_no_clean", False))
            for wsi_path, mask_path in zip(wsi_list, mask_list or [None]*len(wsi_list)):
                basename = pathlib.Path(wsi_path).stem
                dat_path = os.path.join(output_dir, "dat", f"{basename}.dat")
                if not os.path.isfile(dat_path):
                    alt = glob.glob(os.path.join(output_dir, "dat", basename + ".*"))
                    if alt:
                        dat_path = alt[0]
                    else:
                        continue
                # skip if exists (exporter handles overwrite/clean)
                try:
                    save_qupath_geojson_from_dat(dat_path, output_dir, fix_invalid=not qupath_no_clean, split_large=qupath_split, use_qpath_engine=not qupath_no_clean)
                except Exception as e:
                    print(f"[cerberus qupath] {basename} failed: {e}")

    def get_info(self):
        return {"name": "Cerberus", "model_dir": self.model_dir, "supports_wsi": True}
