"""Cellpose 适配器 — 复用 seg_platform/models/cellpose/wsi_cerberus.py 的 CellposeWSI"""
import os
import torch
from .base import BaseAdapter

class CellposeAdapter(BaseAdapter):
    def __init__(self, model_name="cpsam", gpu="0"):
        self.model_name = model_name
        self.gpu = gpu
        self._engine = None
        self._init_model()

    def _init_model(self):
        if self.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        # torch.load 兼容
        _orig = torch.load
        def _patched(*a, **kw):
            kw.pop("mmap", None)
            return _orig(*a, **kw)
        torch.load = _patched
        from cellpose.models import CellposeModel
        use_bf16 = False
        if os.path.isdir(self.model_name) or os.path.isfile(self.model_name):
            self.model = CellposeModel(pretrained_model=self.model_name, gpu=torch.cuda.is_available(), use_bfloat16=use_bf16)
        else:
            try:
                self.model = CellposeModel(model_type=self.model_name, gpu=torch.cuda.is_available(), use_bfloat16=use_bf16)
            except TypeError:
                self.model = CellposeModel(pretrained_model=self.model_name, gpu=torch.cuda.is_available(), use_bfloat16=use_bf16)
        # 延迟导入，避免顶层失败
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../models/cellpose"))
        from wsi_cerberus import CellposeWSI
        self._engine = CellposeWSI(self.model)

    def run(self, wsi_list, mask_list, output_dir, **kwargs):
        # passthrough all WSI params including QuPath export
        self._engine.process_wsi_list(
            wsi_list=wsi_list, output_dir=output_dir, mask_list=mask_list,
            wsi_proc_mag=kwargs.get("wsi_proc_mag", 0.5),
            tile_shape=kwargs.get("tile_shape", 4096),
            ambiguous_size=kwargs.get("ambiguous_size", 64),
            flow_threshold=kwargs.get("flow_threshold", 0.4),
            cellprob_threshold=kwargs.get("cellprob_threshold", 0.0),
            min_size=kwargs.get("min_size", 15),
            diameter=kwargs.get("diameter"),
            save_thumb=kwargs.get("save_thumb", False),
            save_mask=kwargs.get("save_mask", False),
            nr_post_proc_workers=kwargs.get("nr_post_proc_workers", 0),
            save_qupath=kwargs.get("save_qupath", False),
            qupath_split=kwargs.get("qupath_split", None),
            qupath_no_clean=kwargs.get("qupath_no_clean", False),
            chunk_shape=kwargs.get("chunk_shape", 6000),
            patch_input_shape=kwargs.get("patch_input_shape", 512),
            patch_output_shape=kwargs.get("patch_output_shape", 512),
            cache_dir=kwargs.get("cache_path", kwargs.get("cache_dir", None)),
            use_cerberus_infer=kwargs.get("use_cerberus_infer", False),
        )

    def get_info(self):
        return {"name": "Cellpose", "model": self.model_name, "supports_wsi": True}
