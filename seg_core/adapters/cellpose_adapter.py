"""Cellpose 适配器 — 复用 seg_platform/models/cellpose/wsi_cerberus.py 的 CellposeWSI"""
import os
import torch
from .base import BaseAdapter

class CellposeAdapter(BaseAdapter):
    def __init__(self, model_name="cpsam", gpu="0", use_bfloat16=False):
        self.model_name = model_name
        self.gpu = gpu
        self.use_bfloat16 = use_bfloat16
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
        use_bf16 = bool(self.use_bfloat16)
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
        # 透传所有 WSI 参数
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
            save_viz_highres=kwargs.get("save_viz_highres", False),
            viz_highres_tile=kwargs.get("viz_highres_tile", 2048),
            viz_highres_mpp=kwargs.get("viz_highres_mpp"),
            viz_highres_max_tiles=kwargs.get("viz_highres_max_tiles", 16),
            save_qupath=kwargs.get("save_qupath", False),
        )

    def get_info(self):
        return {"name": "Cellpose", "model": self.model_name, "supports_wsi": True}
