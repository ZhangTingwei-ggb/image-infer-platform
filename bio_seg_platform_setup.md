# 生物分割平台搭建指南

## 第一阶段：项目初始化与基础结构搭建 (15分钟)

首先，创建项目目录并初始化基础文件。

```bash
# 1. 创建项目根目录
mkdir bio_seg_platform
cd bio_seg_platform

# 2. 创建完整的目录结构
mkdir -p platform/{core,adapters}
mkdir -p models/{cerberus,cellpose}
mkdir -p tests
mkdir -p scripts

# 3. 创建各目录的 __init__.py 文件，使其成为Python包
touch platform/__init__.py platform/core/__init__.py platform/adapters/__init__.py
touch models/__init__.py models/cerberus/__init__.py models/cellpose/__init__.py
touch tests/__init__.py

# 4. 创建主入口文件和基础配置文件
touch cli.py
touch requirements.txt
```

最终项目结构如下：

```
bio_seg_platform/
├── cli.py                     # 平台统一命令行入口
├── requirements.txt           # 平台核心依赖 (轻量级)
├── platform/                  # 平台核心代码
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── loader.py          # 统一图像加载器
│   │   └── exporter.py        # 统一结果导出器
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py            # 适配器抽象基类
│   │   ├── cerberus_adapter.py
│   │   └── cellpose_adapter.py
│   └── scheduler.py           # 任务调度器
├── models/                    # 各模型独立代码与环境
│   ├── cerberus/
│   │   ├── __init__.py
│   │   ├── environment.yml    # Cerberus 的 Conda 环境文件
│   │   └── run_cerberus.py    # Cerberus 处理脚本
│   └── cellpose/
│       ├── __init__.py
│       ├── environment.yml    # Cellpose 的 Conda 环境文件
│       └── run_cellpose.py    # Cellpose 处理脚本
└── tests/
    ├── __init__.py
    └── test_adapters.py
```

## 第二阶段：环境隔离与配置 (30分钟)

这是最关键的一步，确保两个模型在独立的Conda环境中运行，避免依赖冲突。

### 2.1 创建 Cerberus 专用环境

根据 Cerberus 官方仓库的配置说明，创建 cerberus_env 环境。

```bash
# 在 models/cerberus/ 目录下创建 environment.yml
cd models/cerberus/
```

`models/cerberus/environment.yml` 文件内容：

```yaml
name: cerberus_env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.7
  - pip
  - pip:
    - torch==1.10.1+cu102
    - torchvision==0.11.2+cu102
    - opencv-python
    - numpy
    - scikit-image
    - tifffile
    - Pillow
```

创建环境并激活：

```bash
conda env create -f environment.yml
conda activate cerberus_env
```

**注意：** 根据官方文档，Cerberus 依赖 PyTorch 1.10.1，且模型权重需从指定页面下载，并遵守非商业许可证 (Non-commercial License)。

### 2.2 创建 Cellpose 专用环境

Cellpose 的安装较为灵活，可选择较新的 PyTorch 版本。

```bash
cd models/cellpose/
```

`models/cellpose/environment.yml` 文件内容：

```yaml
name: cellpose_env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - cellpose[gui]>=3.0
    - numpy
    - tifffile
```

创建环境并激活：

```bash
conda env create -f environment.yml
conda activate cellpose_env
```

## 第三阶段：实现平台核心组件 (1-2小时)

### 3.1 实现适配器抽象基类 (platform/adapters/base.py)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np


class BaseModelAdapter(ABC):
    """所有模型适配器必须实现的抽象基类"""

    @abstractmethod
    def load_model(self, model_path: Optional[str] = None, **kwargs):
        """加载模型权重和配置"""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        执行模型推理
        Args:
            image: 输入图像 (numpy array, HWC格式)
            **kwargs: 模型特定参数
        Returns:
            统一格式的字典，包含分割结果
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """返回模型基本信息"""
        pass
```

### 3.2 实现 Cerberus 适配器 (platform/adapters/cerberus_adapter.py)

Cerberus 适配器通过子进程调用独立环境中的处理脚本。

```python
import subprocess
import json
import tempfile
import os
import numpy as np
from typing import Dict, Any, Optional
from .base import BaseModelAdapter


class CerberusAdapter(BaseModelAdapter):
    def __init__(self, conda_env_name: str = "cerberus_env"):
        self.conda_env_name = conda_env_name
        self.model_path = None

    def load_model(self, model_path: Optional[str] = None, **kwargs):
        """记录模型路径，实际加载由子进程完成"""
        self.model_path = model_path or kwargs.get('model_path')
        if not self.model_path:
            raise ValueError("Cerberus 需要指定 model_path")
        print(f"[Cerberus] 适配器就绪，使用环境: {self.conda_env_name}")
        return True

    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.model_path is None:
            raise RuntimeError("请先调用 load_model()")

        # 1. 保存图像为临时文件
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp_in:
            import tifffile
            tifffile.imwrite(tmp_in.name, image)
            input_path = tmp_in.name

        output_path = tempfile.mktemp(suffix='.json')

        # 2. 构建子进程命令
        cmd = [
            "conda", "run", "-n", self.conda_env_name,
            "python", "models/cerberus/run_cerberus.py",
            "--input", input_path,
            "--output", output_path,
            "--model", self.model_path
        ]

        # 3. 执行推理
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"[Cerberus] stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"[Cerberus] stderr: {e.stderr}")
            raise RuntimeError(f"Cerberus 推理失败: {e}")

        # 4. 读取结果
        with open(output_path, 'r') as f:
            cerberus_result = json.load(f)

        # 5. 清理临时文件
        os.remove(input_path)
        os.remove(output_path)

        # 6. 返回统一格式
        return {
            "type": "semantic",
            "masks": np.array(cerberus_result.get('masks', [])),
            "classes": cerberus_result.get('classes', []),
            "metadata": {
                "model": "cerberus",
                "model_path": self.model_path,
                "env": self.conda_env_name
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Cerberus",
            "type": "multi_task",
            "supports_wsi": True,
            "tasks": ["gland_seg", "nucleus_seg", "lumen_seg", "tissue_class"]
        }
```

### 3.3 实现 Cellpose 适配器 (platform/adapters/cellpose_adapter.py)

Cellpose 可以直接在当前进程中使用其 Python API。

```python
import numpy as np
from typing import Dict, Any, Optional
from .base import BaseModelAdapter


class CellposeAdapter(BaseModelAdapter):
    def __init__(self, model_type: str = 'cyto3', gpu: bool = True):
        self.model_type = model_type
        self.use_gpu = gpu
        self.model = None

    def load_model(self, model_path: Optional[str] = None, **kwargs):
        """直接加载 Cellpose 模型"""
        from cellpose import models
        self.model = models.CellposeModel(
            gpu=self.use_gpu,
            model_type=self.model_type
        )
        print(f"[Cellpose] 模型 ({self.model_type}) 加载完成")
        return True

    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("请先调用 load_model()")

        # 准备参数
        diameter = kwargs.get('diameter', 30)
        channels = kwargs.get('channels', [0, 0])
        flow_threshold = kwargs.get('flow_threshold', 0.4)
        cellprob_threshold = kwargs.get('cellprob_threshold', 0.0)

        # 执行推理
        masks, flows, styles = self.model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )

        return {
            "type": "instance",
            "masks": masks,
            "flows": flows,
            "metadata": {
                "model": "cellpose",
                "model_type": self.model_type,
                "diameter": diameter,
                "flow_threshold": flow_threshold,
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Cellpose",
            "type": "instance_segmentation",
            "supports_wsi": False,
            "model_types": ["cyto", "cyto2", "cyto3", "nuclei"]
        }
```

### 3.4 实现适配器工厂 (platform/adapters/__init__.py)

```python
from .base import BaseModelAdapter
from .cerberus_adapter import CerberusAdapter
from .cellpose_adapter import CellposeAdapter


class AdapterFactory:
    _adapters = {
        "cerberus": CerberusAdapter,
        "cellpose": CellposeAdapter,
    }

    @classmethod
    def create_adapter(cls, model_type: str, **kwargs) -> BaseModelAdapter:
        adapter_class = cls._adapters.get(model_type.lower())
        if not adapter_class:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return adapter_class(**kwargs)
```

## 第四阶段：实现模型的独立处理脚本 (1-2小时)

### 4.1 Cerberus 处理脚本 (models/cerberus/run_cerberus.py)

这个脚本在 cerberus_env 环境中运行，调用 Cerberus 官方推理逻辑。

```python
#!/usr/bin/env python
"""
Cerberus 推理脚本 - 在 cerberus_env 环境中运行
"""
import argparse
import json
import os
import sys
import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入图像路径')
    parser.add_argument('--output', required=True, help='输出JSON路径')
    parser.add_argument('--model', required=True, help='模型权重目录路径')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # 加载图像
    image = tifffile.imread(args.input)

    # TODO: 调用 Cerberus 官方推理函数
    # 参考: https://github.com/TissueImageAnalytics/cerberus
    # from infer.run_infer_tile import run_inference
    # results = run_inference(image, model_path=args.model, ...)

    # 临时模拟结果 (实际使用时应替换为真实推理)
    print(f"[Cerberus] 处理图像 shape: {image.shape}")
    print(f"[Cerberus] 模型路径: {args.model}")

    # 模拟输出：一个简单的语义掩膜
    dummy_masks = np.zeros(image.shape[:2], dtype=np.int32)
    dummy_classes = [0, 1, 2]  # 示例类别

    # 保存结果
    result = {
        "masks": dummy_masks.tolist(),
        "classes": dummy_classes
    }
    with open(args.output, 'w') as f:
        json.dump(result, f)

    print(f"[Cerberus] 结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
```

**重要提示：** 实际使用时，需要将 TODO 部分替换为 Cerberus 官方的推理代码。你需要将 Cerberus 仓库的 infer、loader、models 等目录复制或软链接到 models/cerberus/ 下。

### 4.2 Cellpose 处理脚本 (models/cellpose/run_cellpose.py)

```python
#!/usr/bin/env python
"""
Cellpose 推理脚本 - 在 cellpose_env 环境中运行
"""
import argparse
import json
import numpy as np
import tifffile
from cellpose import models, io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入图像路径')
    parser.add_argument('--output', required=True, help='输出JSON路径')
    parser.add_argument('--model_type', default='cyto3', choices=['cyto', 'cyto2', 'cyto3', 'nuclei'])
    parser.add_argument('--diameter', type=float, default=30)
    parser.add_argument('--flow_threshold', type=float, default=0.4)
    parser.add_argument('--cellprob_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', action='store_true', default=True)
    args = parser.parse_args()

    # 加载图像
    image = tifffile.imread(args.input)
    print(f"[Cellpose] 处理图像 shape: {image.shape}")

    # 加载模型
    model = models.CellposeModel(gpu=args.gpu, model_type=args.model_type)

    # 执行推理
    masks, flows, styles = model.eval(
        image,
        diameter=args.diameter,
        channels=[0, 0],
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold
    )

    # 保存结果
    result = {
        "masks": masks.tolist(),
        "num_objects": int(np.max(masks)),
        "metadata": {
            "model_type": args.model_type,
            "diameter": args.diameter,
            "flow_threshold": args.flow_threshold,
        }
    }
    with open(args.output, 'w') as f:
        json.dump(result, f)

    print(f"[Cellpose] 检测到 {result['num_objects']} 个对象，结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
```

## 第五阶段：实现调度器与命令行入口 (1小时)

### 5.1 调度器 (platform/scheduler.py)

```python
from typing import Dict, Any, Optional
import numpy as np
from .adapters import AdapterFactory
from .core.loader import load_image
from .core.exporter import export_result


class Scheduler:
    def __init__(self):
        self._adapters = {}

    def run(self, model_type: str, input_path: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """统一任务调度入口"""
        # 1. 加载图像
        image = load_image(input_path)

        # 2. 获取或创建适配器
        if model_type not in self._adapters:
            adapter = AdapterFactory.create_adapter(model_type, **kwargs)
            adapter.load_model(**kwargs)
            self._adapters[model_type] = adapter
        adapter = self._adapters[model_type]

        # 3. 执行推理
        result = adapter.predict(image, **kwargs)

        # 4. 导出结果
        if output_path:
            export_result(result, output_path)

        return result
```

### 5.2 图像加载器 (platform/core/loader.py)

```python
import numpy as np
import tifffile
from PIL import Image
import os


def load_image(path: str) -> np.ndarray:
    """统一图像加载接口，支持多种格式"""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff']:
        return tifffile.imread(path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        return np.array(Image.open(path))
    else:
        # 尝试用 tifffile 作为后备
        return tifffile.imread(path)
```

### 5.3 结果导出器 (platform/core/exporter.py)

```python
import json
import numpy as np
from typing import Dict, Any


def export_result(result: Dict[str, Any], output_path: str):
    """统一结果导出"""
    # 将 numpy 数组转换为列表以便 JSON 序列化
    exportable = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            exportable[key] = value.tolist()
        else:
            exportable[key] = value

    with open(output_path, 'w') as f:
        json.dump(exportable, f, indent=2)
```

### 5.4 命令行入口 (cli.py)

```python
#!/usr/bin/env python
"""
BioSeg Platform - 统一细胞分割平台
"""
import argparse
from platform.scheduler import Scheduler


def main():
    parser = argparse.ArgumentParser(description="BioSeg Platform")
    parser.add_argument('--model', required=True, choices=['cerberus', 'cellpose'],
                        help='选择模型')
    parser.add_argument('--input', required=True, help='输入图像路径')
    parser.add_argument('--output', required=True, help='输出结果路径')
    parser.add_argument('--diameter', type=float, default=30, help='Cellpose: 细胞直径')
    parser.add_argument('--model_path', help='Cerberus: 模型权重目录')
    parser.add_argument('--flow_threshold', type=float, default=0.4, help='Cellpose: flow阈值')

    args = parser.parse_args()

    scheduler = Scheduler()
    result = scheduler.run(
        model_type=args.model,
        input_path=args.input,
        output_path=args.output,
        diameter=args.diameter,
        model_path=args.model_path,
        flow_threshold=args.flow_threshold
    )

    print(f"✅ 处理完成! 结果保存至: {args.output}")
    print(f"📊 结果摘要: {result.get('metadata', {})}")


if __name__ == '__main__':
    main()
```

## 第六阶段：测试 (30分钟)

### 测试命令

```bash
# 测试 Cellpose
python cli.py --model cellpose --input test_image.tif --output cellpose_result.json --diameter 30

# 测试 Cerberus (需要先下载权重)
python cli.py --model cerberus --input test_image.tif --output cerberus_result.json --model_path /path/to/cerberus/weights
```

### 单元测试 (tests/test_adapters.py)

```python
import unittest
import numpy as np
from platform.adapters import AdapterFactory


class TestAdapters(unittest.TestCase):
    def test_cellpose_adapter(self):
        adapter = AdapterFactory.create_adapter('cellpose')
        adapter.load_model()
        dummy_image = np.random.rand(256, 256, 3).astype(np.float32)
        result = adapter.predict(dummy_image)
        self.assertIn('masks', result)
        self.assertEqual(result['type'], 'instance')


if __name__ == '__main__':
    unittest.main()
```

## 第七阶段：扩展与优化建议

完成基础功能后，可以考虑以下优化方向：

1. **添加 WSI 支持**：在 loader.py 中增加对 openslide 的支持，并在 Cerberus 适配器中调用 run_infer_wsi.py。

2. **异步处理**：对于耗时任务，使用 asyncio 或 celery 实现异步调度。

3. **Docker 化**：为每个模型构建 Docker 镜像，替代 Conda 环境，提供更强的隔离性和可移植性。

4. **结果可视化**：增加 --visualize 参数，自动生成分割结果的可视化图片。

5. **批量处理**：支持输入目录，批量处理多张图像。

## 实施检查清单

| 阶段 | 任务 | 状态 |
|------|------|------|
| 第一阶段 | 创建项目目录结构 | ☐ |
| 第二阶段 | 创建 Cerberus Conda 环境 | ☐ |
| 第二阶段 | 创建 Cellpose Conda 环境 | ☐ |
| 第二阶段 | 下载 Cerberus 模型权重 | ☐ |
| 第三阶段 | 实现 BaseModelAdapter | ☐ |
| 第三阶段 | 实现 CerberusAdapter | ☐ |
| 第三阶段 | 实现 CellposeAdapter | ☐ |
| 第三阶段 | 实现 AdapterFactory | ☐ |
| 第四阶段 | 实现 run_cerberus.py | ☐ |
| 第四阶段 | 实现 run_cellpose.py | ☐ |
| 第五阶段 | 实现 Scheduler | ☐ |
| 第五阶段 | 实现 CLI 入口 | ☐ |
| 第六阶段 | 运行测试 | ☐ |

按照这个步骤推进，你应该能在半天内搭建好一个可运行的统一细胞分割平台。
