from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseAdapter(ABC):
    @abstractmethod
    def run(self, wsi_list: List[str], mask_list: List[Optional[str]], output_dir: str, **kwargs) -> None:
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass
