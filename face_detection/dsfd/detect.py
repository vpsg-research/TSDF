import torch
import numpy as np
import typing
from .face_ssd import SSD
from .config import resnet152_model_config
from .. import torch_utils
from ..base import Detector
from ..build import DETECTOR_REGISTRY

# 默认模型权重路径
default_model_path = "/home/zhr-23/code/TSDF_code/dsfd_vgg_0.880.pth"

@DETECTOR_REGISTRY.register_module
class DSFDDetector(Detector):

    def __init__(self, *args, model_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        # 使用传入的模型路径或默认路径
        model_path = model_path if model_path is not None else default_model_path
        
        print(f"正在加载DSFD模型权重: {model_path}")
        
        # 加载模型权重
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 初始化 SSD 网络
            self.net = SSD(resnet152_model_config)
            self.net.load_state_dict(state_dict)
            self.net.eval()
            self.net = self.net.to(self.device)
            print("DSFD模型权重加载成功!")
        except Exception as e:
            print(f"加载DSFD模型权重出错: {e}")
            # 初始化网络但不加载权重
            self.net = SSD(resnet152_model_config)
            self.net.eval()
            self.net = self.net.to(self.device)
            print("使用未初始化的DSFD模型继续...")

    @torch.no_grad()
    def _detect(self, x: torch.Tensor) -> typing.List[np.ndarray]:
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        # Expects BGR
        x = x[:, [2, 1, 0], :, :]
        # 修改为
        with torch.amp.autocast('cuda', enabled=self.fp16_inference):
            boxes = self.net(
                x, self.confidence_threshold, self.nms_iou_threshold
            )
        return boxes
