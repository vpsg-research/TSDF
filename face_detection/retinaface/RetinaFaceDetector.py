import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import typing
from face_detection.retinaface.models.retinaface import RetinaFace
from face_detection.box_utils import batched_decode
from face_detection.retinaface.utils import decode_landm
from face_detection.retinaface.config import cfg_mnet
from face_detection.retinaface.prior_box import PriorBox
from torchvision.ops import nms

class RetinaFaceDetector(nn.Module):
    
    def __init__(self,
                 confidence_threshold=0.5,
                 nms_iou_threshold=0.3,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 model_type="mobilenet",
                 fp16_inference=False,
                 clip_boxes=False):
        super(RetinaFaceDetector, self).__init__()
        
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.fp16_inference = fp16_inference
        self.clip_boxes = clip_boxes
        
        # 选择模型配置和权重
        if model_type == "mobilenet":
            self.cfg = cfg_mnet
            state_dict = torch.load(
                "/home/zhr-23/code/TSDF_code/face_detection/retinaface/weights/mobilenet0.25_Final.pth",
                map_location=device
            )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        self.net = RetinaFace(cfg=self.cfg)
        self.net.eval()
        self.net.load_state_dict(state_dict)
        self.net = self.net.to(device)
        
        # 预处理参数
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.prior_box_cache = {}        
    def get_features(self, image):
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            # 预处理
            image = image.detach().cpu().numpy()
            image = np.transpose(image, (0, 2, 3, 1))  # BCHW -> BHWC
            image = image.astype(np.float32)
            image = (image * 0.5 + 0.5) * 255  # [-1,1] -> [0,255]
            image = image - self.mean
            image = np.transpose(image, (0, 3, 1, 2))  # BHWC -> BCHW
            image = torch.from_numpy(image).to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16_inference):
                    features = self.net.get_features(image)
            return features
        else:
            raise TypeError("must be tensor")
            
    @torch.no_grad()
    def _detect(self, image: torch.Tensor, return_landmarks=False):
        
        image = image[:, [2, 1, 0]]  # RGB -> BGR
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            loc, conf, landms = self.net(image) 
            
            face_probs = torch.sigmoid(conf[:, :, 1]) 
            scores_for_concat = face_probs.unsqueeze(-1) 
            
            height, width = image.shape[2:]
            if (height, width) in self.prior_box_cache:
                priors = self.prior_box_cache[(height, width)]
            else:
                priorbox = PriorBox(self.cfg, image_size=(height, width))
                priors = priorbox.forward()
                self.prior_box_cache[(height, width)] = priors
            priors = priors.to(self.device)
            
            decoded_boxes_coords = batched_decode(loc, priors.data, self.cfg['variance'])
            boxes = torch.cat((decoded_boxes_coords, scores_for_concat), dim=-1) 
            
            if return_landmarks:
                landms = decode_landm(landms, priors.data, self.cfg['variance'])
                return boxes, landms
            return boxes
            
    def detect(self, image):
       
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            batch_size = image.size(0)
            image = image.detach().cpu().numpy()
            image = np.transpose(image, (0, 2, 3, 1))  # BCHW -> BHWC
            image = (image * 0.5 + 0.5) * 255  # [-1,1] -> [0,255]
            image = image.astype(np.float32) - self.mean
            image = np.transpose(image, (0, 3, 1, 2))  # BHWC -> BCHW
            image = torch.from_numpy(image).to(self.device)
            
            boxes = self._detect(image)
            
            batch_results = []
            for i in range(batch_size):
                boxes_i = boxes[i]
                scores_i = boxes_i[:, -1]
                
                mask = scores_i >= self.confidence_threshold
                boxes_i = boxes_i[mask]
                scores_i = scores_i[mask]
                
                if len(boxes_i) > 0:
                    keep_idx = nms(
                        boxes_i[:, :4], scores_i, self.nms_iou_threshold)
                    boxes_i = boxes_i[keep_idx]
                    
                    if self.clip_boxes:
                        boxes_i[:, :4] = boxes_i[:, :4].clamp(0, 1)
                    
                    boxes_i[:, [0, 2]] *= image.shape[3]
                    boxes_i[:, [1, 3]] *= image.shape[2]
                    
                    batch_results.append({
                        'boxes': boxes_i[:, :4].to(self.device),
                        'scores': boxes_i[:, 4].to(self.device)
                    })
                else:
                    batch_results.append({
                        'boxes': torch.empty((0, 4), device=self.device),
                        'scores': torch.empty(0, device=self.device)
                    })
                    
            return batch_results
        else:
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(0)
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                image = (image.float() - 127.5) / 127.5
                
            return self.detect(image)[0]
            
    def forward(self, image):

        return self.detect(image)
            
    def get_raw_face_probabilities(self, image_tensor_rgb_minus1_1: torch.Tensor):
        
        if not isinstance(image_tensor_rgb_minus1_1, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")

        if image_tensor_rgb_minus1_1.dim() == 3:
            image_tensor_rgb_minus1_1 = image_tensor_rgb_minus1_1.unsqueeze(0)

        if image_tensor_rgb_minus1_1.device != self.device:
            image_tensor_rgb_minus1_1 = image_tensor_rgb_minus1_1.to(self.device)
            
        img = image_tensor_rgb_minus1_1.float() 
        img = (img * 0.5 + 0.5) * 255
        mean_tensor = torch.tensor(self.mean, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        img_mean_sub = img - mean_tensor
        processed_tensor_for_net_input = img_mean_sub[:, [2, 1, 0], :, :]

        _loc, conf, _landms = self.net(processed_tensor_for_net_input) 
        face_probs = torch.sigmoid(conf[:, :, 1]) 
        
        return face_probs

def init_detector(device="cuda" if torch.cuda.is_available() else "cpu"):
    detector = RetinaFaceDetector(
        confidence_threshold=0.5,
        nms_iou_threshold=0.3,
        device=device,
        model_type="mobilenet",
        fp16_inference=True,
        clip_boxes=True
    )
    return detector 