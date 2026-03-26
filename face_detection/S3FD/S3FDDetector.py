import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from itertools import product

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x
        return out

class S3FD(nn.Module):
    def __init__(self, phase='test', num_classes=2):
        super(S3FD, self).__init__()
        
        self.phase = phase
        self.num_classes = num_classes
        
        self.variance = [0.1, 0.2]
        self.conf_threshold = 0.5
        self.nms_threshold = 0.3 
        self.top_k = 750
        
        self.vgg = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding=1),        # conv1_1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),       # conv1_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                    # pool1
            
            nn.Conv2d(64, 128, 3, padding=1),      # conv2_1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),     # conv2_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                    # pool2
            
            nn.Conv2d(128, 256, 3, padding=1),     # conv3_1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),     # conv3_2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),     # conv3_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),    # pool3
            
            nn.Conv2d(256, 512, 3, padding=1),     # conv4_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),     # conv4_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),     # conv4_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                    # pool4
            
            nn.Conv2d(512, 512, 3, padding=1),     # conv5_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),     # conv5_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),     # conv5_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                    # pool5
            
            nn.Conv2d(512, 1024, 3, padding=6, dilation=6),  # conv6
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1),              # conv7
            nn.ReLU(inplace=True),
        ])
        
        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1),                      # conv8_1
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # conv8_2
            nn.Conv2d(512, 128, 1),                       # conv9_1
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # conv9_2
        ])
        
        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)
        
        self.loc = nn.ModuleList([
            nn.Conv2d(256, 4, kernel_size=3, padding=1),  # conv3_3
            nn.Conv2d(512, 4, kernel_size=3, padding=1),  # conv4_3
            nn.Conv2d(512, 4, kernel_size=3, padding=1),  # conv5_3
            nn.Conv2d(1024, 4, kernel_size=3, padding=1), # conv_fc7
            nn.Conv2d(512, 4, kernel_size=3, padding=1),  # conv8_2
            nn.Conv2d(256, 4, kernel_size=3, padding=1),  # conv9_2
        ])
        
        self.conf = nn.ModuleList([
            nn.Conv2d(256, 4, kernel_size=3, padding=1),      # conv3_3
            nn.Conv2d(512, 2, kernel_size=3, padding=1),      # conv4_3
            nn.Conv2d(512, 2, kernel_size=3, padding=1),      # conv5_3
            nn.Conv2d(1024, 2, kernel_size=3, padding=1),     # conv_fc
            nn.Conv2d(512, 2, kernel_size=3, padding=1),      # conv8_2
            nn.Conv2d(256, 2, kernel_size=3, padding=1)       # conv9_2
        ])
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, get_features=False):

        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()
        for k in range(16):
            x = self.vgg[k](x)
        
        s = self.L2Norm3_3(x) 
        sources.append(s)
        for k in range(16, 23):
            x = self.vgg[k](x)
            
        s = self.L2Norm4_3(x)
        sources.append(s)
        
        for k in range(23, 30):
            x = self.vgg[k](x)
            
        s = self.L2Norm5_3(x)
        sources.append(s)
        
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        
        x = F.relu(self.extras[0](x))
        x = F.relu(self.extras[1](x))
        sources.append(x)
        
        x = F.relu(self.extras[2](x))
        x = F.relu(self.extras[3](x))
        sources.append(x)
        
        if get_features:
            return sources
            
        features_maps = []
        for i in range(len(sources)):
            features_maps.append(sources[i].size()[2:])
            
        self.priorbox = PriorBox(size, features_maps)
        priors = self.priorbox.forward().to(x.device)
        
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])
        
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)
        
        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        
        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        batch_size = x.size(0)
        loc = loc.view(batch_size, -1, 4)
        conf = conf.view(batch_size, -1, 2)
        
        conf_scores = self.softmax(conf)
        
        output = []
        for i in range(batch_size):
            boxes = decode(loc[i], priors, self.variance)
            scores = conf_scores[i][:, 1].clone()
            
            mask = scores > self.conf_threshold
            boxes_filtered = boxes[mask]
            scores_filtered = scores[mask]
            
            if scores_filtered.size(0) > 0:
                ids, count = nms(boxes_filtered, scores_filtered, self.nms_threshold, self.top_k)
                if count > 0:
                    output.append({
                        'boxes': boxes_filtered[ids[:count]],
                        'scores': scores_filtered[ids[:count]],
                        'labels': torch.ones(count, dtype=torch.long, device=x.device)
                    })
                else:
                    output.append({
                        'boxes': torch.zeros((0, 4), device=x.device),
                        'scores': torch.zeros(0, device=x.device),
                        'labels': torch.zeros(0, dtype=torch.long, device=x.device)
                    })
            else:
                output.append({
                    'boxes': torch.zeros((0, 4), device=x.device),
                    'scores': torch.zeros(0, device=x.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=x.device)
                })
        
        total_boxes = sum(out['boxes'].size(0) for out in output)
        if total_boxes == 0:
            print("Warning: No face was detected!")
        else:
            avg_boxes = total_boxes / batch_size
                
        return output
        
    def load_weights(self, base_file):
        if not os.path.exists(base_file):
            print(f"Weight file does not exist: {base_file}")
            return False
            
        try:
            print(f"Loading weights: {base_file}")
            checkpoint = torch.load(base_file, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'weight' in checkpoint:
                    weights = checkpoint['weight']
                elif 'model' in checkpoint:
                    weights = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    weights = checkpoint['state_dict']
                else:
                    weights = checkpoint
            else:
                weights = checkpoint
                
            self.load_state_dict(weights, strict=False)
            print("Weight loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading: {e}")
            import traceback
            traceback.print_exc()
            return False

class PriorBox(object):
    def __init__(self, input_size, feature_maps, min_sizes=None, steps=None, clip=False):
        super(PriorBox, self).__init__()
        
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        self.feature_maps = feature_maps
        
        if min_sizes is None:
            self.min_sizes = [16, 32, 64, 128, 256, 512]
        else:
            self.min_sizes = min_sizes
            
        if steps is None:
            self.steps = [4, 8, 16, 32, 64, 128]
        else:
            self.steps = steps
            
        self.clip = clip
        
        if len(self.feature_maps) != len(self.min_sizes) or len(self.feature_maps) != len(self.steps):
            self.min_sizes = self.min_sizes[:len(self.feature_maps)]
            self.steps = self.steps[:len(self.feature_maps)]
        
    def forward(self):
        mean = []
        
        for k, f in enumerate(self.feature_maps):
            if k >= len(self.min_sizes):
                continue
                
            min_size = self.min_sizes[k]
            step = self.steps[k]
            
            if isinstance(f, (list, tuple)):
                h, w = f
            else:
                h, w = f.shape[-2:]
                
            h = int(h)
            w = int(w)
            
            for i, j in product(range(h), range(w)):
                cx = (j + 0.5) * step / self.input_width
                cy = (i + 0.5) * step / self.input_height
                
                s_k = min_size / self.input_width
                mean += [cx, cy, s_k, s_k]
        
        if len(mean) > 0:
            output = torch.tensor(mean).view(-1, 4)
            if self.clip:
                output.clamp_(max=1, min=0)
            return output

def decode(loc, priors, variances=[0.1, 0.2]):
    if priors.device != loc.device:
        priors = priors.to(loc.device)
    
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    
    return boxes

def nms(boxes, scores, overlap=0.5, top_k=200):
    device = boxes.device
    keep = torch.zeros(scores.size(0), dtype=torch.long, device=device)
    
    if boxes.numel() == 0:
        return keep, 0
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    
    count = 0
    while idx.numel() > 0:
        i = idx[0]
        keep[count] = i
        count += 1
        
        if idx.size(0) == 1:
            break
            
        idx = idx[1:]
        
        xx1 = torch.max(x1[i], x1[idx])
        yy1 = torch.max(y1[i], y1[idx])
        xx2 = torch.min(x2[i], x2[idx])
        yy2 = torch.min(y2[i], y2[idx])
        
        w = torch.max(torch.zeros_like(xx2, device=device), xx2 - xx1)
        h = torch.max(torch.zeros_like(yy2, device=device), yy2 - yy1)
        inter = w * h
        
        ovr = inter / (area[i] + area[idx] - inter)
        
        idx = idx[ovr <= overlap]
            
    return keep[:count], count

def preprocess_image(image, input_size=300):
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    height, width, _ = image.shape
    max_im_shrink = np.sqrt(1500 * 1000 / (image.shape[0] * image.shape[1]))
    resized = cv2.resize(image, None, None, fx=max_im_shrink, fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    resized = resized.transpose(2, 0, 1)
    resized = resized.astype('float32')
    resized -= np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
    resized = resized[[2, 1, 0], :, :]
    x = torch.from_numpy(resized).unsqueeze(0)
    
    return x

def init_detector(device='cuda', weights_path=None, conf_threshold=None, nms_threshold=None):
    
    net = S3FD(phase='test')
    if conf_threshold is not None:
        net.conf_threshold = conf_threshold
    if nms_threshold is not None:
        net.nms_threshold = nms_threshold
                
    if weights_path is None:
        weights_path = '/home/zhr-23/code/TSDF_code/face_detection/S3FD/weights/sfd_face.pth'                   
    if weights_path is None or not os.path.exists(weights_path):
        print(f"Warning: No valid weight file was found")
    else:
        net.load_weights(weights_path)
        
    net.eval()
    net = net.to(device)
    
    return net

def get_features(detector, img):

    is_train = detector.training
    detector.train() 
    device = next(detector.parameters()).device # Use device from detector
    img = img.to(device)
    all_source_features = detector(img, get_features=True)

    if len(all_source_features) < 3:
        selected_features = all_source_features[:len(all_source_features)] 
        while len(selected_features) < 3: # Simple padding if too few
                selected_features.append(torch.zeros_like(selected_features[0]) if selected_features else torch.zeros(1,1,1,1, device=device))
    else:
        selected_features = all_source_features[0:3] # L2Norm(conv3_3), L2Norm(conv4_3), L2Norm(conv5_3)

    conv3_3_l2normed_feat = all_source_features[0]
    conv3_3_raw_conf = detector.conf[0](conv3_3_l2normed_feat) # Shape: [B, 4, H, W]

    if conv3_3_l2normed_feat.requires_grad and not conv3_3_raw_conf.requires_grad:
        conv3_3_raw_conf = conv3_3_raw_conf.clone().requires_grad_(True)

    if not is_train:
        detector.eval()
    
    with torch.no_grad() if not is_train else torch.enable_grad(): # Keep grad if was training, else no_grad for final output
        final_outputs = detector(img, get_features=False) # Standard forward pass for detections

    if not is_train:
        detector.train() # Set back to train if we changed it, to not affect outer loops
    else:
        detector.train() # Was already train, ensure it stays train

    new_selected_features = []
    for sf in selected_features:
        if img.requires_grad and not sf.requires_grad:
                new_selected_features.append(sf.clone().requires_grad_(True))
        else:
                new_selected_features.append(sf)
    selected_features = new_selected_features
    
    return selected_features, final_outputs, conv3_3_raw_conf
    

def detect_faces(detector, img, conf_threshold=None, nms_threshold=None):
    
    original_conf_threshold = detector.conf_threshold
    original_nms_threshold = detector.nms_threshold
    
    if conf_threshold is not None:
        detector.conf_threshold = conf_threshold
    if nms_threshold is not None:
        detector.nms_threshold = nms_threshold
    
    detector.eval()
    if not isinstance(img, torch.Tensor):
        x = preprocess_image(img)
    else:        
        x = img
            
    device = next(detector.parameters()).device
    x = x.to(device)
    
    with torch.no_grad():
        output = detector(x)
        
    detector.conf_threshold = original_conf_threshold
    detector.nms_threshold = original_nms_threshold
    
    return output[0] 

    
    