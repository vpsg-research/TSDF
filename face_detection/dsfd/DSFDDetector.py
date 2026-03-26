import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import sys
import cv2
import numpy as np
from itertools import product as product
from collections import OrderedDict

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class FEM(nn.Module):
    def __init__(self, in_planes):
        super(FEM, self).__init__()
        inter_planes = in_planes // 3
        inter_planes1 = in_planes - 2 * inter_planes
        self.branch1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes1, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3, stride=1, padding=3, dilation=3)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = F.relu(out, inplace=True)
        return out

class DSFD(nn.Module):
    def __init__(self, phase='test', num_classes=2):
        super(DSFD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        
        self.variance = [0.1, 0.2]
        self.conf_threshold = 0.5
        self.nms_threshold = 0.3
        self.top_k = 750
        self.nms_top_k = 5000

        # --- VGG Backbone ---
        self.vgg = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(256, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3, 1, padding=3, dilation=3), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, 1), nn.ReLU(inplace=True),
        ])

        self.L2Normof1 = L2Norm(256, 10)
        self.L2Normof2 = L2Norm(512, 8)
        self.L2Normof3 = L2Norm(512, 5)

        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, padding=1),
        ])

        # --- FPN Layers ---
        self.fpn_topdown = nn.ModuleList([
            nn.Conv2d(256, 256, 1, 1, padding=0),
            nn.Conv2d(256, 512, 1, 1, padding=0),
            nn.Conv2d(512, 1024, 1, 1, padding=0),
            nn.Conv2d(1024, 512, 1, 1, padding=0),
            nn.Conv2d(512, 512, 1, 1, padding=0),
            nn.Conv2d(512, 256, 1, 1, padding=0),
        ])

        self.fpn_latlayer = nn.ModuleList([
            nn.Conv2d(512, 512, 1, 1, padding=0),
            nn.Conv2d(1024, 1024, 1, 1, padding=0),
            nn.Conv2d(512, 512, 1, 1, padding=0),
            nn.Conv2d(512, 512, 1, 1, padding=0),
            nn.Conv2d(256, 256, 1, 1, padding=0),
        ])

        self.fpn_fem = nn.ModuleList([
            FEM(256), FEM(512), FEM(512),
            FEM(1024), FEM(512), FEM(256),
        ])

        self.L2Normef1 = L2Norm(256, 10)
        self.L2Normef2 = L2Norm(512, 8)
        self.L2Normef3 = L2Norm(512, 5)
        
        # --- Head Layers ---
        self.loc_pal2 = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(1024, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(256, 4, 3, 1, padding=1),
        ])

        self.conf_pal2 = nn.ModuleList([
            nn.Conv2d(256, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(1024, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(256, 2, 3, 1, padding=1),
        ])

        # PAL1 for compatibility
        self.loc_pal1 = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1), nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1), nn.Conv2d(1024, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1), nn.Conv2d(256, 4, 3, 1, padding=1),
        ])
        self.conf_pal1 = nn.ModuleList([
            nn.Conv2d(256, 2, 3, 1, padding=1), nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1), nn.Conv2d(1024, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1), nn.Conv2d(256, 2, 3, 1, padding=1),
        ])

        self.softmax = nn.Softmax(dim=-1)

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) * y

    def load_weights(self, base_file):

        if not os.path.exists(base_file):
            print(f"The weight file does not exist: {base_file}")
            return False
        
        print(f"Loading weight: {base_file}")
        checkpoint = torch.load(base_file, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'weight' in checkpoint: weights = checkpoint['weight']
            elif 'model' in checkpoint: weights = checkpoint['model']
            elif 'state_dict' in checkpoint: weights = checkpoint['state_dict']
            else: weights = checkpoint
        else:
            weights = checkpoint
        self.load_state_dict(weights, strict=False)
        return True

    def forward(self, x, get_features=False):

        size = x.size()[2:]
        pal1_sources = list()
        
        # --- VGG Forward ---
        for k in range(16): x = self.vgg[k](x)
        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)

        for k in range(16, 23): x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)

        for k in range(23, 30): x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)

        for k in range(30, len(self.vgg)): x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)

        # --- Extras Forward ---
        for k in range(2): x = F.relu(self.extras[k](x), inplace=True)
        of5 = x
        pal1_sources.append(of5)

        for k in range(2, 4): x = F.relu(self.extras[k](x), inplace=True)
        of6 = x
        pal1_sources.append(of6)

        # --- FPN Forward ---
        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)
        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(self._upsample_prod(x, self.fpn_latlayer[0](of5)), inplace=True)
        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        convfc7_2 = F.relu(self._upsample_prod(x, self.fpn_latlayer[1](of4)), inplace=True)
        x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
        conv5 = F.relu(self._upsample_prod(x, self.fpn_latlayer[2](of3)), inplace=True)
        x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
        conv4 = F.relu(self._upsample_prod(x, self.fpn_latlayer[3](of2)), inplace=True)
        x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
        conv3 = F.relu(self._upsample_prod(x, self.fpn_latlayer[4](of1)), inplace=True)

        # --- FEM Forward ---
        ef1 = self.L2Normef1(self.fpn_fem[0](conv3))
        ef2 = self.L2Normef2(self.fpn_fem[1](conv4))
        ef3 = self.L2Normef3(self.fpn_fem[2](conv5))
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)

        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)

        if get_features:
            return pal1_sources

        loc_pal2 = list()
        conf_pal2 = list()
        
        for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc_pal2)):
            feat = []
            feat += [loc_pal2[i].size(1), loc_pal2[i].size(2)]
            features_maps += [feat]

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_pal2], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf_pal2], 1)
        
        batch_size = loc.size(0)
        loc = loc.view(batch_size, -1, 4)
        conf = conf.view(batch_size, -1, self.num_classes)
        conf_scores = self.softmax(conf)

        with torch.no_grad():
            self.priorbox = PriorBox(size, features_maps, pal=2)
            priors = self.priorbox.forward().to(loc.device)

        output = []
        for i in range(batch_size):
            boxes = decode(loc[i], priors, self.variance)
            scores = conf_scores[i][:, 1].clone()

            mask = scores > self.conf_threshold
            boxes_filtered = boxes[mask]
            scores_filtered = scores[mask]
            
            if scores_filtered.size(0) > 0:
                ids, count = nms(boxes_filtered, scores_filtered, self.nms_threshold, self.nms_top_k)
                count = count if count < self.top_k else self.top_k
                
                output.append({
                    'boxes': boxes_filtered[ids[:count]],
                    'scores': scores_filtered[ids[:count]],
                    'labels': torch.ones(count, dtype=torch.long, device=loc.device)
                })
            else:
                 output.append({
                    'boxes': torch.zeros((0, 4), device=loc.device),
                    'scores': torch.zeros(0, device=loc.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=loc.device)
                })
        return output

class PriorBox(object):
    def __init__(self, input_size, feature_maps, variance=[0.1, 0.2],
                 steps=[4, 8, 16, 32, 64, 128], clip=False, pal=2):
        super(PriorBox, self).__init__()
        self.imh = input_size[0]
        self.imw = input_size[1]
        self.feature_maps = feature_maps
        self.variance = variance
        if pal == 1:
            self.min_sizes = [8, 16, 32, 64, 128, 256]
        elif pal == 2:
            self.min_sizes = [16, 32, 64, 128, 256, 512]
        self.steps = steps
        self.clip = clip

    def forward(self):
        mean = []
        for k, fmap in enumerate(self.feature_maps):
            feath = fmap[0]
            featw = fmap[1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh

                mean += [cx, cy, s_kw, s_kh]

        output = torch.FloatTensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]
        
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union
        
        idx = idx[IoU.le(overlap)]
    return keep, count

def preprocess_image(image, input_size=None):
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = np.array(image)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    img = img.astype('float32')
    img -= np.array([104., 117., 123.])
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    return img

def init_dsfd_detector(device='cuda', weights_path=None, conf_threshold=None, nms_threshold=None):
    
    net = DSFD(phase='test')
    if conf_threshold is not None: net.conf_threshold = conf_threshold
    if nms_threshold is not None: net.nms_threshold = nms_threshold
        
    if weights_path is None:
        weights_path = '/home/zhr-23/code/TSDF_code/face_detection/dsfd/weights/dsfd_vgg_0.880.pth'
        
    if weights_path is not None:
            if os.path.exists(weights_path):
                net.load_weights(weights_path)
            else:
                print(f"Warning: No valid weight file was found: {weights_path}")
        
    net.eval()
    net = net.to(device)
    return net

def get_features(detector, img):

    is_train = detector.training
    detector.train() 
    device = next(detector.parameters()).device
    img = img.to(device)
    
    all_source_features = detector(img, get_features=True)
    if len(all_source_features) >= 3:
        selected_features = all_source_features[0:3]
    else:
        selected_features = all_source_features
        
    conv3_3_l2normed_feat = all_source_features[0]
    conv3_3_raw_conf = detector.conf_pal1[0](conv3_3_l2normed_feat)
    
    if conv3_3_l2normed_feat.requires_grad and not conv3_3_raw_conf.requires_grad:
        conv3_3_raw_conf = conv3_3_raw_conf.clone().requires_grad_(True)
        
    if not is_train: detector.eval()
    with torch.no_grad() if not is_train else torch.enable_grad():
        final_outputs = detector(img, get_features=False)
        
    if is_train: detector.train()
    
    new_selected_features = []
    for sf in selected_features:
        if img.requires_grad and not sf.requires_grad:
            new_selected_features.append(sf.clone().requires_grad_(True))
        else:
            new_selected_features.append(sf)
    selected_features = new_selected_features
    
    return selected_features, final_outputs, conv3_3_raw_conf


def detect_faces(detector, img, conf_threshold=None, nms_threshold=None):

    original_conf = detector.conf_threshold
    original_nms = detector.nms_threshold
    
    if conf_threshold is not None: detector.conf_threshold = conf_threshold
    if nms_threshold is not None: detector.nms_threshold = nms_threshold
    
    detector.eval()
    
    if not isinstance(img, torch.Tensor):
        x = preprocess_image(img)
    else:
        x = img
        
    device = next(detector.parameters()).device
    x = x.to(device)
    
    with torch.no_grad():
        output = detector(x)
        
    detector.conf_threshold = original_conf
    detector.nms_threshold = original_nms
    
    return output[0]
