import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import GaussianBlur
import math

FEATURE_LAYER_CONFIG = {
    "backbone_early": {   
        "retinaface_mobilenet": ["stage1"],
        "retinaface_resnet": ["layer1"],
        "dsfd": ["conv1_0", "conv1_2", "layer1_5", "layer1_7"],
        "s3fd": ["conv1_1", "conv1_2", "conv2_1", "conv2_2"]  
    },
    "backbone_middle": { 
        "retinaface_mobilenet": ["stage2"],
        "retinaface_resnet": ["layer2"],
        "dsfd": ["layer2_10", "layer2_12", "layer2_14"],
        "s3fd": ["conv3_1", "conv3_2", "conv3_3"]  
    },
    "backbone_late": {   
        "retinaface_mobilenet": ["stage3"],
        "retinaface_resnet": ["layer3", "layer4"],
        "dsfd": ["layer3_17", "layer3_19", "layer3_21", "layer4_24", "layer4_26", "layer4_28"],
        "s3fd": ["conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]  
    },
    "detection_specific": {  
        "retinaface_mobilenet": ["ssh1", "ssh2", "ssh3", "fpn_neck.p6", "fpn_neck.p7"],
        "retinaface_resnet": ["ssh1", "ssh2", "ssh3", "fpn_neck.p6", "fpn_neck.p7"],
        "dsfd": ["extras1", "extras2", "extras3", "loc1", "conf1", "fem1", "fem2", "fem3"],
        "s3fd": ["conv6", "conv7", "conv_final"] 
}}


def get_detector_features(img, detector, detector_type="retinaface"):
    
    s3fd_conv3_3_raw_conf = None 
    if detector_type.lower() == "s3fd":
        extracted_s3fd_features_dict, outputs, s3fd_conv3_3_raw_conf = get_s3fd_features(img, detector)
        features = [
            extracted_s3fd_features_dict["backbone_middle"][0], # L2Norm(conv3_3)
            extracted_s3fd_features_dict["backbone_late"][0],    # L2Norm(conv4_3)
            extracted_s3fd_features_dict["detection_specific"][0] # L2Norm(conv5_3)
        ]
    elif detector_type.lower() == "retinaface":
        features, outputs = extract_retinaface_features_direct(img, detector) 
    elif detector_type.lower() == "dsfd":
        features, outputs = extract_dsfd_features_direct(img, detector) 
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    
    if len(features) != 3:
        if len(features) > 3: features = features[:3]
        while len(features) < 3:
            if features: features.append(torch.zeros_like(features[0]))
            else: features.append(torch.zeros(img.size(0),1,1,1, device=img.device).requires_grad_(True))
    
    return features, outputs, s3fd_conv3_3_raw_conf
        

def normalize_feature_dimensions(features_list, target_size=(64, 64)):
    
    normalized_features = []
    for features in features_list:
        if not isinstance(features, list):
            features = [features]
        
        for feat in features:
            if feat is None:
                continue
                
            target_channels = 128  
            if feat.size(1) != target_channels:
                channel_adaptor = torch.nn.Conv2d(
                    feat.size(1), target_channels, kernel_size=1, 
                    bias=False
                ).to(feat.device)
                
                torch.nn.init.kaiming_uniform_(channel_adaptor.weight, a=0)
                feat = channel_adaptor(feat)
            
            normalized_feat = F.interpolate(
                feat, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            normalized_features.append(normalized_feat)
    
    return normalized_features

def fuse_features_by_type(features_dict, fusion_type="attention", target_size=(64, 64)):
    
    collected_features = {
        "vgg_conv3": [],   
        "vgg_conv4": [],    
        "vgg_conv5": []  
    }
    outputs_features = []
    for detector_name, (detector_features, outputs) in features_dict.items():
        if len(detector_features) == 3:
            collected_features["vgg_conv3"].append(detector_features[0])  
            collected_features["vgg_conv4"].append(detector_features[1])  
            collected_features["vgg_conv5"].append(detector_features[2])  
        elif len(detector_features) == 3 and 'retinaface' in detector_name.lower():
            collected_features["vgg_conv3"].append(detector_features[0])  
            collected_features["vgg_conv4"].append(detector_features[1])  
            collected_features["vgg_conv5"].append(detector_features[2])  
        else:
            available = len(detector_features)
            if available >= 1:
                collected_features["vgg_conv3"].append(detector_features[0])
            if available >= 2:
                collected_features["vgg_conv4"].append(detector_features[1])
            if available >= 3:
                collected_features["vgg_conv5"].append(detector_features[2])
        
        if outputs:
            if isinstance(outputs, (list, tuple)):
                outputs_features.extend(outputs)
            else:
                outputs_features.append(outputs)
    
    fused_features = {}
    for feature_type, features in collected_features.items():
        if not features:
            fused_features[feature_type] = None
            continue
        
        normalized_features = normalize_feature_dimensions(features, target_size)
        
        if fusion_type == "concat" and normalized_features:
            fused_features[feature_type] = torch.cat(normalized_features, dim=1)
            
        elif fusion_type == "attention" and normalized_features:
            attention_weights = []
            for feat in normalized_features:
                channel_att = torch.sigmoid(F.adaptive_avg_pool2d(feat, 1))
                spatial_att = torch.sigmoid(feat.mean(dim=1, keepdim=True))
                combined_att = channel_att * spatial_att
                attention_weights.append(combined_att)
            
            sum_weights = sum(attention_weights)
            normalized_weights = [w / (sum_weights + 1e-8) for w in attention_weights]
            
            weighted_features = [feat * weight for feat, weight in zip(normalized_features, normalized_weights)]
            fused_features[feature_type] = torch.cat(weighted_features, dim=1)
            
        elif fusion_type == "weighted" and normalized_features:
            detector_weights = {"dsfd": 1.2, "s3fd": 1.0, "retinaface": 1.1 }
            
            weights = [1.0 / len(normalized_features)] * len(normalized_features)
            weighted_features = [feat * w for feat, w in zip(normalized_features, weights)]
            fused_features[feature_type] = sum(weighted_features)
    
    return fused_features, outputs_features

def fuse_features(features_dict, fusion_type="attention", target_size=(64, 64)):
    fused_features_dict, outputs_features = fuse_features_by_type(features_dict, fusion_type, target_size)
    all_features = []
    for feature_type in ["vgg_conv3", "vgg_conv4", "vgg_conv5"]:
        if feature_type in fused_features_dict and fused_features_dict[feature_type] is not None:
            all_features.append(fused_features_dict[feature_type])
        else:
            mock_feature = torch.ones(
                1, 128, target_size[0], target_size[1], 
                device=next(iter(features_dict.values()))[0][0].device
            ).requires_grad_(True)
            all_features.append(mock_feature)
    
    return all_features, outputs_features

def compute_multiscale_adversarial_loss(
    watermarked_features,
    clean_features,
    watermarked_outputs,
    clean_outputs,
    s3fd_conv3_3_raw_conf_wm=None
    ):
    device = watermarked_features[0].device if watermarked_features and watermarked_features[0] is not None else (s3fd_conv3_3_raw_conf_wm.device if s3fd_conv3_3_raw_conf_wm is not None else torch.device('cpu'))
    total_feature_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_output_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    has_valid_boxes = False
    if watermarked_outputs:
        for out_list_item in watermarked_outputs:
            if isinstance(out_list_item, list): 
                for out_dict in out_list_item:
                    if isinstance(out_dict, dict) and 'boxes' in out_dict and isinstance(out_dict['boxes'], torch.Tensor):
                        if out_dict['boxes'].size(0) > 0:
                            has_valid_boxes = True
                            break
            elif isinstance(out_list_item, dict): 
                 if 'boxes' in out_list_item and isinstance(out_list_item['boxes'], torch.Tensor):
                    if out_list_item['boxes'].size(0) > 0:
                        has_valid_boxes = True
            if has_valid_boxes:
                    break
    
    feature_weights = [5.0, 3.0, 4.0]
    if watermarked_features and clean_features and len(watermarked_features) == len(clean_features):
        for i, (wm_feat, clean_feat) in enumerate(zip(watermarked_features, clean_features)):
            if wm_feat is None or clean_feat is None: continue
            if not wm_feat.requires_grad: wm_feat = wm_feat.clone().requires_grad_(True)
        
        noise_scale = 0.2 if has_valid_boxes else 0.5
        random_noise = torch.randn_like(wm_feat) * noise_scale
        target = -clean_feat + random_noise
        
        importance = torch.abs(clean_feat).mean(dim=(1,2,3), keepdim=True) 
        importance_mask = (torch.abs(clean_feat) > importance).float() 
        
        region_weight_factor = 2.0 if has_valid_boxes else 4.0

        target_for_loss = target.clone()
        target_for_loss[importance_mask == 1] *= region_weight_factor


        current_feat_loss_weight = feature_weights[i] if i < len(feature_weights) else feature_weights[-1]
        feat_loss = F.mse_loss(wm_feat, target_for_loss) * current_feat_loss_weight
        total_feature_loss = total_feature_loss + feat_loss
        
        if not has_valid_boxes:
            suppress_loss = torch.mean(torch.abs(wm_feat)) * 2.0
            total_feature_loss = total_feature_loss + suppress_loss
    
    base_loss_value = -0.1 if has_valid_boxes else -1.0 
    current_total_output_loss_from_final_outputs = torch.tensor(base_loss_value, device=device) 
    
    outputs_to_process = []
    if watermarked_outputs:
        if isinstance(watermarked_outputs[0], list):
                outputs_to_process.extend(det_batch_out)
        elif isinstance(watermarked_outputs[0], dict):
            outputs_to_process = watermarked_outputs

    for wm_out_item in outputs_to_process: 
        if wm_out_item is None or not isinstance(wm_out_item, dict):
                continue
                
        if 'scores' in wm_out_item and isinstance(wm_out_item['scores'], torch.Tensor) and wm_out_item['scores'].numel() > 0:
            scores_tensor = wm_out_item['scores']
            if not scores_tensor.requires_grad: scores_tensor = scores_tensor.detach().requires_grad_(True)
            zero_target_scores = torch.zeros_like(scores_tensor)
            score_loss = F.mse_loss(scores_tensor, zero_target_scores) * 12.0
            current_total_output_loss_from_final_outputs = current_total_output_loss_from_final_outputs + score_loss
        
        if 'boxes' in wm_out_item and isinstance(wm_out_item['boxes'], torch.Tensor) and wm_out_item['boxes'].size(0) > 0:
            boxes_tensor = wm_out_item['boxes']
            if not boxes_tensor.requires_grad: boxes_tensor = boxes_tensor.detach().requires_grad_(True)
            center_x = (boxes_tensor[:, 0] + boxes_tensor[:, 2]) / 2
            center_y = (boxes_tensor[:, 1] + boxes_tensor[:, 3]) / 2
            min_size_box = 0.001
            target_boxes_min = torch.stack([
                center_x - min_size_box, center_y - min_size_box,
                center_x + min_size_box, center_y + min_size_box
            ], dim=1)
            box_loss = F.l1_loss(boxes_tensor, target_boxes_min) * 12.0
            current_total_output_loss_from_final_outputs = current_total_output_loss_from_final_outputs + box_loss
            
    total_output_loss = total_output_loss + current_total_output_loss_from_final_outputs

    if s3fd_conv3_3_raw_conf_wm is not None:
        if not s3fd_conv3_3_raw_conf_wm.requires_grad:
            s3fd_conv3_3_raw_conf_wm = s3fd_conv3_3_raw_conf_wm.clone().requires_grad_(True)

        face_logits_c3 = s3fd_conv3_3_raw_conf_wm[:, 3, :, :]
        bg_logits_c3_all = s3fd_conv3_3_raw_conf_wm[:, 0:3, :, :]
        max_bg_logits_c3, _ = torch.max(bg_logits_c3_all, dim=1)

        s3fd_conv3_3_objective = torch.mean(max_bg_logits_c3 - face_logits_c3)
        
        s3fd_conv3_3_specific_loss_weight = 20.0
        
        total_output_loss = total_output_loss + (s3fd_conv3_3_specific_loss_weight * s3fd_conv3_3_objective)
    if total_feature_loss.dim() > 0 : total_feature_loss = total_feature_loss.mean()
    if total_output_loss.dim() > 0 : total_output_loss = total_output_loss.mean()

    return total_feature_loss, total_output_loss

def optimize_watermark_with_multi_detector_features(
    watermark, 
    clean_images, 
    detectors_dict,
    device,
    pgd_steps=100, 
    pgd_eps=0.5, 
    pgd_lr=0.01, 
    pgd_momentum=0.9,
    fusion_type="attention",
    feature_size=(64, 64)):

    print(f"Optimization attack based on multi-detector characteristics (maximum number of steps ={pgd_steps}, disturbance range ={pgd_eps})")
    original_watermark = watermark.clone().detach()
    batch_size = clean_images.size(0)
    
    if original_watermark.size(0) != batch_size:
        if original_watermark.size(0) == 1:
            original_watermark = original_watermark.expand(batch_size, -1, -1, -1)
        else:
            original_watermark = original_watermark[:batch_size]
    
    if original_watermark.size(2) != clean_images.size(2) or original_watermark.size(3) != clean_images.size(3):
        original_watermark = F.interpolate(
            original_watermark,
            size=(clean_images.size(2), clean_images.size(3)),
            mode='bilinear',
            align_corners=False
        )
    
    threshold = 0.040  
    watermark_magnitude = torch.abs(original_watermark)
    basic_mask = (watermark_magnitude < threshold).float()
    decay_factor = torch.exp(-5.0 * watermark_magnitude)  
    poison_mask = decay_factor * basic_mask
    delta = torch.randn_like(original_watermark) * 0.001 * poison_mask
    delta.requires_grad = True

    clean_features_dict = {}
    with torch.no_grad():
        for detector_name, (detector, detector_type) in detectors_dict.items():
            features, outputs, _ = get_detector_features(clean_images, detector, detector_type)
            clean_features_dict[detector_name] = (features, outputs)
        
        clean_fused_features, clean_out_features = fuse_features(
            clean_features_dict, 
            fusion_type=fusion_type, 
            target_size=feature_size
        )
    
    optimizer = torch.optim.Adam([delta], lr=pgd_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pgd_steps)
    
    best_delta = None
    best_loss = float('inf')  
    for step in range(pgd_steps):
        optimizer.zero_grad()
        
        current_watermark = original_watermark + delta
        watermarked_images = torch.clamp(clean_images + current_watermark, -1, 1)
        
        watermarked_features_dict = {}
        s3fd_conv3_3_raw_conf_for_loss = None

        for detector_name, (detector, detector_type) in detectors_dict.items():
            if hasattr(detector, 'eval'):
                detector.eval()
            features, outputs, s3fd_specific_raw_conf = get_detector_features(watermarked_images, detector, detector_type)
            watermarked_features_dict[detector_name] = (features, outputs)
            if detector_type.lower() == "s3fd" and s3fd_specific_raw_conf is not None:
                s3fd_conv3_3_raw_conf_for_loss = s3fd_specific_raw_conf
        watermarked_fused_features, watermarked_out_features = fuse_features(
            watermarked_features_dict, 
            fusion_type=fusion_type, 
            target_size=feature_size
        )
        feature_loss = torch.tensor(0.0, device=device, requires_grad=True)
        output_loss = torch.tensor(0.0, device=device, requires_grad=True)
        feature_loss, output_loss = compute_multiscale_adversarial_loss(
            watermarked_fused_features,
            clean_fused_features,
            watermarked_out_features,
            clean_out_features,
            s3fd_conv3_3_raw_conf_wm=s3fd_conv3_3_raw_conf_for_loss 
        )
        
        if not isinstance(feature_loss, torch.Tensor) or not feature_loss.requires_grad:
            feature_loss = delta.sum() * 0.0 - 0.1
        
        if not isinstance(output_loss, torch.Tensor) or not output_loss.requires_grad:
            output_loss = delta.sum() * 0.0 - 0.01
        
        feature_loss.backward(retain_graph=True)
        output_loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            delta.grad = torch.randn_like(delta) * 0.01
        
        torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            delta.data = delta.data * poison_mask
            delta.data = torch.clamp(delta.data, -pgd_eps, pgd_eps)
        
        current_feature_loss = feature_loss.item() if isinstance(feature_loss, torch.Tensor) else float('inf')
        current_output_loss = output_loss.item() if isinstance(output_loss, torch.Tensor) else float('inf')
        
        if current_output_loss < best_loss:
            best_loss = current_output_loss
            best_delta = delta.clone().detach()
        if "s3fd" in detector_name.lower(): 
            allowed_pixels_ratio = poison_mask.sum() / poison_mask.numel()
    
    if best_delta is not None:
        final_watermark = original_watermark + best_delta
    else:
        final_watermark = original_watermark + delta.detach()
    
    final_watermark = torch.clamp(final_watermark, -1, 1)
    
    return final_watermark, float(best_loss)


def extract_retinaface_features_direct(img, detector):
    
    if not isinstance(img, torch.Tensor):
        raise ValueError("torch.Tensor")
    
    if img.dim() == 3:
        img = img.unsqueeze(0)
        
    device = img.device
    
    with torch.no_grad():
        if img.min() < 0:
            img_input = (img + 1) * 127.5
        else:
            img_input = img * 255.0
            
        img_input = img_input[:, [2, 1, 0], :, :]
        mean = torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)
        img_input = img_input - mean

    if hasattr(detector, 'net'):
        net = detector.net
    elif hasattr(detector, 'module'): # DataParallel
        net = detector.module
    else:
        net = detector
    
    features = []
    
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output)

    hooks = []
    try:
        target_layers = ['ssh1', 'ssh2', 'ssh3']
        valid_layers = []
        
        for name in target_layers:
            if hasattr(net, name):
                layer = getattr(net, name)
                hooks.append(layer.register_forward_hook(hook_feature))
                valid_layers.append(name)
            elif hasattr(net, 'body') and hasattr(net.body, name):
                layer = getattr(net.body, name)
                hooks.append(layer.register_forward_hook(hook_feature))
                valid_layers.append(name)

        outputs_tuple = net(img_input)
        
        features = feature_blobs
        
        if len(features) < 3:
            feature_blobs.clear()
            out = net.body(img_input)
            fpn_features = net.fpn(out)
            features = list(fpn_features.values()) if isinstance(fpn_features, dict) else list(fpn_features)
            
            if hasattr(net, 'ssh1'): features[0] = net.ssh1(features[0])
            if hasattr(net, 'ssh2') and len(features)>1: features[1] = net.ssh2(features[1])
            if hasattr(net, 'ssh3') and len(features)>2: features[2] = net.ssh3(features[2])

    except Exception as e:
        print(f"Exception in RetinaFace feature extraction: {e}")
    finally:
        for h in hooks:
            h.remove()

    while len(features) < 3:

        scale = 2**(3 + len(features)) # 8, 16, 32
        features.append(torch.zeros(img.size(0), 64, img.size(2)//scale, img.size(3)//scale, 
                                    device=device, requires_grad=True))
    
    features = features[:3]
    formatted_outputs = []
    
    if isinstance(outputs_tuple, tuple) and len(outputs_tuple) >= 2:
        loc, conf = outputs_tuple[0], outputs_tuple[1]
        batch_size = conf.size(0)
        for i in range(batch_size):
            scores = conf[i, :, 1]
            boxes_pred = loc[i]
            
            output_dict = {
                'scores': scores,
                'boxes': boxes_pred 
            }
            formatted_outputs.append(output_dict)
    else:
        formatted_outputs = None

    return features, formatted_outputs

def get_s3fd_features(img, detector):
    
    device = img.device
    # Image preprocessing (as before)
    img_min, img_max = img.min().item(), img.max().item()
    preprocessed_img = img
    if img_min >= -1 and img_min < 0 and img_max <= 1:
        preprocessed_img = (img + 1) * 127.5
        mean = torch.tensor([104., 117., 123.]).view(1, 3, 1, 1).to(device)
        preprocessed_img = preprocessed_img[:, [2, 1, 0], :, :] - mean
    elif img_min >= 0 and img_max <= 1:
        preprocessed_img = img * 255.0
        mean = torch.tensor([104., 117., 123.]).view(1, 3, 1, 1).to(device)
        preprocessed_img = preprocessed_img[:, [2, 1, 0], :, :] - mean
    
    original_conf_thresh = None
    if hasattr(detector, 'conf_threshold'):
        original_conf_thresh = detector.conf_threshold
    import face_detection.S3FD.S3FDDetector as sfd_detector_module 
    selected_main_features, final_detection_outputs, conv3_3_raw_conf = sfd_detector_module.get_features(detector, preprocessed_img)
    if hasattr(detector, 'conf_threshold') and original_conf_thresh is not None:
        pass 

    if not selected_main_features or len(selected_main_features) < 3 : # Basic check
        dummy_device = img.device
        sf0 = torch.zeros(img.size(0), 256, img.size(2)//4, img.size(3)//4, device=dummy_device).requires_grad_(True)
        sf1 = torch.zeros(img.size(0), 512, img.size(2)//8, img.size(3)//8, device=dummy_device).requires_grad_(True)
        sf2 = torch.zeros(img.size(0), 512, img.size(2)//16, img.size(3)//16, device=dummy_device).requires_grad_(True)
        selected_main_features = [sf0,sf1,sf2]
        if conv3_3_raw_conf is None:
                conv3_3_raw_conf = torch.zeros(img.size(0), 4, img.size(2)//4, img.size(3)//4, device=dummy_device).requires_grad_(True)
    
    extracted_features_dict = {
        "backbone_early": [], 
        "backbone_middle": [selected_main_features[0]], 
        "backbone_late": [selected_main_features[1]],   
        "detection_specific": [selected_main_features[2]] 
    }
    return extracted_features_dict, final_detection_outputs, conv3_3_raw_conf


def extract_dsfd_features_direct(img, detector):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    
    device = img.device
    backup_features = [
        torch.ones(img.size(0), the_size[0], img.size(2)//the_size[1], img.size(3)//the_size[1], device=device).requires_grad_(True)
        for the_size in [(256, 4), (512, 8), (512, 16)]]
    
    if hasattr(detector, 'model'):
        model = detector.model
    else:
        model = detector
    
    training_mode = model.training
    model.train()
    img_min, img_max = img.min().item(), img.max().item()
    preprocessed_img = img.clone()
    if img_min >= -1 and img_min < 0 and img_max <= 1:
        preprocessed_img = (img + 1) * 127.5
        mean = torch.tensor([104., 117., 123.]).view(1, 3, 1, 1).to(device)
        preprocessed_img = preprocessed_img[:, [2, 1, 0], :, :] - mean
    elif img_min >= 0 and img_max <= 1:
        preprocessed_img = img * 255.0
        mean = torch.tensor([104., 117., 123.]).view(1, 3, 1, 1).to(device)
        preprocessed_img = preprocessed_img[:, [2, 1, 0], :, :] - mean
    
    with torch.enable_grad():  
        if hasattr(model, 'forward') and hasattr(model.forward, '__code__') and 'get_features' in model.forward.__code__.co_varnames:
            features = model(preprocessed_img, get_features=True)
        
        for i in range(len(features)):
            if not features[i].requires_grad:
                features[i] = features[i].clone().requires_grad_(True)
    
    if len(features) > 3:
        features = features[:3]
        
    while len(features) < 3:
        backup_idx = len(features)
        features.append(backup_features[backup_idx])
    
    if not training_mode:
        model.eval()
        
    with torch.no_grad():
        outputs = model(preprocessed_img)
        
    if training_mode:
        model.train()
    
    return features, outputs
