import argparse
import json
import os
from os.path import join
from tqdm import tqdm
import torch
import get_output
from model_data_prepare import prepare
from evaluate_fid import evaluate_multiple_models
import torch.nn.functional as f
import torchvision.models as models
from face_poison_integration import get_detector_features,  optimize_watermark_with_multi_detector_features

def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

# Init the attacker
def init_get_outputs(args_attack):
    get_output_models = get_output.get_all_features(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                                    epsilon=args_attack.attacks.epsilon, args=args_attack.attacks)
    return get_output_models


def calculate_iou(box1, box2):
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou

def test_watermark(watermark,  test_dataloader, detector_models, device, num_test=50, iou_threshold=0.7, mode="default", random_noise_epsilon=0.05):
    """
        Watermark: watermark tensor (used when mode="default ").
        Mode (str): test mode. It can be "default", "no_watermark" or "random_noise".
        Random_noise_epsilon (float): L-infinity amplitude of random noise used when mode="random_noise ".
    """
    n_samples = 0
    thresh = 0.7

    detector_metrics = {}
    for idx, detector in enumerate(detector_models):
        detector_name = f"detector_{idx}" if not hasattr(detector, '__name__') else detector.__name__
        detector_metrics[detector_name] = {
            'precision': 0.0,
            'recall': 0.0,
            'confidence_before': 0.0,
            'confidence_after': 0.0,
            'duq': 0.0,
            'boxes_per_image_before': [],
            'boxes_per_image_after': [],
        }
    
    total_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'confidence_before': 0.0,
        'confidence_after': 0.0,
        'duq': 0.0,
        'boxes_per_image_before': [],
        'boxes_per_image_after': [],
    }
    
    print(f"\nEvaluate the watermark effect (mode: {mode})")
    for idx, (images, att_a, c_org) in enumerate(tqdm(test_dataloader)):
        if num_test is not None and idx * images.size(0) >= num_test:
            break
            
        images = images.to(device)
        current_batch_size = images.size(0)
        if mode == "default":
            if watermark is None:
                raise ValueError("Watermark tensor must be provided for 'default' mode.")
            if watermark.size(0) != current_batch_size:
                current_watermark_to_apply = watermark[0:1].repeat(current_batch_size, 1, 1, 1)
            else:
                current_watermark_to_apply = watermark
            watermarked_images = torch.clamp(images + current_watermark_to_apply, -1, 1)
        elif mode == "no_watermark":
            watermarked_images = images.clone()
        elif mode == "random_noise":
            noise_delta = (torch.rand_like(images, device=images.device) * 2 - 1) * random_noise_epsilon
            watermarked_images = torch.clamp(images + noise_delta, -1, 1)
        else:
            raise ValueError(f"Unknown mode for test_watermark: {mode}")
            
        for img_idx in range(current_batch_size):
            boxes_before = 0
            boxes_after = 0 
            
            for det_idx, detector in enumerate(detector_models):
                detector_name = f"detector_{det_idx}" if not hasattr(detector, '__name__') else detector.__name__
                is_s3fd = hasattr(detector, 'conf_threshold') and not hasattr(detector, 'net')
                with torch.no_grad():
                    if is_s3fd:
                        original_thresh = detector.conf_threshold                        
                        orig_img = images[img_idx:img_idx+1]
                        water_img = watermarked_images[img_idx:img_idx+1]
                        
                        img_min, img_max = orig_img.min().item(), orig_img.max().item()
                        if img_min >= -1 and img_min < 0 and img_max <= 1:
                            orig_processed = (orig_img + 1) * 127.5
                            water_processed = (water_img + 1) * 127.5
                            mean = torch.tensor([104., 117., 123.]).view(1, 3, 1, 1).to(device)
                            orig_processed = orig_processed[:, [2, 1, 0], :, :] - mean
                            water_processed = water_processed[:, [2, 1, 0], :, :] - mean
                        else:
                            orig_processed = orig_img
                            water_processed = water_img
                        
                        original_detection = detector(orig_processed)
                        watermarked_detection = detector(water_processed)

                    else:
                        original_detection = detector(images[img_idx:img_idx+1])
                        watermarked_detection = detector(watermarked_images[img_idx:img_idx+1])
                
                orig_boxes = original_detection[0].get('boxes', torch.empty((0, 4), device=device))
                water_boxes = watermarked_detection[0].get('boxes', torch.empty((0, 4), device=device))
                orig_scores = original_detection[0].get('scores', torch.tensor([], device=device))
                water_scores = watermarked_detection[0].get('scores', torch.tensor([], device=device))
                
                orig_mask = orig_scores > thresh
                water_mask = water_scores > thresh
                orig_boxes = orig_boxes[orig_mask]
                water_boxes = water_boxes[water_mask]
                orig_scores = orig_scores[orig_mask]
                water_scores = water_scores[water_mask]
                
                boxes_before += len(orig_boxes)
                boxes_after += len(water_boxes)
                
                detector_metrics[detector_name]['boxes_per_image_before'].append(len(orig_boxes))
                detector_metrics[detector_name]['boxes_per_image_after'].append(len(water_boxes))
                
                if len(orig_scores) > 0:
                    total_metrics['confidence_before'] += float(orig_scores.mean().cpu())
                    detector_metrics[detector_name]['confidence_before'] += float(orig_scores.mean().cpu())
                if len(water_scores) > 0:
                    total_metrics['confidence_after'] += float(water_scores.mean().cpu())
                    detector_metrics[detector_name]['confidence_after'] += float(water_scores.mean().cpu())
                
                # Calculating accuracy and recall
                true_detections = 0
                matched_water_boxes = set()
                
                for i, orig_box in enumerate(orig_boxes):
                    max_iou = 0
                    best_match = -1
                    
                    for j, water_box in enumerate(water_boxes):
                        if j not in matched_water_boxes:
                            iou = calculate_iou(orig_box.cpu().numpy(), water_box.cpu().numpy())
                            if iou > max_iou and iou >= iou_threshold:
                                max_iou = iou
                                best_match = j
                    
                    if best_match >= 0:
                        true_detections += 1
                        matched_water_boxes.add(best_match)
                
                precision = true_detections / len(water_boxes) if len(water_boxes) > 0 else 1.0
                recall = true_detections / len(orig_boxes) if len(orig_boxes) > 0 else 1.0
                total_metrics['precision'] += precision
                total_metrics['recall'] += recall
                detector_metrics[detector_name]['precision'] += precision
                detector_metrics[detector_name]['recall'] += recall
                
                # Calculating DUQ
                if len(orig_boxes) > 0:
                    duq = (true_detections - (len(water_boxes) - true_detections)) / len(orig_boxes)
                else:
                    duq = 0.0 if len(water_boxes) == 0 else -1.0
                total_metrics['duq'] += duq
                detector_metrics[detector_name]['duq'] += duq
            
            avg_boxes_before = boxes_before / len(detector_models)
            avg_boxes_after = boxes_after / len(detector_models)
            
            total_metrics['boxes_per_image_before'].append(avg_boxes_before)
            total_metrics['boxes_per_image_after'].append(avg_boxes_after)
        
        n_samples += current_batch_size
    
    total_samples_processed = n_samples    
    avg_metrics = {
        'precision': total_metrics['precision'] / (n_samples * len(detector_models)),
        'recall': total_metrics['recall'] / (n_samples * len(detector_models)),
        'confidence_change': (total_metrics['confidence_after'] - total_metrics['confidence_before']) / (n_samples * len(detector_models)),
        'duq': total_metrics['duq'] / (n_samples * len(detector_models))
    }
    
    avg_boxes_before = sum(total_metrics['boxes_per_image_before']) / len(total_metrics['boxes_per_image_before'])
    avg_boxes_after = sum(total_metrics['boxes_per_image_after']) / len(total_metrics['boxes_per_image_after'])
    
    box_count_change = ((avg_boxes_after - avg_boxes_before) / avg_boxes_before * 100) if avg_boxes_before > 0 else 0.0
    
    avg_f1 = 2 * avg_metrics['precision'] * avg_metrics['recall'] / (avg_metrics['precision'] + avg_metrics['recall']) if avg_metrics['precision'] + avg_metrics['recall'] > 0 else 0
    print("\n Each detector test results:")
    detector_results = {}
    
    for detector_name, metrics in detector_metrics.items():
        det_precision = metrics['precision'] / n_samples
        det_recall = metrics['recall'] / n_samples
        det_confidence_change = (metrics['confidence_after'] - metrics['confidence_before']) / n_samples
        det_duq = metrics['duq'] / n_samples
        
        det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if det_precision + det_recall > 0 else 0
        
        det_boxes_before = sum(metrics['boxes_per_image_before']) / len(metrics['boxes_per_image_before']) if metrics['boxes_per_image_before'] else 0
        det_boxes_after = sum(metrics['boxes_per_image_after']) / len(metrics['boxes_per_image_after']) if metrics['boxes_per_image_after'] else 0
        det_box_change = ((det_boxes_after - det_boxes_before) / det_boxes_before * 100) if det_boxes_before > 0 else 0.0
        
        print(f"\n Detector: {detector_name}")
        print(f"F1-score: {det_f1:.4f}")
        print(f"Confidence_change: {det_confidence_change:.4f}")
        print(f"DUQ: {det_duq:.4f}")
        print(f"Precision: {det_precision:.4f}")
        print(f"Recall: {det_recall:.4f}")
        print(f"boxes_before: {det_boxes_before:.2f}")
        print(f"boxes_after: {det_boxes_after:.2f}")
        print(f"box_change: {det_box_change:.2f}%")
        
        detector_results[detector_name] = {
            'f1': det_f1,
            'confidence_change': det_confidence_change,
            'duq': det_duq,
            'box_change': det_box_change
        }
    
    return avg_f1, avg_metrics['confidence_change'], avg_metrics['duq'], box_count_change, detector_results

def init_detectors(device):
    
    detectors_dict = {} 
    try:
        # Init the RetinaFace
        from face_detection.retinaface.RetinaFaceDetector import init_detector
        retinaface = init_detector(device=device)
        detectors_dict["retinaface"] = (retinaface, "retinaface")
        print("Initialize RetinaFace successfully.")
        
        # Init the S3FD
        from face_detection.S3FD.S3FDDetector import init_detector as init_s3fd
        s3fd = init_s3fd(device=device)
        if s3fd is not None:
            detectors_dict["s3fd"] = (s3fd, "s3fd")
            print("Initialize S3FD successfully.")
         
        # Init the DSFD
        from face_detection.dsfd.DSFDDetector import init_dsfd_detector
        dsfd = init_dsfd_detector(device=device, weights_path=None)
        detectors_dict["dsfd"] = (dsfd, "dsfd")
        print("Initialize DSFD successfully.")
        
    except Exception as e:
        print(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()
    
    return detectors_dict


def train_attacker():
    args_attack = parse()
    print(args_attack)
    device = torch.device('cuda')
    # Init the detectors
    detectors_dict = init_detectors(device)
    print(f"Initialize {len(detectors_dict)} detectors")
    # Init the attacker
    attack_utils = init_get_outputs(args_attack)
    # Init the attacked models
    attack_dataloader, test_dataloader, attgan, attgan_args, stargan_solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
    model_cases = [0, 1, 2, 3]
    import time
    start_time = time.time()
    best_attack_performance = float('inf')
    trained_watermark = torch.load('/home/zhr-23/code/TSDF_code/pert_TSDF_poison.pt').to(attack_utils.device)
    attack_utils.up = trained_watermark
    print("\nFinal evaluation...")

    _, _, _, _ = evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, stargan_solver,
                                          attentiongan_solver,
                                          transform, F, T, G, E, reference, gen_models, attack_utils,
                                          max_samples=50)

    duq_score, adv_loss, _, box_change, detector_results = test_watermark(
        watermark=attack_utils.up,
        test_dataloader=test_dataloader,
        detector_models=[detector for detector, _ in detectors_dict.values()],
        device=device,
        num_test=100)
    
    print("\nDetector result:")
    for detector_name, results in detector_results.items():
        print(f"{detector_name}: F1={results['f1']:.4f}, DUQ={results['duq']:.4f}, box_change={results['box_change']:.2f}%")

if __name__ == "__main__":
    train_attacker()
