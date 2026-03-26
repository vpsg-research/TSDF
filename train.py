import argparse
import copy
import json
import os
from os.path import join
import numpy as np
from tqdm import tqdm
from random import shuffle
import torch
import get_output
from model_data_prepare import prepare
from evaluate_fid import evaluate_multiple_models
from attgan.data import check_attribute_conflict
import torch.nn.functional as TFunction
import torch.nn.functional as f
import torch.nn as nn
from face_poison_integration import optimize_watermark_with_multi_detector_features


do_poison = True
do_feature_ensemble = True
do_end_to_end = True
batch_evaluate = True

def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack


# Init the attacker
def init_get_outputs(args_attack):
    get_output_models = get_output.get_all_features(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                                    epsilon=args_attack.attacks.epsilon, args=args_attack.attacks)
    return get_output_models


def just_mean(d_grads):
    d_grads = torch.stack([d for d in d_grads])
    return torch.mean(d_grads, dim=0)

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def kl_divergence(p, q):
    p = TFunction.softmax(p.view(p.size(0), -1), dim=1)
    q = TFunction.softmax(q.view(q.size(0), -1), dim=1)

    epsilon = 1e-10
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)

    kl_div = torch.sum(p * torch.log(p/q), dim=1).mean()
    return kl_div


def perform_AttGAN(attack, input_imgs, original_imgs, attibutes, attgan_model, attgan_parse, rand_param):
    att_b_list = [attibutes]
    # No need to attack all attributes
    # for i in range(attgan_parse.n_attrs):
    #     tmp = attibutes.clone()
    #     tmp[:, i] = 1 - tmp[:, i]
    #     tmp = check_attribute_conflict(tmp, attgan_parse.attrs[i], attgan_parse.attrs)
    #     att_b_list.append(tmp)
    # att_b_list = [att_b_list[6]]
    for i, att_b in enumerate(att_b_list):
        att_b_ = (att_b * 2 - 1) * attgan_parse.thres_int
        if i > 0:
            att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_parse.test_int / attgan_parse.thres_int
        with torch.no_grad():
            gen_noattack, no_attack_middle = attgan_model.G(original_imgs, att_b_)
        adv_gen, adv_gen_middle = attack.get_attgan_features(input_imgs, att_b_, attgan_model, rand_param)

    return [gen_noattack, adv_gen], [no_attack_middle[-1], adv_gen_middle[-1]]


def perform_HiSD(attack, input_imgs, original_imgs, reference_img, E_model, F_model, T_model, G_model, EFTG_models, rand_param):
    with torch.no_grad():
        # get the original deepfake images
        c = E_model(original_imgs)
        c_trg = c
        s_trg = F_model(reference_img, 1)
        c_trg = T_model(c_trg, s_trg, 1)
        x_trg = G_model(c_trg)

    adv_x_trg, adv_c = attack.get_hisd_features(input_imgs.cuda(), reference_img, F_model, T_model, G_model, E_model, EFTG_models, rand_param)
    return [x_trg, adv_x_trg], [c, adv_c]


def select_model_to_get_feature_pairs(case, img, ori_imgs, reference, attribute_c, attribute_attgan, attack, stargan_s, atggan_s,
                                      attgan_s, attgan_args, EE, FF, TT, GG, g_models, reconstruct=128, attr_aug= False):
    if attr_aug:
        rand_q = np.random.rand()
    else:
        rand_q = 0
    if case == 0:
        # print('attacking stargan...')
        output_pair, middle_pair = stargan_s.perform_stargan(img, ori_imgs, attribute_c, attack, rand_q)
    elif case == 1:
        # print('attacking attentiongan...')
        output_pair, middle_pair = atggan_s.perform_attentiongan(img, ori_imgs, attribute_c, attack, rand_q)
    elif case == 2:
        # print('attacking AttGan...')
        output_pair, middle_pair = perform_AttGAN(attack, img, ori_imgs, attribute_attgan, attgan_s, attgan_args, rand_q)
    elif case == 3:
        # print('attacking HiSD...')
        output_pair, middle_pair = perform_HiSD(attack, img, ori_imgs, reference, EE, FF, TT, GG, g_models, rand_q)
    else:
        raise NotImplementedError('wrong code!')

    # resize feature outputs
    new_middle_pair = []
    for middle in middle_pair:
        new_middle = torch.nn.functional.interpolate(middle, (reconstruct, reconstruct), mode='bilinear')  # 8, 256, 128, 128
        new_middle_pair.append(new_middle)

    return output_pair, new_middle_pair


def DI(X_in):
    import torch.nn.functional as F

    rnd = np.random.randint(256, 290, size=1)[0]
    h_rem = 290 - rnd
    w_rem = 290 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.5:
        X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_right, pad_top, pad_bottom), mode='constant',
                      value=0)
        return F.interpolate(X_out, (256, 256))
    else:
        return F.interpolate(X_in, (256, 256))

def calculate_precision(original_dets, watermarked_dets):
    if sum(watermarked_dets) == 0:
        return 0.0
    true_positives = sum(1 for o, w in zip(original_dets, watermarked_dets) if o > 0 and w > 0)
    return true_positives / sum(watermarked_dets)


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


def enhance_feature_contrast(clean_feat, adv_feat, temperature=1.0, alpha=0.5):
    
    def channel_attention(x):
        b, c = x.size()
        channel_att = torch.matmul(x, x.transpose(1, 0)) / (torch.sqrt(torch.tensor(c).float()) + 1e-7)
        channel_att = torch.softmax(channel_att, dim=1)
        return torch.matmul(channel_att, x)

    clean_mean = torch.mean(clean_feat, dim=1, keepdim=True)
    clean_std = torch.std(clean_feat, dim=1, keepdim=True)
    adv_mean = torch.mean(adv_feat, dim=1, keepdim=True)
    adv_std = torch.std(adv_feat, dim=1, keepdim=True)
    
    clean_norm = (clean_feat - clean_mean) / (clean_std + 1e-7)
    adv_norm = (adv_feat - adv_mean) / (adv_std + 1e-7)
    
    clean_att = channel_attention(clean_norm.view(clean_norm.size(0), -1))
    adv_att = channel_attention(adv_norm.view(adv_norm.size(0), -1))
    
    diff_local = adv_norm - clean_norm 
    diff_global = (adv_mean - clean_mean) / (clean_std + 1e-7) 
    diff_structure = adv_att.view_as(adv_norm) - clean_att.view_as(clean_norm) 
    
    w_local = torch.sigmoid(torch.mean(torch.abs(diff_local), dim=1, keepdim=True))
    w_global = torch.sigmoid(torch.mean(torch.abs(diff_global), dim=1, keepdim=True))
    w_structure = torch.sigmoid(torch.mean(torch.abs(diff_structure), dim=1, keepdim=True))
    
    enhancement = (
        w_local * diff_local + 
        w_global * diff_global + 
        w_structure * diff_structure
    )
    
    scale = torch.exp(torch.mean(torch.abs(enhancement), dim=1, keepdim=True) / temperature)
    enhanced_adv_feat = adv_feat + alpha * scale * enhancement
    enhanced_mean = torch.mean(enhanced_adv_feat, dim=1, keepdim=True)
    enhanced_std = torch.std(enhanced_adv_feat, dim=1, keepdim=True)
    enhanced_adv_feat = (enhanced_adv_feat - enhanced_mean) * (adv_std / (enhanced_std + 1e-7)) + adv_mean
    
    return enhanced_adv_feat

def kl_divergence(p, q):
    p = TFunction.softmax(p.view(p.size(0), -1), dim=1)
    q = TFunction.softmax(q.view(q.size(0), -1), dim=1)
    
    epsilon = 1e-10
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    kl_div = torch.sum(p * torch.log(p/q), dim=1).mean()
    
    return kl_div



def train_attacker():
    args_attack = parse()
    print(args_attack)
    device = torch.device('cuda')
    
    detectors_dict = init_detectors(device)
    if not detectors_dict:
        raise RuntimeError("No detectors were successfully initialized.")
    
    print(f"Successfully initialized the detector:{len(detectors_dict)} ")
    
    attack_utils = init_get_outputs(args_attack)
    attack_dataloader, test_dataloader, attgan, attgan_args, stargan_solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()

    model_cases = [0, 1, 2, 3]
    import time
    start_time = time.time()

    # Some hyperparameters
    attack_utils.epsilon = 0.05
    reconstruct_feature_size = 32
    iteration_out = 30
    iteration_in = 3
    alpha = 1e-3
    momentum = 0
    
    model_losses_ema = {case: 0.0 for case in model_cases} 
    model_weights = {case: 1.0 / len(model_cases) for case in model_cases}
    ema_beta = 0.9  
    temperature = 0.1 
    ema_beta = 0.9  
    temperature = 0.1 
    feature_loss_weight = 0.3

    best_attack_performance = float('inf')
    attack_utils.up = attack_utils.up + torch.tensor(
        np.random.uniform(-attack_utils.epsilon, attack_utils.epsilon, attack_utils.up.shape).astype('float32')
    ).to(attack_utils.device)

    if do_poison:
        best_attack_performance = float('inf')
        trained_watermark = torch.load('/home/zhr-23/code/TSDF_code/pert_TSDF.pt').to(attack_utils.device)
        attack_utils.up = trained_watermark

    for t in range(iteration_out):
        print('%dth iter' % t)

        for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
            print(f'{t}th iter, weights: {[f"{w:.2f}" for w in model_weights.values()]}') 
            print('%dth batch' % idx)
            if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
                break
            img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
            att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
            att_a = att_a.type(torch.float)
            if idx % 1 == 0:
                torch.cuda.empty_cache()
            # Feature-Ensemble
            for _ in range(iteration_in):
                attack_utils.up.requires_grad = True
                new_input = img_a + attack_utils.up
                middle_pairs = []
                shuffle(model_cases)
                for case in model_cases:
                    _, mid_pair = select_model_to_get_feature_pairs(
                        case, 
                        DI(new_input),
                        img_a, reference, c_org, att_a,
                        attack_utils, stargan_solver,
                        attentiongan_solver, attgan, attgan_args,
                        E, F, T, G, gen_models, 
                        reconstruct_feature_size
                    )
                    middle_pairs.append(mid_pair)
                    torch.cuda.empty_cache()

                clean_middle_from_models = [middle_pairs[p][0] for p in range(len(middle_pairs))]
                adv_middle_from_models = [middle_pairs[q][1] for q in range(len(middle_pairs))]
                clean_features_cat = torch.cat(clean_middle_from_models, 1)  # concat all clean features in channel dimension
                adv_features_cat = torch.cat(adv_middle_from_models, 1)  # concat all adv features in channel dimension

                adv_features = enhance_feature_contrast(
                                torch.sum(clean_features_cat, 1), 
                                torch.sum(adv_features_cat, 1),
                                temperature=0.1,
                                alpha=0.4
                            )
                deepfake_loss1 = attack_utils.loss_fn(adv_features, torch.sum(adv_features_cat, 1))
                deepfake_loss3= -1 * attack_utils.loss_fn(torch.sum(adv_features_cat, 1), torch.sum(clean_features_cat, 1))
                deepfake_loss =  deepfake_loss3 + 0.001 * deepfake_loss1  
                loss = deepfake_loss
                loss.backward(retain_graph=True)

                grad_c = attack_utils.up.grad.clone().to(attack_utils.device)
                grad_c_hat = grad_c / (torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1e-12)
                attack_utils.up.grad.zero_()

                attack_utils.up.data = attack_utils.up.data - alpha * torch.sign(grad_c_hat)
                attack_utils.up.data = attack_utils.up.data.clamp(-attack_utils.epsilon, attack_utils.epsilon)
                attack_utils.up = attack_utils.up.detach()

            #end_to_end:
            attack_utils.up.requires_grad = True
            torch.cuda.empty_cache()
            if do_poison:
                optimized_watermark, _ = optimize_watermark_with_multi_detector_features(
                    watermark=attack_utils.up,
                    clean_images=img_a,
                    detectors_dict=detectors_dict,
                    device=device,
                    pgd_steps=50,
                    pgd_eps=0.4,
                    pgd_lr=0.02,
                    pgd_momentum=0.5,
                    fusion_type="concat",
                    feature_size=(64, 64)
                )
                attack_utils.up.data = optimized_watermark.data
                attack_utils.up.data = torch.clamp(attack_utils.up.data, -attack_utils.epsilon, attack_utils.epsilon)
                
            new_new_input = img_a + attack_utils.up
            total_weighted_loss = 0
            current_batch_losses = {}

            shuffle(model_cases)
            for case in model_cases:
                out_pair, mid_pair = select_model_to_get_feature_pairs(
                    case, 
                    DI(new_new_input),
                    img_a, reference, c_org, att_a,
                    attack_utils, stargan_solver,
                    attentiongan_solver, attgan, attgan_args,
                    E, F, T, G, gen_models, 
                    reconstruct_feature_size
                )
                
                loss_end_to_end = -1 * attack_utils.loss_fn(out_pair[1], out_pair[0])
                loss_feature = -1 * attack_utils.loss_fn(mid_pair[1], mid_pair[0])
                total_loss_case = (1.0 - feature_loss_weight) * loss_end_to_end + \
                                    feature_loss_weight * loss_feature
                current_batch_losses[case] = total_loss_case.item()
                total_weighted_loss += model_weights[case] * total_loss_case
            
            total_weighted_loss.backward()
            
            grad_c = attack_utils.up.grad.clone().to(attack_utils.device)
            grad_c_hat = grad_c / (torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1e-12)
            attack_utils.up.grad.zero_() 

            grad_c_hat = grad_c_hat + 0.8 * momentum
            momentum = grad_c_hat
            attack_utils.up.data = attack_utils.up.data - alpha * torch.sign(grad_c_hat)
            attack_utils.up.data = attack_utils.up.data.clamp(-attack_utils.epsilon, attack_utils.epsilon)
            attack_utils.up = attack_utils.up.detach()

            with torch.no_grad():
                if do_end_to_end and current_batch_losses:
                    for case in model_cases:
                        model_losses_ema[case] = ema_beta * model_losses_ema[case] + \
                                                 (1 - ema_beta) * current_batch_losses.get(case, 0.0)
                    
                    losses_tensor = torch.tensor([model_losses_ema[case] for case in model_cases], device=attack_utils.device)
                    
                    new_weights = torch.softmax(losses_tensor / temperature, dim=0)
                    for i, case in enumerate(model_cases):
                        model_weights[case] = new_weights[i].item()
                        
                    torch.cuda.empty_cache()
                        
                if attack_utils.up.size(0) > 1:
                    attack_utils.up = attack_utils.up.mean(dim=0, keepdim=True)
        print('up:', torch.max(attack_utils.up), torch.min(attack_utils.up))
        if batch_evaluate and t % 5 == 0:
            _, _, _, _ = evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, stargan_solver,
                                          attentiongan_solver,
                                          transform, F, T, G, E, reference, gen_models, attack_utils,
                                          max_samples=100)
    end_time = time.time()
    print('cost time:', end_time - start_time)
    if attack_utils.up.size(0) > 1:
        attack_utils.up = attack_utils.up.mean(dim=0, keepdim=True)
    print('up:', torch.max(attack_utils.up), torch.min(attack_utils.up))
    
    torch.save(attack_utils.up, 'pert_TSDF_poison.pt')
    _, _, _, _ = evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, stargan_solver,
                                          attentiongan_solver,
                                          transform, F, T, G, E, reference, gen_models, attack_utils,
                                          max_samples=1000)

if __name__ == "__main__":
    train_attacker()
