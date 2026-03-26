from utils import get_data_iters, prepare_sub_folder, write_loss, get_config, write_2images
import argparse
from trainer import HiSD_Trainer
import torch
import os
import sys
import tensorboardX
import shutil
import random
import yaml
import copy

def save_generator(trainer, checkpoint_directory, iteration):
    """Saves only the generator and its optimizer."""
    this_model = trainer.models.module if hasattr(trainer.models, 'module') else trainer.models
    gen_state = {
        'gen': this_model.gen.state_dict(),
        'gen_test': this_model.gen_test.state_dict(),
        'gen_opt': trainer.gen_opt.state_dict(),
        'iterations': iteration
    }
    gen_name = os.path.join(checkpoint_directory, 'gen_only_%08d.pt' % (iteration + 1))
    torch.save(gen_state, gen_name)
    print(f"Saved generator checkpoint to {gen_name}")

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to the config file for finetuning.')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file to load model for finetuning.')
parser.add_argument('--discriminator_checkpoint', type=str, default=None, help='Path to the separately trained discriminator checkpoint file.')
parser.add_argument('--data_root', type=str, required=True, help='Path to the new dataset root directory.')
parser.add_argument('--output_path', type=str, default='.', help="Outputs path for finetuning.")
parser.add_argument('--generator_save_iter', type=int, default=5000, help='Iteration interval to save generator-only checkpoint.')
parser.add_argument('--lr_factor', type=float, default=1.0, help='Factor to multiply the learning rates.')
parser.add_argument("--gpus", nargs='+')
opts = parser.parse_args()

from torch.backends import cudnn

# For fast training
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)

# Override dataset path
config['data_root'] = opts.data_root
# Adjust learning rates if requested
if opts.lr_factor != 1.0:
    print(f"Adjusting learning rates by factor: {opts.lr_factor}")
    config['lr_dis'] *= opts.lr_factor
    config['lr_gen_mappers'] *= opts.lr_factor
    config['lr_gen_others'] *= opts.lr_factor
    
total_iterations = config['total_iterations']

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0] + "_finetune"
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# Save the used config for this finetuning run
with open(os.path.join(output_directory, 'config.yaml'), 'w') as f:
    yaml.dump(config, f)

# Setup model
multi_gpus = len(opts.gpus) > 1
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opts.gpus)
trainer = HiSD_Trainer(config, multi_gpus=multi_gpus)

# 修改加载逻辑，直接从文件加载生成器，而不是调用resume方法
print(f"Loading pre-trained generator from: {opts.checkpoint}")
gen_state_dict = torch.load(opts.checkpoint)
if 'gen' in gen_state_dict:
    trainer.models.gen.load_state_dict(gen_state_dict['gen'])
    if 'gen_test' in gen_state_dict:
        trainer.models.gen_test.load_state_dict(gen_state_dict['gen_test'])
    else:
        # 如果没有gen_test，就复制gen
        trainer.models.gen_test = copy.deepcopy(trainer.models.gen)
        
    # 尝试加载生成器优化器状态
    if 'gen_opt' in gen_state_dict:
        trainer.gen_opt.load_state_dict(gen_state_dict['gen_opt'])
        # 移动优化器状态到GPU
        for state in trainer.gen_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(int(opts.gpus[0]))
        print("Loaded generator optimizer state")
else:
    # 如果没有嵌套结构，直接尝试加载
    trainer.models.gen.load_state_dict(gen_state_dict)
    trainer.models.gen_test = copy.deepcopy(trainer.models.gen)
    print("Warning: No generator optimizer state found. Optimizer reinitialized.")

# 根据参数加载判别器
if opts.discriminator_checkpoint is not None:
    print(f"Loading separately trained discriminator from: {opts.discriminator_checkpoint}")
    dis_state_dict = torch.load(opts.discriminator_checkpoint)
    if 'dis' in dis_state_dict:
        trainer.models.dis.load_state_dict(dis_state_dict['dis'])
        if 'dis_opt' in dis_state_dict:
            trainer.dis_opt.load_state_dict(dis_state_dict['dis_opt'])
            # 移动优化器状态到GPU
            for state in trainer.dis_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(int(opts.gpus[0]))
    else:
        # 如果没有嵌套结构，直接尝试加载
        trainer.models.dis.load_state_dict(dis_state_dict)
    print("Separately trained discriminator loaded successfully")

iterations = 0

if multi_gpus:
    trainer.cuda(int(opts.gpus[0]))
    print("Using GPUs: %s" % str(opts.gpus))
    trainer.models= torch.nn.DataParallel(trainer.models, device_ids=[int(gpu) for gpu in opts.gpus])
else:
    trainer.cuda(int(opts.gpus[0]))

# Setup data loader
train_iters = get_data_iters(config, opts.gpus)
tags = list(range(len(train_iters)))

import time
start = time.time()
while True:
    """
    i: tag
    j: source attribute, j_trg: target attribute
    x: image, y: tag-irrelevant conditions
    """
    i = random.sample(tags, 1)[0]
    j, j_trg = random.sample(list(range(len(train_iters[i]))), 2) 
    x, y = train_iters[i][j].next()
    train_iters[i][j].preload()

    G_adv, G_sty, G_rec, D_adv = trainer.update(x, y, i, j, j_trg)

    if (iterations + 1) % config['image_save_iter'] == 0:
        for i in range(len(train_iters)):
            j, j_trg = random.sample(list(range(len(train_iters[i]))), 2) 

            x, _ = train_iters[i][j].next()
            x_trg, _ = train_iters[i][j_trg].next()
            train_iters[i][j].preload()
            train_iters[i][j_trg].preload()

            test_image_outputs = trainer.sample(x, x_trg, j, j_trg, i)
            write_2images(test_image_outputs,
                          config['batch_size'], 
                          image_directory, 'sample_%08d_%s_%s_to_%s' % (iterations + 1, config['tags'][i]['name'], config['tags'][i]['attributes'][j]['name'], config['tags'][i]['attributes'][j_trg]['name']))
    
    torch.cuda.synchronize()

    if (iterations + 1) % config['log_iter'] == 0:
        write_loss(iterations, trainer, train_writer)
        now = time.time()
        print(f"[#{iterations + 1:06d}|{total_iterations:d}] {now - start:5.2f}s")
        start = now

    if (iterations + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(checkpoint_directory, iterations)

    # 每隔一定迭代次数保存单独的生成器模型
    if (iterations + 1) % opts.generator_save_iter == 0:
        save_generator(trainer, checkpoint_directory, iterations)

    if (iterations + 1) == total_iterations:
        print('Finish finetuning!')
        # 在训练结束时也保存一次生成器模型
        save_generator(trainer, checkpoint_directory, iterations)
        exit(0)

    iterations += 1 
    
    
# python finetune.py \
#     --config /root/tf-logs/FOUND_code/HiSD/configs/celeba-hq_256.yaml \
#     --checkpoint /root/tf-logs/FOUND_code/HiSD/gen_00600000.pt \
#     --discriminator_checkpoint /root/tf-logs/FOUND_code/HiSD/new_discriminator/outputs/celeba-hq_256_train_dis/checkpoints/dis_00010000.pt \
#     --data_root /root/autodl-tmp/Celeba_watermarket \
#     --output_path /root/tf-logs/FOUND_code/HiSD/examples \
#     --generator_save_iter 1000 \
#     --gpus 0

