from utils import get_data_iters, prepare_sub_folder, write_loss, get_config
import argparse
from trainer import HiSD_Trainer
import torch
import os
import tensorboardX
import shutil
import random
import torch.nn as nn

def save_discriminator(trainer, checkpoint_directory, iteration):
    """Saves only the discriminator and its optimizer."""
    dis_state = {
        'dis': trainer.models.dis.state_dict(),
        'dis_opt': trainer.dis_opt.state_dict(),
        'iterations': iteration
    }
    dis_name = os.path.join(checkpoint_directory, 'dis_%08d.pt' % (iteration + 1))
    torch.save(dis_state, dis_name)
    print(f"Saved new discriminator checkpoint to {dis_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--generator_checkpoint', type=str, required=True, help='Path to the pre-trained generator checkpoint file.')
    parser.add_argument('--output_path', type=str, default='.', help="Path to save the new discriminator.")
    parser.add_argument("--gpus", nargs='+', required=True)
    parser.add_argument('--training_iterations', type=int, default=100000, help='Number of iterations to train the new discriminator.')
    parser.add_argument('--snapshot_save_iter', type=int, default=10000, help='Iteration interval to save discriminator checkpoint.')
    parser.add_argument('--log_iter', type=int, default=100, help='Iteration interval to log loss.')
    opts = parser.parse_args()

    from torch.backends import cudnn
    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0] + "_train_dis"
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, _ = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

    # Setup model and trainer
    multi_gpus = len(opts.gpus) > 1
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opts.gpus)
    trainer = HiSD_Trainer(config, multi_gpus=multi_gpus)

    # Load pre-trained generator
    print(f"Loading pre-trained generator from: {opts.generator_checkpoint}")
    gen_state_dict = torch.load(opts.generator_checkpoint)
    trainer.models.gen.load_state_dict(gen_state_dict['gen'])
    print("Generator loaded. Discriminator is randomly initialized.")

    # Freeze generator weights
    for param in trainer.models.gen.parameters():
        param.requires_grad = False
    
    # Send to GPU
    if multi_gpus:
        trainer.cuda(int(opts.gpus[0]))
        print("Using GPUs: %s" % str(opts.gpus))
        trainer.models = torch.nn.DataParallel(trainer.models, device_ids=[int(gpu) for gpu in opts.gpus])
    else:
        trainer.cuda(int(opts.gpus[0]))
        
    trainer.models.gen.eval() # Set generator to evaluation mode

    # Setup data loader
    train_iters = get_data_iters(config, opts.gpus)
    tags = list(range(len(train_iters)))

    print(f"Starting discriminator training for {opts.training_iterations} iterations...")
    import time
    start = time.time()
    
    iterations = 0
    while iterations < opts.training_iterations:
        i = random.sample(tags, 1)[0]
        j, j_trg = random.sample(list(range(len(train_iters[i]))), 2)
        x, y = train_iters[i][j].next()
        train_iters[i][j].preload()

        # Update discriminator only
        D_adv_loss = trainer.update_D(x, y, j_trg)
        
        torch.cuda.synchronize()

        if (iterations + 1) % opts.log_iter == 0:
            train_writer.add_scalar('D/adv_loss', D_adv_loss, iterations)
            now = time.time()
            print(f"[#{iterations + 1:06d}/{opts.training_iterations:d}] D_Loss: {D_adv_loss:.4f} | {now - start:5.2f}s")
            start = now

        if (iterations + 1) % opts.snapshot_save_iter == 0:
            save_discriminator(trainer, checkpoint_directory, iterations)

        iterations += 1
    
    # Save final discriminator
    save_discriminator(trainer, checkpoint_directory, iterations - 1)
    print('Finish training new discriminator!')

if __name__ == '__main__':
    main() 
    
    # python train_discriminator.py \
    #     --config /root/tf-logs/FOUND_code/HiSD/configs/celeba-hq_256.yaml \
    #     --generator_checkpoint /root/tf-logs/FOUND_code/HiSD/gen_00600000.pt \
    #     --output_path /root/tf-logs/FOUND_code/HiSD/new_discriminator \
    #     --gpus 0