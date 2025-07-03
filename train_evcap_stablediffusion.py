import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import argparse
import numpy as np
import utils
from optims import LinearWarmupCosineLRScheduler, set_optimizer
import torch.nn as nn
import torch.nn.functional as F

from dataset.coco_dataset import COCODataset
from models.evcap import EVCap


from common.dist_utils import (
    get_rank,
    init_distributed_mode,
    get_world_size,
)

class UNet(nn.Module):
    """
    UNet architecture for denoising diffusion models.
    This model takes a noisy image and time step and predicts the noise component.
    """
    def __init__(self, dim=768, cond_dim=768, time_emb_dim=128):
        super().__init__()
        print(f"Initializing UNet noise predictor with dim: {dim}, cond_dim: {cond_dim}")
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Conditioning embedding
        self.cond_proj = nn.Linear(cond_dim, dim)
        
        # Down blocks
        self.down1 = nn.Sequential(
            nn.Linear(dim + time_emb_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2)
        )
        
        self.down2 = nn.Sequential(
            nn.Linear(dim * 2 + time_emb_dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
        
        # Middle block
        self.mid = nn.Sequential(
            nn.Linear(dim * 4 + time_emb_dim + dim, dim * 4),  # Added conditioning dim
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
        
        # Up blocks
        self.up1 = nn.Sequential(
            nn.Linear(dim * 8 + time_emb_dim, dim * 2),  # Skip connection from down2
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2)
        )
        
        self.up2 = nn.Sequential(
            nn.Linear(dim * 4 + time_emb_dim, dim),  # Skip connection from down1
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Output
        self.final = nn.Linear(dim, dim)
        
    def forward(self, x, time, cond=None):
        # Time embedding
        t_emb = self.time_mlp(time.view(-1, 1))
        
        # Prepare conditioning if provided
        cond_emb = self.cond_proj(cond) if cond is not None else torch.zeros_like(x)
        
        # Down pass
        x1 = self.down1(torch.cat([x, t_emb], dim=1))
        x2 = self.down2(torch.cat([x1, t_emb], dim=1))
        
        # Middle block with conditioning
        x_mid = self.mid(torch.cat([x2, t_emb, cond_emb], dim=1))
        
        # Up pass with skip connections
        x_up1 = self.up1(torch.cat([x_mid, x2, t_emb], dim=1))
        x_up2 = self.up2(torch.cat([x_up1, x1, t_emb], dim=1))
        
        # Output
        return self.final(x_up2)


class DiffusionModel:
    """
    Diffusion model implementation.
    This class handles the forward and backward diffusion processes.
    """
    def __init__(self, noise_predictor, beta_start=1e-4, beta_end=0.02, n_timesteps=1000):
        self.noise_predictor = noise_predictor
        self.n_timesteps = n_timesteps
        
        # Define beta schedule
        self.beta = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # Pre-compute values for sampling
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha = torch.sqrt(1. - self.alpha)
        
    def to(self, device):
        """Move model and schedule buffers to device"""
        self.noise_predictor = self.noise_predictor.to(device)
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        self.sqrt_alpha = self.sqrt_alpha.to(device)
        self.sqrt_one_minus_alpha = self.sqrt_one_minus_alpha.to(device)
        return self
    
    def diffusion_forward(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to x_0 according to timestep t
        """
        batch_size = x_0.shape[0]
        
        # Get alpha_t
        a_t = self.alpha_hat[t]
        a_t = a_t.view(-1, 1)
        
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Add noise to x_0 according to schedule
        x_t = torch.sqrt(a_t) * x_0 + torch.sqrt(1 - a_t) * noise
        
        return x_t, noise
    
    def sample(self, cond, n_steps=50):
        """
        Sample from the diffusion model using DDIM sampling
        """
        device = next(self.noise_predictor.parameters()).device
        batch_size = cond.shape[0]
        
        # Start from pure noise
        x = torch.randn(batch_size, 768, device=device)
        
        # Sampling iterations
        time_steps = list(range(0, self.n_timesteps, self.n_timesteps // n_steps))
        time_steps = [0] + list(reversed(time_steps[1:]))
        
        for i in range(len(time_steps) - 1):
            t_curr = torch.full((batch_size,), time_steps[i], device=device, dtype=torch.long)
            t_next = torch.full((batch_size,), time_steps[i+1], device=device, dtype=torch.long)
            
            # Get alpha values
            alpha_curr = self.alpha_hat[t_curr].view(-1, 1)
            alpha_next = self.alpha_hat[t_next].view(-1, 1)
            
            # Predict noise
            pred_noise = self.noise_predictor(x, t_curr / self.n_timesteps, cond)
            
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_curr) * pred_noise) / torch.sqrt(alpha_curr)
            
            # Get next x (deterministic DDIM step)
            x = torch.sqrt(alpha_next) * x_0_pred + torch.sqrt(1 - alpha_next) * pred_noise
        
        return x
    
    def ddpm_sample_step(self, x_t, t, cond=None):
        """
        Single step of DDPM sampling for inference
        """
        device = x_t.device
        
        # Predict noise
        pred_noise = self.noise_predictor(x_t, t / self.n_timesteps, cond)
        
        # Get alpha values
        alpha_t = self.alpha_hat[t]
        alpha_t_prev = self.alpha_hat[t-1] if t > 0 else torch.ones_like(alpha_t)
        
        # Coefficients
        c1 = torch.sqrt(alpha_t_prev / alpha_t)
        c2 = torch.sqrt(1 - alpha_t_prev) - torch.sqrt(alpha_t_prev / alpha_t) * torch.sqrt(1 - alpha_t)
        
        # DDPM update
        mean = c1 * x_t + c2 * pred_noise
        
        # Add noise for t > 0
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            x_t_prev = mean + sigma_t.view(-1, 1) * noise
        else:
            x_t_prev = mean
            
        return x_t_prev
        
    def train_step(self, x_0, cond=None):
        """
        Single training step for diffusion model
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device)
        
        # Add noise to get x_t
        x_t, noise = self.diffusion_forward(x_0, t)
        
        # Predict the noise
        noise_pred = self.noise_predictor(x_t, t / self.n_timesteps, cond)
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, cur_epoch, output_dir, noise_predictor=None, diffusion_optimizer=None):
    """
    Save the checkpoint at the current epoch.
    """
    model_no_ddp = model.module if hasattr(model, "module") else model
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    }
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": cur_epoch,
    }
    
    # Save diffusion model if provided
    if noise_predictor is not None:
        noise_model_no_ddp = noise_predictor.module if hasattr(noise_predictor, "module") else noise_predictor
        noise_state_dict = noise_model_no_ddp.state_dict()
        save_obj["noise_predictor"] = noise_state_dict
    
    if diffusion_optimizer is not None:
        save_obj["diffusion_optimizer"] = diffusion_optimizer.state_dict()
    
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, output_dir))
    torch.save(save_obj, output_dir)


def train(dataset, model, args):
    device = torch.device(f"cuda:{get_rank()}")
    batch_size = args.bs
    epochs = args.epochs
    accum_grad_iters = 1
    output_dir = args.out_dir
    lambda_diffusion = getattr(args, 'lambda_diffusion', 1.0)  # Weight for diffusion loss
    n_timesteps = getattr(args, 'n_timesteps', 1000)  # Number of diffusion timesteps
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[get_rank()])
    else: 
        sampler = None
        model = model.to(device)

    # Initialize diffusion model
    noise_predictor = UNet(dim=768, cond_dim=768).to(device)
    if args.distributed:
        noise_predictor = torch.nn.parallel.DistributedDataParallel(noise_predictor, device_ids=[get_rank()])
    
    diffusion = DiffusionModel(noise_predictor, n_timesteps=n_timesteps).to(device)
    
    # Optimizers
    diffusion_optimizer = torch.optim.Adam(noise_predictor.parameters(), lr=1e-4, weight_decay=0.05)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=sampler, shuffle=False, drop_last=True)
    
    model.train()
    noise_predictor.train()
    
    optimizer = set_optimizer(model, init_lr=1e-4, weight_decay=0.05)
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=epochs,
        iters_per_epoch=len(train_dataloader),
        min_lr=8e-5,
        init_lr=1e-4,
        decay_rate=None,
        warmup_start_lr=1e-6,
        warmup_steps=5000,
    )
    
    # Automatic mixed precision if enabled
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    use_amp = scaler is not None
    print('use_amp', use_amp)

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.update(loss=1000.0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        print_freq = 50
        header = 'Train Epoch: [{}]'.format(epoch)
                
        # Projection layers to match dimensions
        projection_x = nn.Linear(5120, 768).to(device).half()
        projection_cond = nn.Linear(5120, 768).to(device).half()
        
        for idx, samples in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
            samples['image'] = samples['image'].to(device)
            scheduler.step(cur_epoch=epoch, cur_step=idx)                
            
            with torch.no_grad():
                # Get Q-former features from EVCap model
                cond_features, _ = model.module.encode_img(samples['image']) if args.distributed else model.encode_img(samples['image'])
                cond_features = cond_features.mean(dim=1)
                
                # Project features to match diffusion dimensions
                z_0 = projection_x(cond_features.detach())  # Target features
                z_cond = projection_cond(cond_features.detach())  # Conditioning features
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Train EVCap model
                model_out = model(samples)
                model_loss = model_out["loss"]
                
                # Train diffusion model
                diffusion_loss = diffusion.train_step(z_0, z_cond)
                
                # Combined loss
                total_loss = model_loss + lambda_diffusion * diffusion_loss
                
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
                
            if (idx + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.step(diffusion_optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                    diffusion_optimizer.step()
                optimizer.zero_grad()
                diffusion_optimizer.zero_grad()
                
            metric_logger.update(loss=total_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(diffusion_loss=diffusion_loss.item())
            
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())
 
        if epoch == epochs - 1:
            # Save combined checkpoint
            output_dir_model = os.path.join(output_dir, f"{epoch:03d}.pt")
            save_checkpoint(model, optimizer, epoch, output_dir_model, noise_predictor, diffusion_optimizer)
            
            # Also save diffusion model separately for easier loading in sampling script
            diffusion_model_path = os.path.join(output_dir, f"diffusion_model_{epoch:03d}.pt")
            if get_rank() == 0:  # Only save once in distributed training
                noise_model_to_save = noise_predictor.module if hasattr(noise_predictor, "module") else noise_predictor
                torch.save(noise_model_to_save.state_dict(), diffusion_model_path)
                print(f"Diffusion model saved to {diffusion_model_path}")
                
    return model


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--device', default = 'cuda', help = 'gpu for training')
    parser.add_argument('--distributed', default = True)
    parser.add_argument('--amp', default = True)
    parser.add_argument('--dist_url', default = "env://")
    parser.add_argument('--world_size', type = int, default = 1)
    parser.add_argument('--num_query_token_txt', type = int, default = 8)
    parser.add_argument('--topn', type = int, default = 9)
    parser.add_argument('--n_timesteps', type=int, default=1000, help='Number of timesteps in diffusion process')
    parser.add_argument('--lambda_diffusion', type=float, default=1.0, help='Weight for diffusion loss')
    parser.add_argument('--disable_random_seed', action = 'store_true', default = False, help = 'set random seed for reproducing')
    parser.add_argument('--random_seed', type = int, default = 42, help = 'set random seed for reproducing')
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    if not args.disable_random_seed:
        set_seed(args.random_seed)
    init_distributed_mode(args)
    print(f'args: {vars(args)}')
    data_root = 'data/coco/coco2014'
    dataset = COCODataset(data_root=data_root)
    model_type = "lmsys/vicuna-13b-v1.5"
    model = EVCap(
            ext_path = 'ext_data/ext_memory_lvis.pkl',
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            num_query_token_txt=args.num_query_token_txt,
            topn = args.topn,
            llama_model=model_type,
            prompt_path="prompts/prompt_evcap.txt",
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=128,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    )
    train(dataset, model, args)


if __name__ == '__main__':
    main()