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

class ConditionalFlow(nn.Module):
    def __init__(self, dim: int = 768, cond_dim: int = 768, h: int = 1024):
        super().__init__()
        print("Initializing cond flow model with dim:", dim, "cond_dim:", cond_dim, "h:", h)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim)
            
        )
    
    def forward(self, t:torch.Tensor, x_t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        if cond is not None:
            print("cond is not None")
            inp = torch.cat((x_t, t, cond), dim=-1)
        else:
            print("cond is None")
            inp = torch.cat((x_t, t), dim=-1)
        return self.net(inp)

    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Euler-like update: x_t + Δt * f(x_t, t, cond)
        """
        #the implementation is similar to the original one, but with lesser accuracy
    # def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
    #     t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        
    #     return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)

        if len(t_start.shape) == 1:
            t_start = t_start.view(-1, 1)
        if len(t_end.shape) == 1:
            t_end = t_end.view(-1, 1)

        dt = t_end - t_start
        t_mid = t_start + dt / 2
        velocity_start = self(x_t=x_t, t=t_start, cond=cond)
        x_mid = x_t + velocity_start * (dt / 2)
        velocity_mid = self(x_t=x_mid, t=t_mid, cond=cond)
        x_next = x_t + velocity_mid * dt
        return x_next

        
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model,optimizer, cur_epoch, output_dir):
    """
    Save the checkpoint at the current epoch.
    """
    model_no_ddp = model
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
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, output_dir))
    torch.save(save_obj, output_dir)


def train(dataset, model, args):
    device = torch.device(f"cuda:{get_rank()}")
    batch_size = args.bs
    epochs = args.epochs
    accum_grad_iters = 1
    output_dir = args.out_dir
    lambda_flow = getattr(args, 'lambda_flow', 1.0) #this is the weight for the flow_loss variable
    
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

    flow_model = ConditionalFlow(dim=768).to(device)
    if args.distributed:
        flow_model = torch.nn.parallel.DistributedDataParallel(flow_model, device_ids=[get_rank()])
    flow_model.train()
    # flow_optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4, weight_decay=0.05)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=sampler,shuffle=False, drop_last=True)
    model.train()
    optimizer = set_optimizer(model, init_lr=1e-4, weight_decay=0.05)
    scheduler = LinearWarmupCosineLRScheduler(optimizer= optimizer,
                max_epoch=epochs,
                iters_per_epoch=len(train_dataloader),
                min_lr=8e-5,
                init_lr=1e-4,
                decay_rate=None,
                warmup_start_lr=1e-6,
                warmup_steps=5000,)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    use_amp = scaler is not None
    print('use_amp', use_amp)
    
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
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
        for idx, samples in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
            samples['image'] = samples['image'].to(device)
            scheduler.step(cur_epoch=epoch, cur_step=idx)    
            
            projection_x = nn.Linear(5120, 768).to(device).half()
            projection_cond = nn.Linear(5120, 768).to(device).half()
            
            with torch.no_grad():
                cond_features, _ = model.module.encode_img(samples['image']) if args.distributed else model.encode_img(samples['image'])
                cond_features = cond_features.mean(dim=1)
                #without projection to fit the dimensions
                # z = cond_features.detach()
                # eps = torch.randn_like(z)
                # t = torch.rand(z.size(0), 1).to(z.device)
                # x_t = z + eps * (1.0 - t)# sample
                
                #with projction to fit dimensions
                z = projection_cond(cond_features.detach())
                eps = torch.randn_like(z)
                t = torch.rand(z.size(0), 1).to(z.device)
                x_t = z + eps * (1.0 - t)  # Now matches expected dimensions
            

            with torch.cuda.amp.autocast(enabled=use_amp):
                model_out = model(samples)
                model_loss = model_out["loss"]
                print(f"t shape: {t.shape}")
                print(f"x_t shape: {x_t.shape}")
                print(f"z (cond) shape: {z.shape}")
                pred_eps = flow_model(t=t, x_t=x_t, cond=z)
                flow_loss = F.mse_loss(pred_eps, eps)
                total_loss = model_loss + lambda_flow * flow_loss
                
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            if (idx + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.step(flow_optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                    # flow_optimizer.step()
                optimizer.zero_grad()
                # flow_optimizer.zero_grad()
            metric_logger.update(loss=total_loss.item())
            metric_logger.update(lr = optimizer.param_groups[0]["lr"])
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())
 
        if epoch == epochs - 1:
            output_dir_model = os.path.join(output_dir, f"{epoch:03d}.pt")
            save_checkpoint(model, optimizer, epoch, output_dir_model)
    return model

def train_flow_matching_step(flow_model, x0, cond, optimizer):
    device = x0.device
    t = torch.rand(x0.size(0), 1).to(device) # sample t ~ Uniform(0,1)
    eps = torch.randn_like(x0)
    xt = x0 + eps * t
    
    pred_eps = flow_model(t, xt, cond)
    loss = torch.nn.functional.mse_loss(pred_eps, eps)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

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
