import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from models.evcap import EVCap
from train_evcap_stablediffusion import UNet, DiffusionModel


def load_image(image_path, image_size=224):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def load_models(model_path, diffusion_model_path=None, device='cuda'):
    """Load the EVCap model and diffusion model."""
    # Load EVCap model configuration
    model_type = "lmsys/vicuna-13b-v1.5"
    model = EVCap(
        ext_path='ext_data/ext_memory_lvis.pkl',
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model=model_type,
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=128,
        end_sym='\n',
        low_resource=False,
        device_8bit=0,
    )
    
    # Load model state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()
    
    # Initialize and load diffusion model if path is provided
    diffusion_model = None
    if diffusion_model_path:
        noise_predictor = UNet(dim=768, cond_dim=768)
        
        # Load state dict
        diffusion_state = torch.load(diffusion_model_path, map_location=device)
        if 'noise_predictor' in diffusion_state:
            noise_predictor.load_state_dict(diffusion_state['noise_predictor'])
        else:
            noise_predictor.load_state_dict(diffusion_state)
            
        # Create diffusion model
        diffusion_model = DiffusionModel(noise_predictor, n_timesteps=1000).to(device)
        noise_predictor.eval()
        
    return model, diffusion_model


def sample_from_diffusion(diffusion_model, cond_vector, num_steps=100, sample_batch_size=1, device='cuda'):
    """Generate samples using the diffusion model's sampling method."""
    # Create a projection layer to match dimensions if needed
    if cond_vector.shape[-1] != 768:
        proj = torch.nn.Linear(cond_vector.shape[-1], 768).to(device)
        z_cond = proj(cond_vector)
    else:
        z_cond = cond_vector
    
    # Generate samples
    with torch.no_grad():
        # Use diffusion model's sample method
        samples = diffusion_model.sample(z_cond, n_steps=num_steps)
            
    return samples


def generate_conditional_samples(model_path, diffusion_model_path, image_path, num_samples=5, num_steps=100, device='cuda'):
    """Generate samples conditioned on image features."""
    # Load models
    model, diffusion_model = load_models(model_path, diffusion_model_path, device)
    
    # Load image
    image = load_image(image_path).to(device)
    
    # Extract features
    with torch.no_grad():
        cond_features, _ = model.encode_img(image)
        cond_features = cond_features.mean(dim=1)  # Average over tokens
    
    # Project features if needed
    if cond_features.shape[-1] != 768:
        proj = torch.nn.Linear(cond_features.shape[-1], 768).to(device)
        z_cond = proj(cond_features)
    else:
        z_cond = cond_features
    
    # Generate samples
    all_samples = []
    for i in range(num_samples):
        samples = sample_from_diffusion(
            diffusion_model,
            z_cond,
            num_steps=num_steps,
            device=device
        )
        all_samples.append(samples)
    
    return torch.cat(all_samples, dim=0), cond_features


def compare_samples_to_original(samples, original, output_dir):
    """Compare generated samples to original features and save statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute statistics
    distances = []
    for i in range(samples.shape[0]):
        sample = samples[i]
        distance = torch.norm(sample - original, dim=-1).mean().item()
        distances.append(distance)
    
    # Save statistics
    stats = {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'distances': distances
    }
    
    stats_file = os.path.join(output_dir, 'sample_statistics.pt')
    torch.save(stats, stats_file)
    
    # Also save in plain text
    with open(os.path.join(output_dir, 'sample_statistics.txt'), 'w') as f:
        for k, v in stats.items():
            if k != 'distances':
                f.write(f"{k}: {v}\n")
    
    # Save all samples and original
    samples_file = os.path.join(output_dir, 'samples.pt')
    torch.save({
        'samples': samples.cpu(),
        'original': original.cpu()
    }, samples_file)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Sample from diffusion model conditioned on image features")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained EVCap model checkpoint")
    parser.add_argument("--diffusion_model_path", type=str, required=True, help="Path to the trained diffusion model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="diffusion_samples", help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of diffusion sampling steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples from image: {args.image_path}")
    samples, original = generate_conditional_samples(
        args.model_path,
        args.diffusion_model_path,
        args.image_path,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=args.device
    )
    
    # Compare and save
    print("Analyzing samples...")
    stats = compare_samples_to_original(samples, original, args.output_dir)
    
    print(f"Generated {args.num_samples} samples. Statistics:")
    print(f"Mean distance from original: {stats['mean_distance']:.6f}")
    print(f"Std deviation of distances: {stats['std_distance']:.6f}")
    print(f"Min/Max distances: {stats['min_distance']:.6f} / {stats['max_distance']:.6f}")
    print(f"Results saved to {args.output_dir}")
    
    
if __name__ == "__main__":
    main()