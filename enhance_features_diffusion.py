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


def enhance_features(diffusion_model, base_features, num_steps=50, device='cuda'):
    """
    Enhance features using the diffusion model with guided sampling.
    This enhances the original features by denoising from a slightly perturbed state.
    """
    # Project features if needed
    if base_features.shape[-1] != 768:
        projection = torch.nn.Linear(base_features.shape[-1], 768).to(device)
        base_features_proj = projection(base_features)
    else:
        base_features_proj = base_features
    
    # Start from a slightly perturbed version of the original features
    # This is instead of starting from pure noise, as we want to stay close to original
    noise_level = 0.3  # 30% noise, 70% original features
    noise = torch.randn_like(base_features_proj) * noise_level
    x_noisy = base_features_proj + noise
    
    # Get timestep corresponding to noise level (t=noise_level*T)
    t_start = int(noise_level * diffusion_model.n_timesteps)
    
    # Sample from diffusion model
    batch_size = base_features_proj.shape[0]
    device = base_features_proj.device
    
    # Start from noise-perturbed features
    x = x_noisy.clone()
    
    # Sampling iterations (from t_start down to 0)
    time_steps = list(range(0, t_start, t_start // num_steps))
    time_steps = time_steps + [0]  # Ensure we reach t=0
    time_steps = list(reversed(time_steps))
    
    for i in range(len(time_steps) - 1):
        t_curr = torch.full((batch_size,), time_steps[i], device=device, dtype=torch.long)
        t_next = torch.full((batch_size,), time_steps[i+1], device=device, dtype=torch.long)
        
        # Predict noise
        with torch.no_grad():
            x = diffusion_model.ddpm_sample_step(x, t_curr, base_features_proj)
            
    return x


def generate_q_former_features(model, image, device='cuda'):
    """Extract Q-former features from an image."""
    with torch.no_grad():
        image = image.to(device)
        cond_features, attn_qform = model.encode_img(image)
        return cond_features


def batch_process_images(model, diffusion_model, image_dir, output_dir, batch_size=8, num_steps=50, device='cuda'):
    """Process a batch of images to generate enhanced features."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process images in batches
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            img = load_image(img_path)
            batch_images.append(img)
            
        # Stack images into a batch
        if batch_images:
            batch_tensor = torch.cat(batch_images, dim=0).to(device)
            
            # Get base Q-former features
            qformer_features = generate_q_former_features(model, batch_tensor, device)
            
            # Get enhanced features for each token
            enhanced_token_features = torch.zeros_like(qformer_features)
            
            # Process each example in the batch
            for b in range(qformer_features.shape[0]):
                # Process each token position separately
                for t in range(qformer_features.shape[1]):
                    # Take single token features and enhance them
                    token_feat = qformer_features[b, t:t+1]
                    enhanced_token_feat = enhance_features(
                        diffusion_model, 
                        token_feat, 
                        num_steps=num_steps,
                        device=device
                    )
                    enhanced_token_features[b, t] = enhanced_token_feat[0]
            
            # Save enhanced features for this batch
            for j, img_file in enumerate(batch_files):
                output_file = os.path.join(output_dir, os.path.splitext(img_file)[0] + '_enhanced_diffusion.pt')
                torch.save({
                    'original_features': qformer_features[j],
                    'enhanced_features': enhanced_token_features[j]
                }, output_file)
    
    print(f"Processed {len(image_files)} images. Enhanced features saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhance Q-former features using diffusion model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained EVCap model checkpoint")
    parser.add_argument("--diffusion_model_path", type=str, required=True, help="Path to the trained diffusion model checkpoint")
    parser.add_argument("--image_path", type=str, help="Path to a single input image")
    parser.add_argument("--image_dir", type=str, help="Directory containing multiple images to process")
    parser.add_argument("--output_dir", type=str, default="enhanced_diffusion_features", help="Directory to save enhanced features")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for directory processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    if not args.image_path and not args.image_dir:
        print("Error: Either --image_path or --image_dir must be provided")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    model, diffusion_model = load_models(args.model_path, args.diffusion_model_path, device=args.device)
    
    if args.image_path:  # Process a single image
        # Load and process the image
        print("Loading image...")
        image = load_image(args.image_path)
        image = image.to(args.device)
        
        # Get base Q-former features
        print("Generating Q-former features...")
        qformer_features = generate_q_former_features(model, image, device=args.device)
        
        # Enhance features
        print("Enhancing features with diffusion model...")
        enhanced_token_features = torch.zeros_like(qformer_features)
        
        for t in tqdm(range(qformer_features.shape[1])):
            token_feat = qformer_features[:, t:t+1]
            enhanced_token = enhance_features(
                diffusion_model, 
                token_feat, 
                num_steps=args.num_steps,
                device=args.device
            )
            enhanced_token_features[0, t] = enhanced_token[0]
        
        # Save the enhanced features
        output_file = os.path.join(args.output_dir, os.path.basename(args.image_path).split('.')[0] + '_enhanced_diffusion.pt')
        torch.save({
            'original_features': qformer_features.cpu(),
            'enhanced_features': enhanced_token_features.cpu(),
            'image_path': args.image_path
        }, output_file)
        
        print(f"Enhanced features saved to {output_file}")
        
        # Calculate and print improvement statistics
        diff = (enhanced_token_features - qformer_features).abs().mean().item()
        print(f"Average feature change magnitude: {diff:.6f}")
        
    else:  # Process all images in a directory
        print(f"Processing all images in {args.image_dir}...")
        batch_process_images(
            model, 
            diffusion_model, 
            args.image_dir, 
            args.output_dir, 
            batch_size=args.batch_size,
            num_steps=args.num_steps, 
            device=args.device
        )
    
    return enhanced_token_features if args.image_path else None


if __name__ == "__main__":
    main()