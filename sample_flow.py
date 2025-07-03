import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from models.evcap import EVCap
from train_evcap_flowmatching import ConditionalFlow


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


def load_models(model_path, flow_model_path=None, device='cuda'):
    """Load the EVCap model and flow model."""
    # Load EVCap model configuration from your training script
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
    
    # Initialize and load flow model if path is provided
    flow_model = None
    if flow_model_path:
        flow_model = ConditionalFlow(dim=768, cond_dim=768)
        flow_model_state = torch.load(flow_model_path, map_location=device)
        if 'flow_model' in flow_model_state:
            flow_model.load_state_dict(flow_model_state['flow_model'])
        else:
            flow_model.load_state_dict(flow_model_state)
        flow_model = flow_model.to(device)
        flow_model.eval()
        
    return model, flow_model


def sample_from_flow_model(flow_model, cond_vector, num_steps=100, sample_batch_size=1, device='cuda'):
    """Generate samples using the flow model's step function."""
    # Create a projection layer to match dimensions if needed
    proj = torch.nn.Linear(cond_vector.shape[-1], 768).to(device)
    z_cond = proj(cond_vector) if cond_vector.shape[-1] != 768 else cond_vector
    
    # Initialize from random noise
    z_sample = torch.randn(sample_batch_size, 768).to(device)
    
    # Time steps for the ODE solver (from t=1 to t=0)
    time_steps = torch.linspace(1.0, 0.0, num_steps + 1).to(device)
    
    # Euler steps for ODE solving
    with torch.no_grad():
        for i in range(num_steps):
            t_start = time_steps[i]
            t_end = time_steps[i + 1]
            
            # Use the step function to update the sample
            z_sample = flow_model.step(
                x_t=z_sample,
                t_start=t_start,
                t_end=t_end,
                cond=z_cond
            )
            
    return z_sample


def generate_features(model, image, device='cuda'):
    """Extract Q-former features from an image."""
    with torch.no_grad():
        image = image.to(device)
        cond_features, _ = model.encode_img(image)
        # Mean pooling across token dimension
        cond_vector = cond_features.mean(dim=1)
        
    return cond_vector


def main():
    parser = argparse.ArgumentParser(description="Sample from trained flow matching model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained EVCap model checkpoint")
    parser.add_argument("--flow_model_path", type=str, required=True, help="Path to the trained flow model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, default="flow_samples", help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of sampling steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    model, flow_model = load_models(args.model_path, args.flow_model_path, device=args.device)
    
    # Load and process the image
    print("Loading image...")
    image = load_image(args.image_path)
    
    # Generate conditional features
    print("Generating conditional features...")
    cond_vector = generate_features(model, image, device=args.device)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = []
    for i in tqdm(range(args.num_samples)):
        sample = sample_from_flow_model(
            flow_model, 
            cond_vector, 
            num_steps=args.num_steps,
            sample_batch_size=1,
            device=args.device
        )
        samples.append(sample)
    
    # Save the samples
    samples_tensor = torch.cat(samples, dim=0)
    output_path = os.path.join(args.output_dir, "flow_samples.pt")
    torch.save(samples_tensor, output_path)
    print(f"Saved {args.num_samples} samples to {output_path}")
    
    # You could add feature visualization or additional processing here
    
    return samples_tensor


if __name__ == "__main__":
    main()
