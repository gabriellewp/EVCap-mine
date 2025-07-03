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


def enhance_features(flow_model, base_features, attn_mask=None, num_steps=100, device='cuda'):
    """
    Enhance features using the flow model's step function.
    This treats the base features as conditioning and generates improved features via flow.
    
    Parameters:
    - flow_model: The trained conditional flow model
    - base_features: Original Q-former features to enhance
    - attn_mask: Optional attention mask for conditioning
    - num_steps: Number of refinement steps
    - device: Device to run on
    """
    # Ensure base_features is properly sized and on the right device
    base_features = base_features.to(device)
    
    # Initialize with a small noise perturbation of the original features
    # Starting close to the original features and refining them
    enhanced_features = base_features + torch.randn_like(base_features) * 0.1
    
    # Time steps for the ODE solver (from t=0.5 to t=0)
    # Starting from t=0.5 means we're halfway between noise and signal
    # This gives the flow model room to refine but keeps close to original
    time_steps = torch.linspace(0.5, 0.0, num_steps + 1).to(device)
    
    # Euler steps for ODE solving
    with torch.no_grad():
        for i in range(num_steps):
            t_start = time_steps[i]
            t_end = time_steps[i + 1]
            
            # Use the step function to update the features
            # Include attention mask if the model supports it and mask is provided
            if hasattr(flow_model, 'use_attention_mask') and flow_model.use_attention_mask and attn_mask is not None:
                enhanced_features = flow_model.step(
                    x_t=enhanced_features,
                    t_start=t_start,
                    t_end=t_end,
                    cond=base_features,
                    attn_mask=attn_mask
                )
            else:
                enhanced_features = flow_model.step(
                    x_t=enhanced_features,
                    t_start=t_start,
                    t_end=t_end,
                    cond=base_features
                )
            
    return enhanced_features


def generate_q_former_features(model, image, device='cuda'):
    """Extract Q-former features and attention masks from an image."""
    with torch.no_grad():
        image = image.to(device)
        cond_features, attn_masks = model.encode_img(image)
        return cond_features, attn_masks
    

def batch_process_images(model, flow_model, image_dir, output_dir, batch_size=8, num_steps=50, device='cuda'):
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
            
            # Get base Q-former features and attention masks
            qformer_features, qformer_attn_masks = generate_q_former_features(model, batch_tensor, device)
            
            # Get enhanced features for each token
            enhanced_token_features = torch.zeros_like(qformer_features)
            
            # Process each example in the batch
            for b in range(qformer_features.shape[0]):
                # Process each token position separately
                for t in range(qformer_features.shape[1]):
                    # Take single token features and enhance them
                    token_feat = qformer_features[b, t:t+1]
                    enhanced_token_feat = enhance_features(
                        flow_model, 
                        token_feat, 
                        num_steps=num_steps,
                        device=device
                    )
                    enhanced_token_features[b, t] = enhanced_token_feat[0]
            
            # Save enhanced features for this batch
            for j, img_file in enumerate(batch_files):
                output_file = os.path.join(output_dir, os.path.splitext(img_file)[0] + '_enhanced.pt')
                torch.save({
                    'original_features': qformer_features[j],
                    'enhanced_features': enhanced_token_features[j]
                }, output_file)
    
    print(f"Processed {len(image_files)} images. Enhanced features saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhance Q-former features using flow model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained EVCap model checkpoint")
    parser.add_argument("--flow_model_path", type=str, required=True, help="Path to the trained flow model checkpoint")
    parser.add_argument("--image_path", type=str, help="Path to a single input image")
    parser.add_argument("--image_dir", type=str, help="Directory containing multiple images to process")
    parser.add_argument("--output_dir", type=str, default="enhanced_features", help="Directory to save enhanced features")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of refinement steps")
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
    model, flow_model = load_models(args.model_path, args.flow_model_path, device=args.device)
    
    if args.image_path:  # Process a single image
        # Load and process the image
        print("Loading image...")
        image = load_image(args.image_path)
        image = image.to(args.device)
        
        # Get base Q-former features and attention masks
        print("Generating Q-former features...")
        qformer_features, qformer_attn_masks = generate_q_former_features(model, image, device=args.device)
        
        # Enhance features
        print("Enhancing features...")
        enhanced_token_features = torch.zeros_like(qformer_features)
        
        for t in tqdm(range(qformer_features.shape[1])):
            token_feat = qformer_features[:, t:t+1]
            enhanced_token = enhance_features(
                flow_model, 
                token_feat, 
                num_steps=args.num_steps,
                device=args.device
            )
            enhanced_token_features[0, t] = enhanced_token[0]
        
        # Save the enhanced features
        output_file = os.path.join(args.output_dir, os.path.basename(args.image_path).split('.')[0] + '_enhanced.pt')
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
            flow_model, 
            args.image_dir, 
            args.output_dir, 
            batch_size=args.batch_size,
            num_steps=args.num_steps, 
            device=args.device
        )
    
    return enhanced_token_features if args.image_path else None


if __name__ == "__main__":
    main()
