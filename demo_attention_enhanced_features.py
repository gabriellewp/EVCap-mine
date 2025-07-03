"""
Demo script showing how to use attention masks for enhanced feature generation.
This demonstrates both approaches: with and without attention mask conditioning.
"""

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


def load_models(model_path, flow_model_path, use_attention_mask=False, device='cuda'):
    """Load the EVCap model and flow model with attention mask support."""
    # Load EVCap model
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
    
    # Load flow model with attention mask support
    flow_model = ConditionalFlow(dim=768, cond_dim=768, use_attention_mask=use_attention_mask)
    flow_model_state = torch.load(flow_model_path, map_location=device)
    if 'flow_model' in flow_model_state:
        flow_model.load_state_dict(flow_model_state['flow_model'])
    else:
        flow_model.load_state_dict(flow_model_state)
    flow_model = flow_model.to(device)
    flow_model.eval()
        
    return model, flow_model


def enhance_features_with_attention(flow_model, base_features, attn_mask, num_steps=50, device='cuda'):
    """Enhanced feature generation using attention-weighted processing."""
    # Project features to flow model dimension
    projection = torch.nn.Linear(base_features.shape[-1], 768).to(device)
    
    # Apply attention-weighted mean pooling
    attn_weights = attn_mask.float()
    attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Compute weighted features
    weighted_features = (base_features * attn_weights.unsqueeze(-1)).sum(dim=1)
    projected_features = projection(weighted_features)
    
    # Create attention summary for conditioning
    attn_summary = attn_weights.mean(dim=1, keepdim=True)
    
    # Initialize with small perturbation
    enhanced_features = projected_features + torch.randn_like(projected_features) * 0.1
    
    # Flow refinement steps
    time_steps = torch.linspace(0.5, 0.0, num_steps + 1).to(device)
    
    with torch.no_grad():
        for i in range(num_steps):
            t_start = time_steps[i]
            t_end = time_steps[i + 1]
            
            # Use flow model with attention conditioning
            enhanced_features = flow_model.step(
                x_t=enhanced_features,
                t_start=t_start,
                t_end=t_end,
                cond=projected_features,
                attn_mask=attn_summary
            )
    
    return enhanced_features, projected_features


def enhance_features_baseline(flow_model, base_features, num_steps=50, device='cuda'):
    """Baseline feature enhancement without attention conditioning."""
    # Project features to flow model dimension
    projection = torch.nn.Linear(base_features.shape[-1], 768).to(device)
    
    # Simple mean pooling
    mean_features = base_features.mean(dim=1)
    projected_features = projection(mean_features)
    
    # Initialize with small perturbation
    enhanced_features = projected_features + torch.randn_like(projected_features) * 0.1
    
    # Flow refinement steps
    time_steps = torch.linspace(0.5, 0.0, num_steps + 1).to(device)
    
    with torch.no_grad():
        for i in range(num_steps):
            t_start = time_steps[i]
            t_end = time_steps[i + 1]
            
            # Use flow model without attention conditioning
            enhanced_features = flow_model.step(
                x_t=enhanced_features,
                t_start=t_start,
                t_end=t_end,
                cond=projected_features
            )
    
    return enhanced_features, projected_features


def compare_enhancement_methods(model, flow_model, image, device='cuda'):
    """Compare attention-enhanced vs baseline feature enhancement."""
    
    # Extract Q-former features and attention masks
    with torch.no_grad():
        qformer_features, attn_masks = model.encode_img(image)
    
    print(f"Q-former features shape: {qformer_features.shape}")
    print(f"Attention masks shape: {attn_masks.shape}")
    
    # Method 1: Attention-enhanced features (if flow model supports it)
    if hasattr(flow_model, 'use_attention_mask') and flow_model.use_attention_mask:
        enhanced_attn, original_attn = enhance_features_with_attention(
            flow_model, qformer_features, attn_masks, device=device
        )
        print("✓ Generated attention-enhanced features")
    else:
        enhanced_attn, original_attn = None, None
        print("✗ Flow model doesn't support attention masks")
    
    # Method 2: Baseline features (simple mean pooling)
    enhanced_baseline, original_baseline = enhance_features_baseline(
        flow_model, qformer_features, device=device
    )
    print("✓ Generated baseline enhanced features")
    
    # Calculate differences
    if enhanced_attn is not None:
        attn_diff = torch.norm(enhanced_attn - original_attn, dim=-1).mean().item()
        baseline_diff = torch.norm(enhanced_baseline - original_baseline, dim=-1).mean().item()
        
        print(f"\nFeature Enhancement Results:")
        print(f"Attention-enhanced change magnitude: {attn_diff:.6f}")
        print(f"Baseline change magnitude: {baseline_diff:.6f}")
        
        # Compare the two enhanced versions
        method_diff = torch.norm(enhanced_attn - enhanced_baseline, dim=-1).mean().item()
        print(f"Difference between methods: {method_diff:.6f}")
        
        return {
            'enhanced_attention': enhanced_attn,
            'enhanced_baseline': enhanced_baseline,
            'original_attention': original_attn,
            'original_baseline': original_baseline,
            'attention_change': attn_diff,
            'baseline_change': baseline_diff,
            'method_difference': method_diff
        }
    else:
        baseline_diff = torch.norm(enhanced_baseline - original_baseline, dim=-1).mean().item()
        print(f"\nBaseline Enhancement Results:")
        print(f"Change magnitude: {baseline_diff:.6f}")
        
        return {
            'enhanced_baseline': enhanced_baseline,
            'original_baseline': original_baseline,
            'baseline_change': baseline_diff
        }


def main():
    parser = argparse.ArgumentParser(description="Demo attention-enhanced feature generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to EVCap model checkpoint")
    parser.add_argument("--flow_model_path", type=str, required=True, help="Path to flow model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="demo_results", help="Output directory")
    parser.add_argument("--use_attention_mask", action='store_true', help="Use attention mask conditioning")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    model, flow_model = load_models(
        args.model_path, 
        args.flow_model_path, 
        use_attention_mask=args.use_attention_mask,
        device=args.device
    )
    
    # Load image
    print(f"Loading image: {args.image_path}")
    image = load_image(args.image_path).to(args.device)
    
    # Compare enhancement methods
    print("\nComparing feature enhancement methods...")
    results = compare_enhancement_methods(model, flow_model, image, device=args.device)
    
    # Save results
    output_file = os.path.join(args.output_dir, "enhancement_comparison.pt")
    torch.save({
        'results': {k: v.cpu() if torch.is_tensor(v) else v for k, v in results.items()},
        'image_path': args.image_path,
        'use_attention_mask': args.use_attention_mask
    }, output_file)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    if 'method_difference' in results:
        print(f"Attention mask conditioning {'ENABLED' if args.use_attention_mask else 'DISABLED'}")
        print(f"Both methods successfully generated enhanced features")
        print(f"Difference between attention and baseline methods: {results['method_difference']:.6f}")
    else:
        print("Only baseline method available (attention mask support not detected)")


if __name__ == "__main__":
    main()
