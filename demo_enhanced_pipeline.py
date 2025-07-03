#!/usr/bin/env python3
"""
Demo script showing the complete enhanced EVCap pipeline.
This script demonstrates:
1. Loading pre-trained models
2. Processing an image
3. Comparing original vs enhanced captions
4. Showing the feature enhancement process
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Demo Enhanced EVCap Pipeline")
    parser.add_argument("--image_path", type=str, default="data/example_mg1.jpg", 
                       help="Path to test image")
    parser.add_argument("--evcap_model", type=str, default="checkpoints/000.pt",
                       help="Path to EVCap model")
    parser.add_argument("--flow_model", type=str, default="checkpoints/flow_model_000.pt",
                       help="Path to flow model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Enhanced EVCap Pipeline Demo")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found at {args.image_path}")
        return
        
    if not os.path.exists(args.evcap_model):
        print(f"❌ Error: EVCap model not found at {args.evcap_model}")
        print("Please train the EVCap model first using train_evcap.py")
        return
        
    if not os.path.exists(args.flow_model):
        print(f"❌ Error: Flow model not found at {args.flow_model}")
        print("Please train the flow model first using train_evcap_flowmatching.py")
        return
    
    print(f"✅ Using image: {args.image_path}")
    print(f"✅ Using EVCap model: {args.evcap_model}")
    print(f"✅ Using Flow model: {args.flow_model}")
    print(f"✅ Using device: {args.device}")
    print()
    
    try:
        # Import and load models
        print("🔄 Loading models...")
        from eval_evcap_enhanced import load_enhanced_evcap_pipeline, EnhancedEVCap
        
        evcap_model, flow_model = load_enhanced_evcap_pipeline(
            args.evcap_model, args.flow_model, args.device
        )
        
        enhanced_captioner = EnhancedEVCap(evcap_model, flow_model, args.device)
        print("✅ Models loaded successfully!")
        print()
        
        # Load and preprocess image
        print("🔄 Processing image...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711))
        ])
        
        image = Image.open(args.image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(args.device)
        print("✅ Image processed!")
        print()
        
        # Generate captions
        print("🔄 Generating captions...")
        print("   📝 Original caption (without enhancement)...")
        original_caption = enhanced_captioner.generate_caption(
            image_tensor, use_enhancement=False
        )
        
        print("   🎯 Enhanced caption (with flow matching)...")
        enhanced_caption = enhanced_captioner.generate_caption(
            image_tensor, use_enhancement=True, num_enhancement_steps=50
        )
        print("✅ Captions generated!")
        print()
        
        # Display results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"📷 Image: {args.image_path}")
        print()
        print("📝 Original Caption:")
        print(f"   {original_caption}")
        print()
        print("🎯 Enhanced Caption:")
        print(f"   {enhanced_caption}")
        print()
        
        # Show feature analysis
        print("=" * 60)
        print("FEATURE ANALYSIS")
        print("=" * 60)
        
        with torch.no_grad():
            # Extract original features
            original_features, attn_masks = evcap_model.encode_img(image_tensor)
            print(f"📊 Original features shape: {original_features.shape}")
            print(f"📊 Attention masks shape: {attn_masks.shape}")
            
            # Project and enhance
            projected_features = enhanced_captioner.feature_projection(original_features)
            print(f"📊 Projected features shape: {projected_features.shape}")
            
            from enhance_features import enhance_features
            enhanced_features = enhance_features(
                flow_model, projected_features, attn_masks, num_steps=50, device=args.device
            )
            print(f"📊 Enhanced features shape: {enhanced_features.shape}")
            
            # Compute feature differences
            feature_diff = torch.norm(enhanced_features - projected_features, dim=-1).mean()
            print(f"📊 Average feature change magnitude: {feature_diff:.4f}")
            
        print()
        print("=" * 60)
        print("Demo completed successfully! 🎉")
        print("=" * 60)
        
        # Save results
        results_file = f"demo_results_{os.path.basename(args.image_path)}.txt"
        with open(results_file, 'w') as f:
            f.write("Enhanced EVCap Demo Results\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Original Caption: {original_caption}\n")
            f.write(f"Enhanced Caption: {enhanced_caption}\n")
            f.write(f"Feature Change: {feature_diff:.4f}\n")
        
        print(f"📄 Results saved to: {results_file}")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all required modules are available.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your model files and image path.")

if __name__ == "__main__":
    main()
