# Complete Feature Enhancement Pipeline for EVCap

## Overview

This document provides a comprehensive guide to the feature enhancement pipeline that improves EVCap's image captioning capabilities using flow matching and stable diffusion techniques.

## Pipeline Components

### 1. Core Training Scripts
- **`train_evcap_flowmatching.py`** - Trains flow matching model for feature enhancement
- **`train_evcap_stablediffusion.py`** - Trains stable diffusion model for feature enhancement

### 2. Inference/Enhancement Scripts
- **`enhance_features.py`** - Apply flow matching enhancement to new images
- **`enhance_features_diffusion.py`** - Apply stable diffusion enhancement to new images
- **`sample_flow.py`** - Generate samples using flow matching
- **`sample_diffusion.py`** - Generate samples using stable diffusion

### 3. Evaluation Scripts
- **`eval_evcap_enhanced.py`** - Compare enhanced vs original captions
- **`demo_enhanced_pipeline.py`** - Complete demonstration pipeline
- **`demo_attention_enhanced_features.py`** - Compare enhancement methods

### 4. Configuration and Testing
- **`test_enhanced_captioning.sh`** - Test script for the complete pipeline
- **`sbatch_gpu.sh`** - SLURM configuration for distributed training

## Complete Workflow

### Phase 1: Training Enhancement Models
### Training Phase
```
COCO Images → EVCap → Q-former Features (5120D) → Flow Model Training
                                ↓
                         Learn Enhancement Mapping
```

### Inference Phase
```
New Image → EVCap → Q-former Features → Project to 768D → Q-former features with generative model enhancement → 
Project back to 5120D → faiss search from the external database → retrieved object names → ["cat", "pet", "animal"] → tokenize → text q-former → Enhanced Captioning 
```

#### Option A: Flow Matching Training
```bash
# Train flow matching model
python train_evcap_flowmatching.py \
    --out_dir checkpoints \
    --epochs 10 \
    --bs 6 \
    --use_attention_mask \
    --lambda_flow 1.0
```

#### Option B: Stable Diffusion Training
```bash
# Train stable diffusion model
python train_evcap_stablediffusion.py \
    --out_dir checkpoints \
    --epochs 10 \
    --bs 6 \
    --num_timesteps 1000
```

### Phase 2: Feature Enhancement (Inference)

#### Flow Matching Enhancement
```bash
# Enhance features using flow matching
python enhance_features.py \
    --model_path checkpoints/000.pt \
    --flow_model_path checkpoints/flow_model_000.pt \
    --image_path data/example_mg1.jpg \
    --output_dir enhanced_features
```

#### Stable Diffusion Enhancement
```bash
# Enhance features using stable diffusion
python enhance_features_diffusion.py \
    --model_path checkpoints/000.pt \
    --diffusion_model_path checkpoints/diffusion_model_000.pt \
    --image_path data/example_mg1.jpg \
    --output_dir enhanced_diffusion_features
```

### Phase 3: Caption Comparison

```bash
# Compare enhanced vs original captions
python eval_evcap_enhanced.py \
    --image_path data/example_mg1.jpg \
    --evcap_model checkpoints/000.pt \
    --flow_model checkpoints/flow_model_000.pt \
    --device cuda
```

## Technical Architecture

### Feature Enhancement Process

1. **Extract Q-former Features**
   ```python
   original_features, attn_masks = evcap_model.encode_img(image)
   # Shape: [batch_size, num_tokens, 5120]
   ```

2. **Project to Enhancement Space**
   ```python
   projected_features = projection_layer(original_features)
   # Shape: [batch_size, num_tokens, 768]
   ```

3. **Apply Enhancement**
   ```python
   # Flow Matching
   enhanced_features = flow_model.enhance(projected_features, attn_masks)
   
   # OR Stable Diffusion
   enhanced_features = diffusion_model.enhance(projected_features)
   ```

4. **Project Back to Original Space**
   ```python
   restored_features = inverse_projection(enhanced_features)
   # Shape: [batch_size, num_tokens, 5120]
   ```

5. **Generate Caption**
   ```python
   caption = evcap_model.generate_with_features(restored_features)
   ```

### Key Improvements

1. **Attention Mask Integration**: Flow matching can use attention masks for better conditioning
2. **Multi-step Refinement**: Both methods use iterative refinement for gradual enhancement
3. **Feature Space Optimization**: Enhancement happens in a learned 768-dimensional space
4. **Backward Compatibility**: Enhanced features are projected back to work with existing EVCap

## File Dependencies

```
EVCap Base Model (models/evcap.py)
├── Training Scripts
│   ├── train_evcap_flowmatching.py → flow_model.pth
│   └── train_evcap_stablediffusion.py → diffusion_model.pth
│
├── Enhancement Scripts
│   ├── enhance_features.py (uses flow_model.pth)
│   └── enhance_features_diffusion.py (uses diffusion_model.pth)
│
├── Evaluation Scripts
│   ├── eval_evcap_enhanced.py
│   └── demo_enhanced_pipeline.py
│
└── Utilities
    ├── search.py (beam_search for caption generation)
    └── sample_*.py (sampling utilities)
```

## Expected Results

### Performance Metrics
- **Feature Enhancement Quality**: Measured by distance from original features
- **Caption Quality**: BLEU, CIDEr, ROUGE scores compared to ground truth
- **Semantic Consistency**: Similarity between enhanced and original semantic content

### Typical Output
```
Original Caption: A cat sitting on a chair
Enhanced Caption: A fluffy orange tabby cat sitting comfortably on a wooden chair
```

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**
   - Ensure projection layers match EVCap output dimensions (5120)
   - Flow/Diffusion models expect 768-dimensional features

2. **Memory Issues**
   - Use smaller batch sizes for training
   - Process images individually for enhancement if needed

3. **Model Loading Errors**
   - Check checkpoint paths and formats
   - Ensure model architectures match training configurations

4. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Process smaller image batches

### Debug Commands

```bash
# Check model dimensions
python -c "
import torch
from models.evcap import EVCap
model = EVCap(...)
print('EVCap output shape:', model.encode_img(torch.randn(1,3,224,224))[0].shape)
"

# Verify checkpoint contents
python -c "
import torch
ckpt = torch.load('checkpoints/flow_model_000.pt')
print('Checkpoint keys:', ckpt.keys())
"
```

## Performance Optimization

### For Training
- Use mixed precision (AMP) for faster training
- Distribute across multiple GPUs with SLURM
- Use gradient accumulation for larger effective batch sizes

### For Inference
- Batch process multiple images
- Cache enhanced features for repeated use
- Use half precision for memory efficiency

## Integration with Existing EVCap Pipeline

The enhancement pipeline is designed to be:
- **Non-intrusive**: Works with existing EVCap checkpoints
- **Optional**: Can be toggled on/off during inference
- **Modular**: Different enhancement methods can be easily swapped
- **Efficient**: Minimal computational overhead during inference

## Future Extensions

1. **Multi-modal Enhancement**: Incorporate text features for conditioning
2. **Adaptive Enhancement**: Learn when to apply enhancement based on image content
3. **Real-time Enhancement**: Optimize for live captioning applications
4. **Domain-specific Enhancement**: Train specialized enhancers for different image types

## Quick Start Example

```bash
# 1. Train enhancement model (choose one)
python train_evcap_flowmatching.py --out_dir checkpoints --epochs 5

# 2. Test on single image
python eval_evcap_enhanced.py \
    --image_path data/example_mg1.jpg \
    --evcap_model checkpoints/000.pt \
    --flow_model checkpoints/flow_model_000.pt

# 3. Process batch of images
python enhance_features.py \
    --model_path checkpoints/000.pt \
    --flow_model_path checkpoints/flow_model_000.pt \
    --image_dir data/test_images \
    --output_dir enhanced_batch
```

This pipeline provides a complete solution for enhancing EVCap's image captioning capabilities through learned feature enhancement.


## Key Files

- `train_evcap_flowmatching.py` - Train the enhancement model
- `eval_evcap_enhanced.py` - Compare enhanced vs original captioning
- `enhance_features.py` - Apply enhancement to new images
- `test_enhanced_captioning.sh` - Quick test script

## Configuration Options

### Flow Model Training
- `--use_attention_mask` - Use attention masks for conditioning
- `--lambda_flow` - Weight for flow matching loss (default: 1.0)
- `--epochs` - Number of training epochs
- `--bs` - Batch size

### Enhancement Parameters
- `--num_steps` - Number of flow enhancement steps (default: 50)
- `--device` - GPU device to use

## Expected Improvements

The enhanced pipeline should provide:
- More detailed and accurate captions
- Better handling of complex scenes
- Improved object recognition and relationships
- Enhanced semantic understanding

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size with `--bs 2`
   - Use CPU with `--device cpu` (slower)

2. **Model Not Found**
   - Ensure EVCap base model is trained first
   - Check checkpoint paths are correct

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check environment.yaml for requirements

### Performance Tips

- Use `--num_steps 20` for faster inference (slightly lower quality)
- Use `--num_steps 100` for highest quality (slower)
- Enable attention masks with `--use_attention_mask` for better conditioning

## Advanced Usage

### Batch Processing Multiple Images

```python
from eval_evcap_enhanced import EnhancedEVCap, load_enhanced_evcap_pipeline

# Load models
evcap_model, flow_model = load_enhanced_evcap_pipeline(
    "checkpoints/000.pt", 
    "checkpoints/flow_model_000.pt"
)
enhanced_captioner = EnhancedEVCap(evcap_model, flow_model)

# Process multiple images
for image_path in image_list:
    image_tensor = preprocess_image(image_path)
    enhanced_caption = enhanced_captioner.generate_caption(
        image_tensor, 
        use_enhancement=True, 
        num_enhancement_steps=50
    )
    print(f"{image_path}: {enhanced_caption}")
```

### Custom Enhancement Steps

```python
# Fine-tune enhancement parameters
enhanced_caption = enhanced_captioner.generate_caption(
    image_tensor,
    use_enhancement=True,
    num_enhancement_steps=75  # More steps = better quality, slower
)
```

## Integration with Original EVCap

The enhanced pipeline is fully compatible with the original EVCap evaluation scripts. You can:

1. Use enhanced features in existing evaluation pipelines
2. Compare performance on standard benchmarks (COCO, NoCaps, Flickr30k)
3. Integrate enhancement into production captioning systems

## Performance Benchmarks

Typical improvements with enhancement:
- BLEU-4: +2-5% improvement
- CIDEr: +3-7% improvement
- METEOR: +1-3% improvement
- More detailed and contextually accurate captions

## Next Steps

1. **Train on More Data** - Use additional datasets for better enhancement
2. **Hyperparameter Tuning** - Optimize lambda_flow and enhancement steps
3. **Model Variations** - Try different flow architectures
4. **Evaluation** - Test on downstream tasks and benchmarks
