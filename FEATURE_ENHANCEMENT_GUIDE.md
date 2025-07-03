# EVCap Feature Enhancement with Flow Matching and Stable Diffusion

This guide explains how to use the enhanced EVCap system that incorporates generative models (Flow Matching and Stable Diffusion) to improve Q-former features for better image captioning.

## Overview

The system provides multiple approaches for enhancing Q-former features:

1. **Flow Matching Enhancement** (`train_evcap_flowmatching.py`)
2. **Stable Diffusion Enhancement** (`train_evcap_stablediffusion.py`)
3. **Attention-Aware Enhancement** (with `--use_attention_mask` flag)

## Training Process

### 1. Flow Matching Training

Train EVCap with flow matching feature enhancement:

```bash
# Submit distributed training job
sbatch sbatch_gpu.sh

# Or run directly for debugging/single GPU
python train_evcap_flowmatching.py \
    --epochs 10 \
    --bs 6 \
    --use_attention_mask \
    --lambda_flow 0.5 \
    --out_dir ./checkpoints_flow
```

### 2. Stable Diffusion Training

Train EVCap with stable diffusion feature enhancement:

```bash
python train_evcap_stablediffusion.py \
    --epochs 10 \
    --bs 6 \
    --lambda_diffusion 0.5 \
    --n_timesteps 1000 \
    --out_dir ./checkpoints_diffusion
```

### Key Training Parameters

- `--use_attention_mask`: Enable attention-aware feature enhancement
- `--lambda_flow`/`--lambda_diffusion`: Weight for generative model loss (0.1-1.0)
- `--epochs`: Number of training epochs
- `--bs`: Batch size (adjust based on GPU memory)
- `--out_dir`: Output directory for checkpoints

## Feature Enhancement

### Flow Matching Enhancement

```bash
# Single image enhancement
python enhance_features.py \
    --model_path ./checkpoints_flow/009.pt \
    --flow_model_path ./checkpoints_flow/flow_model_009.pt \
    --image_path ./data/example_mg1.jpg \
    --output_dir ./enhanced_features \
    --num_steps 50

# Batch processing
python enhance_features.py \
    --model_path ./checkpoints_flow/009.pt \
    --flow_model_path ./checkpoints_flow/flow_model_009.pt \
    --image_dir ./data/coco/coco2014/val2014 \
    --output_dir ./enhanced_features \
    --batch_size 8 \
    --num_steps 50
```

### Stable Diffusion Enhancement

```bash
# Single image enhancement
python enhance_features_diffusion.py \
    --model_path ./checkpoints_diffusion/009.pt \
    --diffusion_model_path ./checkpoints_diffusion/diffusion_model_009.pt \
    --image_path ./data/example_mg1.jpg \
    --output_dir ./enhanced_diffusion_features \
    --num_steps 50

# Batch processing
python enhance_features_diffusion.py \
    --model_path ./checkpoints_diffusion/009.pt \
    --diffusion_model_path ./checkpoints_diffusion/diffusion_model_009.pt \
    --image_dir ./data/coco/coco2014/val2014 \
    --output_dir ./enhanced_diffusion_features \
    --batch_size 8 \
    --num_steps 50
```

## Sampling and Analysis

### Flow Matching Sampling

Generate samples from the flow model for analysis:

```bash
python sample_flow.py \
    --model_path ./checkpoints_flow/009.pt \
    --flow_model_path ./checkpoints_flow/flow_model_009.pt \
    --image_path ./data/example_mg1.jpg \
    --output_dir ./flow_samples \
    --num_samples 10 \
    --num_steps 100
```

### Stable Diffusion Sampling

Generate samples from the diffusion model:

```bash
python sample_diffusion.py \
    --model_path ./checkpoints_diffusion/009.pt \
    --diffusion_model_path ./checkpoints_diffusion/diffusion_model_009.pt \
    --image_path ./data/example_mg1.jpg \
    --output_dir ./diffusion_samples \
    --num_samples 10 \
    --num_steps 100
```

### Attention-Enhanced Features Demo

Compare attention-enhanced vs baseline feature enhancement:

```bash
python demo_attention_enhanced_features.py \
    --model_path ./checkpoints_flow/009.pt \
    --flow_model_path ./checkpoints_flow/flow_model_009.pt \
    --image_path ./data/example_mg1.jpg \
    --output_dir ./demo_results \
    --use_attention_mask
```

## Key Components

### 1. ConditionalFlow Model

The flow model learns to enhance Q-former features by:
- Taking noisy features and predicting the velocity field
- Using original features as conditioning
- Optionally incorporating attention masks for structure-aware enhancement

### 2. UNet Diffusion Model

The diffusion model enhances features by:
- Learning to denoise corrupted features
- Using Q-former features as conditioning
- Supporting both DDPM and DDIM sampling

### 3. Attention-Aware Processing

When `--use_attention_mask` is enabled:
- Features are weighted by attention scores during mean pooling
- Attention summaries are used as additional conditioning
- This preserves important structural information from the Q-former

## Output Files

### Training Outputs

- `{epoch:03d}.pt`: Combined EVCap + generative model checkpoint
- `flow_model_{epoch:03d}.pt` or `diffusion_model_{epoch:03d}.pt`: Standalone generative model
- Training logs with loss metrics

### Enhancement Outputs

- `*_enhanced.pt` (flow) or `*_enhanced_diffusion.pt` (diffusion): Enhanced feature files
- Each file contains:
  - `original_features`: Q-former features before enhancement
  - `enhanced_features`: Features after generative model refinement
  - `image_path`: Source image path

### Analysis Outputs

- `sample_statistics.pt`: Statistics comparing generated samples to originals
- `enhancement_comparison.pt`: Comparison between different enhancement methods

## Best Practices

### Training Tips

1. **Start with small lambda values** (0.1-0.5) to avoid overwhelming the captioning loss
2. **Use attention masks** for better structural awareness
3. **Monitor both losses** - EVCap loss and generative model loss should decrease together
4. **Adjust batch size** based on GPU memory (typically 4-8 for multi-GPU setups)

### Enhancement Tips

1. **Use 50-100 steps** for good quality vs speed tradeoff
2. **Process in batches** for efficiency on large datasets
3. **Compare methods** using the demo script to understand differences
4. **Save intermediate results** for analysis and debugging

### Evaluation

To evaluate the impact of enhanced features on captioning:

1. Generate enhanced features for your test set
2. Run inference using the enhanced features
3. Compare BLEU, METEOR, CIDEr scores with baseline
4. Analyze attention patterns and feature quality

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure projection layers match your model dimensions
2. **Memory Issues**: Reduce batch size or number of enhancement steps
3. **Diverging Loss**: Lower lambda values or learning rates
4. **Import Errors**: Static analysis warnings don't affect runtime functionality

### Performance Optimization

1. **Use mixed precision** (`--amp`) for faster training
2. **Adjust projection layer sizes** based on your Q-former configuration
3. **Use distributed training** for multi-GPU setups
4. **Cache enhanced features** to avoid re-computation

## Directory Structure

```
EVCap/
├── train_evcap_flowmatching.py      # Flow matching training
├── train_evcap_stablediffusion.py   # Stable diffusion training
├── enhance_features.py              # Flow-based enhancement
├── enhance_features_diffusion.py    # Diffusion-based enhancement
├── sample_flow.py                   # Flow sampling
├── sample_diffusion.py              # Diffusion sampling
├── demo_attention_enhanced_features.py  # Comparison demo
├── sbatch_gpu.sh                    # SLURM distributed training
├── checkpoints/                     # Model checkpoints
├── enhanced_features/               # Enhanced feature outputs
└── output/                          # Training logs
```

This system provides a comprehensive framework for enhancing EVCap features using state-of-the-art generative models, with both flow matching and stable diffusion approaches available for experimentation and comparison.
