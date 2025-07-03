#!/bin/bash

# Test script for the enhanced EVCap evaluation
# This script demonstrates how to use the eval_evcap_enhanced.py

echo "Testing Enhanced EVCap Pipeline"
echo "================================"

# Check if required model files exist
EVCAP_MODEL="checkpoints/000.pt"
FLOW_MODEL="checkpoints/flow_model_000.pt"
TEST_IMAGE="data/example_mg1.jpg"

if [ ! -f "$EVCAP_MODEL" ]; then
    echo "ERROR: EVCap model not found at $EVCAP_MODEL"
    echo "Please ensure you have trained the EVCap model first."
    exit 1
fi

if [ ! -f "$FLOW_MODEL" ]; then
    echo "ERROR: Flow model not found at $FLOW_MODEL"
    echo "Please train the flow model first using:"
    echo "python train_evcap_flowmatching.py --cfg-path train_configs/evcap_train.yaml"
    exit 1
fi

if [ ! -f "$TEST_IMAGE" ]; then
    echo "WARNING: Test image not found at $TEST_IMAGE"
    echo "Using default test image from data directory..."
    TEST_IMAGE="data/example_mg1.jpg"
    if [ ! -f "$TEST_IMAGE" ]; then
        echo "ERROR: No test image available. Please provide a test image."
        exit 1
    fi
fi

echo "All required files found. Running enhanced captioning test..."
echo "Using EVCap model: $EVCAP_MODEL"
echo "Using Flow model: $FLOW_MODEL"
echo "Using test image: $TEST_IMAGE"
echo ""

# Run the enhanced captioning comparison
python eval_evcap_enhanced.py \
    --image_path "$TEST_IMAGE" \
    --evcap_model "$EVCAP_MODEL" \
    --flow_model "$FLOW_MODEL" \
    --device cuda

echo ""
echo "Test completed! Check the output above for caption comparisons."
