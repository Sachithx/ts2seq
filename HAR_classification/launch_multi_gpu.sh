#!/bin/bash
# MULTI-GPU TRAINING LAUNCHER
# Convenient script to launch multi-GPU training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}MULTI-GPU TRAINING LAUNCHER${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. No NVIDIA GPUs available?${NC}"
    exit 1
fi

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}Available GPUs: $NUM_GPUS${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | nl -v 0

# Parse arguments
MODE=""
GPUS=""
BATCH_SIZE=32
EPOCHS=20
LR=""
OTHER_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            if [[ "$2" == "ddp" ]] || [[ "$2" == "dp" ]]; then
                MODE="$2"
                shift 2
            else
                echo -e "${RED}Error: --mode must be 'ddp' or 'dp'${NC}"
                exit 1
            fi
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        *)
            OTHER_ARGS="$OTHER_ARGS $1"
            shift
            ;;
    esac
done

# Interactive mode if no mode specified
if [ -z "$MODE" ]; then
    echo ""
    echo "Select training mode:"
    echo "  1) DistributedDataParallel (DDP) - Recommended for best performance"
    echo "  2) DataParallel (DP) - Simpler, good for quick tests"
    echo "  3) Single GPU - No multi-GPU"
    echo ""
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        1)
            MODE="ddp"
            ;;
        2)
            MODE="dp"
            ;;
        3)
            MODE="single"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
fi

# Ask for number of GPUs if not specified
if [ -z "$GPUS" ] && [ "$MODE" != "single" ]; then
    echo ""
    read -p "How many GPUs to use? [1-$NUM_GPUS]: " GPUS
    
    # Validate input
    if ! [[ "$GPUS" =~ ^[0-9]+$ ]] || [ "$GPUS" -lt 1 ] || [ "$GPUS" -gt "$NUM_GPUS" ]; then
        echo -e "${RED}Invalid number of GPUs${NC}"
        exit 1
    fi
fi

# Auto-scale learning rate if not specified
if [ -z "$LR" ]; then
    if [ "$MODE" == "ddp" ] || [ "$MODE" == "dp" ]; then
        # Scale LR linearly with number of GPUs
        BASE_LR="0.001"
        LR=$(python -c "print($BASE_LR * $GPUS)")
        echo -e "${YELLOW}Auto-scaling learning rate: $BASE_LR × $GPUS = $LR${NC}"
    else
        LR="0.001"
    fi
fi

# Calculate effective batch size
if [ "$MODE" == "single" ]; then
    EFFECTIVE_BATCH=$BATCH_SIZE
else
    EFFECTIVE_BATCH=$((BATCH_SIZE * GPUS))
fi

# Print configuration
echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Training Configuration${NC}"
echo -e "${BLUE}================================${NC}"
echo "Mode: $MODE"
if [ "$MODE" != "single" ]; then
    echo "GPUs: $GPUS"
fi
echo "Batch size per GPU: $BATCH_SIZE"
echo "Effective batch size: $EFFECTIVE_BATCH"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Additional args: $OTHER_ARGS"
echo -e "${BLUE}================================${NC}"
echo ""

# Confirm
read -p "Continue? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo ""

# Launch training
if [ "$MODE" == "ddp" ]; then
    # DistributedDataParallel
    CMD="torchrun --nproc_per_node=$GPUS main_multi_gpu.py \
        --multi-gpu ddp \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        $OTHER_ARGS"
    
    echo "Command: $CMD"
    echo ""
    eval $CMD
    
elif [ "$MODE" == "dp" ]; then
    # DataParallel
    GPU_IDS=$(seq -s, 0 $((GPUS-1)))
    
    CMD="python main_multi_gpu.py \
        --multi-gpu dp \
        --gpu-ids $GPU_IDS \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        $OTHER_ARGS"
    
    echo "Command: $CMD"
    echo ""
    eval $CMD
    
else
    # Single GPU
    CMD="python main.py \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        $OTHER_ARGS"
    
    echo "Command: $CMD"
    echo ""
    eval $CMD
fi

# Done
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}================================${NC}"
