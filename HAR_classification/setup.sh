#!/bin/bash
# SETUP AND INSTALLATION SCRIPT

echo "=================================="
echo "HAR CLASSIFICATION SETUP"
echo "=================================="

# 1. Create directory structure
echo ""
echo "[1/5] Creating directory structure..."
mkdir -p checkpoints
mkdir -p models
mkdir -p data

# 2. Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install torch torchvision tqdm numpy matplotlib seaborn scikit-learn

# Optional: Install TensorBoard
read -p "Install TensorBoard for visualization? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install tensorboard
    echo "✓ TensorBoard installed"
else
    echo "⊘ Skipping TensorBoard"
fi

# 3. Verify PyTorch installation
echo ""
echo "[3/5] Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ CUDA not available - training will use CPU')
"

# 4. Check file structure
echo ""
echo "[4/5] Checking files..."
required_files=(
    "main.py"
    "train_optimized.py"
    "profiler.py"
    "visualize.py"
    "examples.py"
    "README.md"
    "QUICK_REFERENCE.md"
)

all_present=true
for file in "${required_files[@]}"
do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
        all_present=false
    fi
done

if [ "$all_present" = true ]; then
    echo ""
    echo "✓ All required files present"
else
    echo ""
    echo "⚠ Some files are missing. Please ensure all files are in the current directory."
    exit 1
fi

# 5. Run quick test (optional)
echo ""
echo "[5/5] Setup complete!"
echo ""
read -p "Run quick test (5 epochs, ~10 min)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "Starting quick test..."
    echo "=================================="
    python main.py --epochs 5 --batch-size 32 --save-dir checkpoints/test_run
else
    echo ""
    echo "Setup complete! Ready to train."
    echo ""
    echo "To start training:"
    echo "  python main.py"
    echo ""
    echo "For help:"
    echo "  python main.py --help"
    echo ""
    echo "For examples:"
    echo "  python examples.py"
    echo ""
    echo "See README.md for full documentation"
fi

echo ""
echo "=================================="
echo "SETUP COMPLETE"
echo "=================================="
