#!/bin/bash
# setup_node001.sh - Setup script for RectifiedHR on node001

echo "🚀 Setting up RectifiedHR environment on node001..."

# Load required modules
echo "📦 Loading modules..."
module load gcc11/11.3.0
module load cuda11.8/toolkit/11.8.0  
module load cudnn8.1-cuda11.2/8.1.1.33
module load python/3.8.9

# Verify CUDA availability
echo "🔍 Verifying CUDA setup..."
nvidia-smi
echo "CUDA Version: $(nvcc --version)"

# Create/activate virtual environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
echo "⚡ Installing PyTorch with CUDA 11.8..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install diffusion models
echo "🎨 Installing diffusion libraries..."
pip install diffusers==0.21.4
pip install transformers==4.35.0  
pip install accelerate==0.24.0

# Install evaluation metrics
echo "📊 Installing evaluation metrics..."
pip install open_clip_torch>=2.20.0
pip install lpips>=0.1.4

# Install scientific computing
echo "🔬 Installing scientific computing libraries..."
pip install numpy>=1.24.0 scipy>=1.11.0 scikit-image>=0.21.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0

# Install utilities
echo "🛠️ Installing utilities..."
pip install tqdm>=4.66.0 wandb>=0.15.0 tensorboard>=2.14.0

# Test PyTorch CUDA
echo "🧪 Testing PyTorch CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ CUDA not available!')
"

# Test diffusers
echo "🎭 Testing diffusers..."
python -c "
try:
    from diffusers import StableDiffusionPipeline
    print('✅ Diffusers imported successfully')
except ImportError as e:
    print(f'❌ Diffusers import failed: {e}')
"

echo "✅ Setup complete! Ready to run RectifiedHR experiments."
echo ""
echo "🔥 Quick test command:"
echo "python master_runner.py --categories sd15_main --test-mode --max-prompts 1"
