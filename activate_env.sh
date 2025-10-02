#!/bin/bash
# activate_env.sh - Properly activate the environment on node001

echo "🔄 Activating RectifiedHR environment..."

# Load modules first (this is crucial!)
echo "📦 Loading required modules..."
module load gcc11/11.3.0
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33  
module load python/3.8.9

# Verify modules loaded
echo "✅ Modules loaded"
echo "Python: $(which python3)"
echo "CUDA: $(nvcc --version | grep release | head -1)"

# Activate virtual environment
echo "🐍 Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    
    echo "✅ Environment activated"
    echo "Active Python: $(which python)"
    echo "Python version: $(python --version)"
    
    # Test critical imports
    echo "🧪 Testing imports..."
    python -c "
import sys
print('Python path:', sys.executable)

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('✅ CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('✅ GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print('❌ PyTorch issue:', e)

try:
    import diffusers
    print('✅ Diffusers:', diffusers.__version__)
except Exception as e:
    print('❌ Diffusers issue:', e)
"
    
    echo ""
    echo "🎯 Environment ready! You can now run:"
    echo "python master_runner.py --test-mode --max-prompts 1"
    
else
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv"
fi
