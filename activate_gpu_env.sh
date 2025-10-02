#!/bin/bash
# activate_gpu_env.sh - Quick activation script for compute nodes

echo "🚀 Activating RectifiedHR GPU environment..."

# Check if we're on a compute node
if [[ $(hostname) == *"node"* ]] || [[ $(hostname) == *"ciscluster"* ]]; then
    echo "📍 On compute node: $(hostname)"
    
    # Load CUDA modules
    echo "📦 Loading CUDA modules..."
    module load gcc11/11.3.0 2>/dev/null
    module load cuda11.2/toolkit/11.2.2 2>/dev/null  
    module load cudnn8.1-cuda11.2/8.1.1.33 2>/dev/null
    
    # Activate virtual environment
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    
    # Verify setup
    echo "🧪 Verifying environment..."
    echo "Python version: $(python --version)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'False')"
    
    echo "✅ Environment activated!"
    echo "💡 You can now run: python run_single_experiment.py --prompt 'test'"
    
else
    echo "📍 On login node, activating basic environment..."
    source venv/bin/activate
    echo "✅ Basic environment activated!"
fi
