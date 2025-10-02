#!/bin/bash
# setup_cluster_env.sh - Comprehensive cluster environment setup for RectifiedHR

set -e  # Exit on any error

echo "🚀 Setting up RectifiedHR cluster environment..."
echo "=================================="

# Function to check if we're on erdos or compute node
check_node() {
    if [[ $(hostname) == *"erdos"* ]]; then
        echo "📍 Running on erdos (login node)"
        return 0
    elif [[ $(hostname) == *"node"* ]] || [[ $(hostname) == *"ciscluster"* ]]; then
        echo "📍 Running on compute node: $(hostname)"
        return 1
    else
        echo "⚠️  Unknown node: $(hostname)"
        return 2
    fi
}

# Function to setup on erdos
setup_erdos() {
    echo "🔧 Setting up environment on erdos..."
    
    # Check Python version
    echo "🐍 Checking Python version..."
    python --version
    which python
    
    # Remove existing venv if it exists
    if [ -d "venv" ]; then
        echo "🗑️  Removing existing venv..."
        rm -rf venv
    fi
    
    # Create new virtual environment with explicit Python 3
    echo "📦 Creating virtual environment..."
    python3 -m venv venv || python -m venv venv
    
    # Activate and upgrade pip
    echo "🔄 Activating environment and upgrading pip..."
    source venv/bin/activate
    pip install --upgrade pip
    
    # Install CPU version of PyTorch first (works on all nodes)
    echo "🏗️  Installing PyTorch CPU version..."
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu
    
    # Install other requirements
    echo "📚 Installing other packages..."
    pip install diffusers transformers accelerate
    pip install numpy scipy matplotlib pillow
    pip install clip-by-openai lpips
    pip install psutil GPUtil
    
    # Test installation
    echo "🧪 Testing installation..."
    python -c "import torch, diffusers, transformers; print('✅ All packages installed successfully')"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    
    echo "✅ erdos setup complete!"
    echo ""
    echo "📋 Next steps:"
    echo "1. SSH to compute node: ssh ciscluster, then ssh node001"
    echo "2. Run: cd /u/erdos/csga/as505/RectifiedHR"
    echo "3. Run: ./setup_cluster_env.sh"
}

# Function to setup on compute node
setup_compute_node() {
    echo "🔧 Setting up environment on compute node..."
    
    # Load required modules
    echo "📦 Loading CUDA modules..."
    module load gcc11/11.3.0 2>/dev/null || echo "⚠️  gcc11 module not available"
    module load cuda11.2/toolkit/11.2.2 2>/dev/null || echo "⚠️  CUDA module not available"
    module load cudnn8.1-cuda11.2/8.1.1.33 2>/dev/null || echo "⚠️  cuDNN module not available"
    
    # Check if venv exists
    if [ ! -d "venv" ]; then
        echo "❌ Virtual environment not found! Please run setup on erdos first."
        exit 1
    fi
    
    # Activate venv with explicit Python path
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    
    # Check Python version in venv
    echo "🐍 Checking Python in virtual environment..."
    python --version
    which python
    
    # Check if we have the right Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PYTHON_VERSION" < "3.8" ]]; then
        echo "❌ Wrong Python version in venv: $PYTHON_VERSION"
        echo "💡 The venv is not properly activated. Let's fix this..."
        
        # Try to find Python 3 and recreate symlinks
        PYTHON3_PATH=$(which python3 2>/dev/null || echo "")
        if [ -n "$PYTHON3_PATH" ]; then
            echo "🔧 Found Python 3 at: $PYTHON3_PATH"
            echo "🔗 Creating proper symlinks in venv..."
            
            # Backup and recreate python symlink in venv
            cd venv/bin
            if [ -f python ]; then
                mv python python.bak
            fi
            ln -sf "$PYTHON3_PATH" python
            cd ../..
            
            # Reactivate
            source venv/bin/activate
            echo "🔄 Reactivated with fixed Python:"
            python --version
        else
            echo "❌ No Python 3 found on this node!"
            exit 1
        fi
    fi
    
    # Install GPU version of PyTorch (only if we have proper Python)
    echo "🚀 Installing PyTorch with CUDA support..."
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --force-reinstall
    
    # Test CUDA availability
    echo "🧪 Testing CUDA availability..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')" 2>/dev/null || echo "No CUDA devices detected"
    
    echo "✅ Compute node setup complete!"
}

# Main execution
if check_node; then
    # On erdos
    setup_erdos
else
    # On compute node
    setup_compute_node
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "🚀 You can now run experiments with GPU acceleration!"
