#!/bin/bash
# diagnose_venv.sh - Diagnose virtual environment issues

echo "🔍 DIAGNOSING VIRTUAL ENVIRONMENT ISSUES"
echo "========================================"

# Load the same modules you're using
echo "📦 Loading modules..."
module load gcc11/11.3.0
module load cuda11.2/toolkit/11.2.2  # Match what you actually have
module load cudnn8.1-cuda11.2/8.1.1.33
module load python/3.8.9

echo ""
echo "1. 🔍 CHECKING ENVIRONMENT"
echo "------------------------"
echo "Current directory: $(pwd)"
echo "Current user: $(whoami)"
echo "Available Python: $(which python3)"
echo "Python version: $(python3 --version 2>&1)"
echo ""

echo "2. 🗂️ CHECKING VENV STRUCTURE"
echo "-----------------------------"
if [ -d "venv" ]; then
    echo "✅ venv directory exists"
    echo "venv size: $(du -sh venv 2>/dev/null | cut -f1)"
    
    if [ -f "venv/bin/python" ]; then
        echo "✅ venv/bin/python exists"
        echo "Python in venv: $(venv/bin/python --version 2>&1)"
    else
        echo "❌ venv/bin/python missing"
    fi
    
    if [ -f "venv/bin/activate" ]; then
        echo "✅ venv/bin/activate exists"
    else
        echo "❌ venv/bin/activate missing"
    fi
    
    echo ""
    echo "📚 Checking installed packages in venv:"
    if [ -d "venv/lib/python3.8/site-packages" ]; then
        echo "✅ Site-packages directory exists"
        echo "Installed packages:"
        ls venv/lib/python3.8/site-packages/ | grep -E "(torch|diffusers|numpy)" | head -10
    else
        echo "❌ Site-packages directory missing"
        echo "Available lib directories:"
        find venv/lib/ -name "python*" -type d 2>/dev/null
    fi
else
    echo "❌ venv directory does not exist"
fi

echo ""
echo "3. 🧪 TESTING ACTIVATION"
echo "-----------------------"
if [ -f "venv/bin/activate" ]; then
    echo "Activating venv..."
    source venv/bin/activate
    
    echo "After activation:"
    echo "which python: $(which python)"
    echo "python version: $(python --version 2>&1)"
    echo "which pip: $(which pip)"
    
    echo ""
    echo "📦 Testing package imports:"
    
    # Test each package individually
    echo -n "numpy: "
    python -c "import numpy; print('✅ OK -', numpy.__version__)" 2>/dev/null || echo "❌ FAIL"
    
    echo -n "torch: "
    python -c "import torch; print('✅ OK -', torch.__version__)" 2>/dev/null || echo "❌ FAIL"
    
    echo -n "diffusers: "
    python -c "import diffusers; print('✅ OK -', diffusers.__version__)" 2>/dev/null || echo "❌ FAIL"
    
    echo -n "CUDA: "
    python -c "import torch; print('✅ OK -', torch.cuda.is_available())" 2>/dev/null || echo "❌ FAIL"
    
else
    echo "❌ Cannot activate - venv/bin/activate missing"
fi

echo ""
echo "4. 🔧 RECOMMENDATIONS"
echo "--------------------"

if [ ! -d "venv" ]; then
    echo "➤ Need to create venv: python3 -m venv venv"
elif [ ! -f "venv/bin/python" ]; then
    echo "➤ venv is corrupted, recreate: rm -rf venv && python3 -m venv venv"
else
    echo "➤ venv seems OK, check activation and package installation"
fi

echo ""
echo "🎯 QUICK FIX COMMANDS"
echo "--------------------"
echo "# If venv is broken:"
echo "rm -rf venv"
echo "python3 -m venv venv"
echo "source venv/bin/activate"
echo "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
echo "pip install diffusers transformers"
echo ""
echo "# If just activation issues:"
echo "source venv/bin/activate"
echo "python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
