#!/bin/bash
# Week-1 Sprint Plan Dependency Installation Script
# This script installs all required dependencies for the DyGRAV Week-1 Sprint Plan

set -e  # Exit on any error

echo "🚀 Installing Week-1 Sprint Plan Dependencies"
echo "=============================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider creating one:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
    echo ""
    read -p "Continue without virtual environment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update pip
echo "📦 Updating pip..."
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA support if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   CUDA detected, installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   No CUDA detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements_week1.txt

# Install spaCy language model
echo "🌍 Installing spaCy English model..."
python -m spacy download en_core_web_sm

# Install NLTK data
echo "📖 Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    print('✓ NLTK data downloaded')
except Exception as e:
    print(f'⚠️  NLTK download failed: {e}')
"

# Install project in development mode
echo "🔧 Installing DyGRAV in development mode..."
pip install -e .

# Verify installations
echo "🧪 Verifying installations..."

# Test PyTorch
python -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
"

# Test OpenCLIP
python -c "
try:
    import open_clip
    print(f'✓ OpenCLIP available')
    models = open_clip.list_pretrained()
    vit_models = [m for m in models if 'ViT-L-14' in str(m)]
    print(f'  ViT-L-14 models: {len(vit_models)} available')
except ImportError as e:
    print(f'❌ OpenCLIP not available: {e}')
"

# Test Optuna
python -c "
try:
    import optuna
    print(f'✓ Optuna {optuna.__version__}')
except ImportError as e:
    print(f'❌ Optuna not available: {e}')
"

# Test spaCy
python -c "
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print(f'✓ spaCy with en_core_web_sm model')
except Exception as e:
    print(f'❌ spaCy model not available: {e}')
"

# Test other key dependencies
python -c "
import sys
modules = [
    'numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas',
    'sklearn', 'cv2', 'PIL', 'transformers', 'datasets'
]

for module in modules:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError:
        print(f'❌ {module} not available')
"

echo ""
echo "🎉 Installation Complete!"
echo "========================="
echo ""
echo "Next steps:"
echo "1. Test the installation: python examples/week1_sprint_demo.py"
echo "2. Run VLM benchmark: python tools/bench_vlm.py"
echo "3. Build RxR-FG benchmark: python tools/build_rxr_fg.py"
echo "4. Validate backbone: python tools/validate_backbone.py --backbone hamt"
echo ""
echo "For GPU acceleration, ensure CUDA is properly installed."
echo "For best OpenCLIP performance, consider using a GPU-enabled environment."
