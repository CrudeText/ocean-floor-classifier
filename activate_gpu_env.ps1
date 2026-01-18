# PowerShell script to activate the GPU-enabled virtual environment
# This environment uses Python 3.12 with CUDA-enabled PyTorch

Write-Host "Activating GPU-enabled virtual environment..." -ForegroundColor Green
& "venv_gpu\Scripts\Activate.ps1"

Write-Host "`nPython version:" -ForegroundColor Cyan
python --version

Write-Host "`nChecking PyTorch CUDA support..." -ForegroundColor Cyan
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "`nEnvironment ready! You can now run:" -ForegroundColor Green
Write-Host "  python -m ofc.gui.app" -ForegroundColor Yellow
