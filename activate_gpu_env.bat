@echo off
REM Batch script to activate the GPU-enabled virtual environment
REM This environment uses Python 3.12 with CUDA-enabled PyTorch

echo Activating GPU-enabled virtual environment...
call venv_gpu\Scripts\activate.bat

echo.
echo Python version:
python --version

echo.
echo Checking PyTorch CUDA support...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo Environment ready! You can now run:
echo   python -m ofc.gui.app
