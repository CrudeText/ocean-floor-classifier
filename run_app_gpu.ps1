# Run the application with GPU-enabled environment
Write-Host "Activating GPU environment and starting application..." -ForegroundColor Green
& "venv_gpu\Scripts\Activate.ps1"
python -m ofc.gui.app
