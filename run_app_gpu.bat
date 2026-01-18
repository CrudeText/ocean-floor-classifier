@echo off
REM Run the application with GPU-enabled environment
echo Activating GPU environment and starting application...
call venv_gpu\Scripts\activate.bat
python -m ofc.gui.app
