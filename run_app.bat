@echo off
REM Streamlit startup script with warning suppression
REM This script suppresses PyTorch and Streamlit file watcher warnings

echo Starting Streamlit Voice Cloning Pipeline...
echo.

REM Set environment variables to suppress warnings
set TORCH_SHOW_CPP_STACKTRACES=0
set PYTHONWARNINGS=ignore
set STREAMLIT_LOGGER_LEVEL=ERROR
set STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

REM Start Streamlit
streamlit run app.py --logger.level=error

pause
