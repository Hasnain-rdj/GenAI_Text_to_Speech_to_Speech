#!/usr/bin/env python3
"""
Streamlit Application Launcher
Launches the voice cloning pipeline with proper warning suppression
"""

import os
import sys
import subprocess
import warnings

def setup_environment():
    """Set up environment variables to suppress warnings"""
    # Suppress PyTorch warnings
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore"
    
    # Suppress Streamlit warnings
    os.environ["STREAMLIT_LOGGER_LEVEL"] = "ERROR"
    os.environ["STREAMLIT_CLIENT_TOOLBAR_MODE"] = "minimal"
    
    # Suppress Python warnings
    warnings.filterwarnings("ignore")

def launch_streamlit():
    """Launch Streamlit with proper configuration"""
    try:
        print("🎭 Starting Voice Cloning Pipeline...")
        print("🌐 The app will open in your browser shortly...")
        print()
        
        # Use virtual environment Python if available
        python_exe = "venv39/Scripts/python.exe"
        if not os.path.exists(python_exe):
            python_exe = sys.executable
        
        # Launch Streamlit with error-level logging
        subprocess.run([
            python_exe, "-m", "streamlit", "run", "app.py",
            "--logger.level=error",
            "--client.toolbarMode=minimal"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    setup_environment()
    launch_streamlit()
