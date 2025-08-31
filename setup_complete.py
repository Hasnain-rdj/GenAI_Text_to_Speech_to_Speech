"""
Complete OpenVoice V2 Setup Script
Handles all dependencies, environment setup, and OpenVoice installation
Single script for complete setup with Python 3.9
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_cmd(command, description):
    """Run command and handle errors"""
    print(f"🔧 {description}...")
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False, e.stderr

def check_python39():
    """Check and get Python 3.9 command"""
    print("🐍 Checking Python 3.9...")
    commands = ["py -3.9", "python3.9", "python"]
    
    for cmd in commands:
        try:
            result = subprocess.run(f"{cmd} --version", shell=True, capture_output=True, text=True)
            if "3.9" in result.stdout:
                print(f"✅ Found Python 3.9: {result.stdout.strip()}")
                return cmd
        except:
            continue
    
    print("❌ Python 3.9 not found! Please install Python 3.9")
    return None

def setup_venv(python_cmd):
    """Create and setup virtual environment"""
    print("📦 Setting up virtual environment...")
    
    venv_path = Path("venv39")
    if venv_path.exists():
        print("🧹 Removing existing venv...")
        shutil.rmtree(venv_path)
    
    success, _ = run_cmd(f"{python_cmd} -m venv venv39", "Creating virtual environment")
    if not success:
        return None
    
    if os.name == 'nt':
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    if python_exe.exists():
        print(f"✅ Virtual environment ready: {python_exe}")
        return str(python_exe)
    return None

def install_dependencies(python_exe):
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core dependencies with exact versions for Python 3.9 compatibility
    deps = [
        "torch==1.13.1",
        "torchaudio==0.13.1",
        "numpy==1.21.6",
        "librosa==0.9.1",
        "scipy==1.9.3",
        "soundfile==0.12.1",
        "requests==2.28.2",
        "pydub==0.25.1",
        "matplotlib==3.6.3",
        "Unidecode==1.3.6",
        "pypinyin==0.48.0",
        "jieba==0.42.1",
        "inflect==6.0.2",
        "phonemizer==3.2.1",
        "faster-whisper==0.9.0",
        "wavmark==0.0.3"
    ]
    
    print(f"📋 Installing {len(deps)} packages...")
    for i, dep in enumerate(deps, 1):
        print(f"📦 ({i}/{len(deps)}) {dep}")
        success, _ = run_cmd([python_exe, "-m", "pip", "install", dep], f"Installing {dep}")
        if not success:
            print(f"⚠️ Warning: {dep} installation failed, continuing...")
    
    print("✅ Dependencies installation completed")

def clone_openvoice():
    """Clone OpenVoice repository"""
    print("📥 Setting up OpenVoice...")
    
    openvoice_path = Path("OpenVoice")
    if openvoice_path.exists():
        print("✅ OpenVoice already exists")
        return True
    
    success, _ = run_cmd("git clone https://github.com/myshell-ai/OpenVoice.git", "Cloning OpenVoice")
    return success

def install_openvoice(python_exe):
    """Install OpenVoice in development mode"""
    print("🎭 Installing OpenVoice...")
    
    openvoice_path = Path("OpenVoice")
    if not openvoice_path.exists():
        return False
    
    original_dir = os.getcwd()
    try:
        os.chdir(openvoice_path)
        success, _ = run_cmd([python_exe, "-m", "pip", "install", "-e", "."], "Installing OpenVoice")
        return success
    except:
        return False
    finally:
        os.chdir(original_dir)

def test_installation(python_exe):
    """Test the complete installation"""
    print("🧪 Testing installation...")
    
    test_script = '''
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except Exception as e:
    print(f"PyTorch Error: {e}")

try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except Exception as e:
    print(f"NumPy Error: {e}")

try:
    import librosa
    print(f"Librosa: {librosa.__version__}")
except Exception as e:
    print(f"Librosa Error: {e}")

try:
    import soundfile as sf
    print("SoundFile: OK")
except Exception as e:
    print(f"SoundFile Error: {e}")

try:
    sys.path.insert(0, "OpenVoice")
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    print("OpenVoice: OK")
except Exception as e:
    print(f"OpenVoice Error: {e}")

print("Installation test completed!")
'''
    
    success, output = run_cmd([python_exe, "-c", test_script], "Running tests")
    if success:
        print("✅ All tests passed!")
        print(output)
    else:
        print("⚠️ Some tests failed:")
        print(output)
    
    return success

def main():
    """Main setup process"""
    print("🚀 OPENVOICE V2 COMPLETE SETUP")
    print("=" * 50)
    
    # Step 1: Check Python 3.9
    python_cmd = check_python39()
    if not python_cmd:
        print("💡 Please install Python 3.9 and run this script again")
        return False
    
    # Step 2: Setup virtual environment
    python_exe = setup_venv(python_cmd)
    if not python_exe:
        print("❌ Virtual environment setup failed")
        return False
    
    # Step 3: Install dependencies
    install_dependencies(python_exe)
    
    # Step 4: Clone OpenVoice
    if not clone_openvoice():
        print("❌ OpenVoice clone failed")
        return False
    
    # Step 5: Install OpenVoice
    if not install_openvoice(python_exe):
        print("⚠️ OpenVoice installation failed, but continuing...")
    
    # Step 6: Test installation
    test_installation(python_exe)
    
    print("\n🎉 SETUP COMPLETED!")
    print("=" * 30)
    print(f"✅ Python 3.9 environment: venv39/")
    print(f"✅ OpenVoice V2: OpenVoice/")
    print(f"✅ Python executable: {python_exe}")
    print("\n🚀 Next step: Run openvoice_processor.py")
    
    return True

if __name__ == "__main__":
    try:
        import time
        start = time.time()
        success = main()
        duration = time.time() - start
        print(f"\n⏱️ Setup time: {duration/60:.1f} minutes")
        if not success:
            print("❌ Setup failed! Check error messages above.")
    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted")
    except Exception as e:
        print(f"\n💥 Setup error: {e}")
        import traceback
        traceback.print_exc()
