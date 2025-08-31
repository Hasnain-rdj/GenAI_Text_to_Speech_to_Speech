"""
OpenVoice V2 Processor
Main working script for voice cloning using OpenVoice V2
Requires setup_complete.py to be run first
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Python executable from virtual environment
PYTHON_EXE = "venv39\\Scripts\\python.exe"

def check_setup():
    """Check if setup is complete"""
    print("🔍 Checking setup...")
    
    # Check virtual environment
    if not Path(PYTHON_EXE).exists():
        print("❌ Python 3.9 virtual environment not found!")
        print("💡 Run setup_complete.py first")
        return False
    
    # Check OpenVoice
    if not Path("OpenVoice").exists():
        print("❌ OpenVoice directory not found!")
        print("💡 Run setup_complete.py first")
        return False
    
    print("✅ Setup check passed")
    return True

def find_audio_files():
    """Find source and reference audio files"""
    print("🔍 Finding audio files...")
    
    # Source files (TTS generated or processed)
    source_patterns = [
        "Life_3.0_AudioBook_*.mp3",
        "final_processed_*.wav",
        "voice_cloned_*.mp3"
    ]
    
    source_files = []
    for pattern in source_patterns:
        source_files.extend(list(Path(".").glob(pattern)))
    
    # Reference files
    reference_files = []
    ref_names = ["cloned_voice.mp3", "My_Voice.mp3", "my_voice.mp3"]
    for name in ref_names:
        if Path(name).exists():
            reference_files.append(name)
    
    print(f"📁 Found {len(source_files)} source file(s)")
    print(f"🎤 Found {len(reference_files)} reference file(s)")
    
    return source_files, reference_files

def run_openvoice_cloning(source_file, reference_file, output_file):
    """Run OpenVoice V2 voice cloning"""
    print(f"🎭 Starting OpenVoice V2 cloning...")
    print(f"📖 Source: {source_file}")
    print(f"🎤 Reference: {reference_file}")
    print(f"📤 Output: {output_file}")
    
    # OpenVoice cloning script
    cloning_code = f'''
import sys
import os
sys.path.insert(0, "OpenVoice")

# Set environment for Unicode handling and FFmpeg
os.environ["PYTHONIOENCODING"] = "utf-8"
# Add current directory to PATH for ffmpeg.exe
import os
current_path = os.getcwd()
if current_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{{current_path}};{{os.environ.get('PATH', '')}}"

import torch
import requests
from pathlib import Path

# Import OpenVoice components
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

def download_models():
    """Download OpenVoice models if needed"""
    print("Checking/downloading models...")
    
    models_dir = Path("OpenVoice/checkpoints")
    converter_dir = models_dir / "converter"
    converter_dir.mkdir(parents=True, exist_ok=True)
    
    model_files = {{
        "config.json": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/converter/config.json",
        "checkpoint.pth": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/converter/checkpoint.pth"
    }}
    
    for filename, url in model_files.items():
        file_path = converter_dir / filename
        if not file_path.exists():
            print(f"Downloading {{filename}}...")
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {{filename}}")
            except Exception as e:
                print(f"Download failed: {{e}}")
                return False
        else:
            print(f"{{filename}} exists")
    
    return True

def clone_voice():
    """Perform voice cloning"""
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {{device}}")
        
        # Download models
        if not download_models():
            return False
        
        # Initialize converter
        ckpt_converter = "OpenVoice/checkpoints/converter"
        tone_color_converter = ToneColorConverter(
            f"{{ckpt_converter}}/config.json", 
            device=device
        )
        tone_color_converter.load_ckpt(f"{{ckpt_converter}}/checkpoint.pth")
        
        # Extract embeddings using the converter model
        print("Extracting source embedding...")
        source_se, _ = se_extractor.get_se("{source_file}", tone_color_converter)
        
        print("Extracting reference embedding...")
        reference_se, _ = se_extractor.get_se("{reference_file}", tone_color_converter)
        
        # Convert voice in smaller chunks to avoid memory issues
        print("Converting voice in chunks for large files...")
        
        # Load source audio to check duration
        import librosa
        y, sr = librosa.load("{source_file}", sr=None)
        duration = len(y) / sr
        
        if duration > 120:  # If longer than 2 minutes, process in chunks
            print(f"Large file detected ({{duration:.1f}}s), processing in 60s chunks...")
            
            # Split into 60-second chunks
            chunk_duration = 60
            chunk_samples = chunk_duration * sr
            num_chunks = int(duration // chunk_duration) + (1 if duration % chunk_duration > 0 else 0)
            
            print(f"Processing {{num_chunks}} chunks...")
            
            output_chunks = []
            for i in range(min(3, num_chunks)):  # Process only first 3 chunks (3 minutes max)
                print(f"Processing chunk {{i+1}}/{{min(3, num_chunks)}}...")
                
                start_sample = i * chunk_samples
                end_sample = min((i + 1) * chunk_samples, len(y))
                
                # Create temporary chunk file
                chunk_file = f"temp_chunk_{{i}}.wav"
                import soundfile as sf
                sf.write(chunk_file, y[start_sample:end_sample], sr)
                
                # Convert this chunk
                chunk_output = f"temp_output_{{i}}.wav"
                try:
                    tone_color_converter.convert(
                        audio_src_path=chunk_file,
                        src_se=source_se,
                        tgt_se=reference_se,
                        output_path=chunk_output,
                        message=f"OpenVoice V2 chunk {{i+1}}"
                    )
                    output_chunks.append(chunk_output)
                    print(f"Chunk {{i+1}} completed")
                except Exception as e:
                    print(f"Chunk {{i+1}} failed: {{e}}")
                    continue
                finally:
                    # Clean up temp chunk
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
            
            # Combine output chunks
            if output_chunks:
                print("Combining chunks...")
                from pydub import AudioSegment
                combined = AudioSegment.empty()
                for chunk_path in output_chunks:
                    if os.path.exists(chunk_path):
                        chunk_audio = AudioSegment.from_wav(chunk_path)
                        combined += chunk_audio
                        os.remove(chunk_path)  # Clean up
                
                # Export final result
                combined.export("{output_file}", format="wav")
                print("Chunks combined successfully!")
            else:
                print("No chunks were processed successfully")
                return False
                
        else:
            # Process normally for short files
            tone_color_converter.convert(
                audio_src_path="{source_file}",
                src_se=source_se,
                tgt_se=reference_se,
                output_path="{output_file}",
                message="OpenVoice V2 cloning"
            )
        
        print("Voice cloning completed!")
        return True
        
    except Exception as e:
        print(f"Voice cloning failed: {{e}}")
        import traceback
        traceback.print_exc()
        return False

# Run the cloning
if clone_voice():
    print("SUCCESS: Voice cloning completed!")
else:
    print("FAILED: Voice cloning failed!")
'''
    
    # Write and execute the cloning script
    script_path = "temp_cloning.py"
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(cloning_code)
        
        # Set environment for Unicode handling and FFmpeg
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PATH"] = f"{os.getcwd()};{env.get('PATH', '')}"  # Add current directory for ffmpeg.exe
        
        # Run the script
        result = subprocess.run([PYTHON_EXE, script_path], 
                              capture_output=True, text=True, timeout=600, env=env)
        
        print("📤 Cloning output:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Errors/Warnings:")
            print(result.stderr)
        
        # Clean up temp script
        if Path(script_path).exists():
            os.remove(script_path)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Cloning timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running cloning: {e}")
        return False
    finally:
        # Clean up temp script
        if Path(script_path).exists():
            os.remove(script_path)

def install_missing_packages():
    """Install any missing packages"""
    print("📦 Checking dependencies...")
    
    essential_packages = ["torch", "librosa", "soundfile", "requests"]
    
    for package in essential_packages:
        try:
            result = subprocess.run([
                PYTHON_EXE, "-c", f"import {package}; print('OK')"
            ], capture_output=True, text=True)
            
            if "OK" in result.stdout:
                print(f"✅ {package}")
            else:
                print(f"📦 Installing {package}...")
                subprocess.run([PYTHON_EXE, "-m", "pip", "install", package], check=True)
                
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}")

def main():
    """Main processing function"""
    print("🎭 OPENVOICE V2 PROCESSOR")
    print("=" * 40)
    
    # Check setup
    if not check_setup():
        return False
    
    # Install missing packages
    install_missing_packages()
    
    # Find audio files
    source_files, reference_files = find_audio_files()
    
    if not source_files:
        print("❌ No source audio files found!")
        print("💡 Need: Life_3.0_AudioBook_*.mp3 or similar")
        return False
    
    if not reference_files:
        print("❌ No reference voice files found!")
        print("💡 Need: cloned_voice.mp3, My_Voice.mp3, or my_voice.mp3")
        return False
    
    # Select files
    source_file = max(source_files, key=lambda x: x.stat().st_mtime)
    reference_file = reference_files[0]
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"openvoice_cloned_{timestamp}.wav"
    
    # Run voice cloning
    success = run_openvoice_cloning(str(source_file), reference_file, output_file)
    
    if success and Path(output_file).exists():
        file_size = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"\n🎉 CLONING COMPLETED!")
        print(f"📁 Output: {output_file}")
        print(f"📊 Size: {file_size:.1f} MB")
        return True
    else:
        print(f"\n❌ CLONING FAILED!")
        print("💡 Check error messages above")
        return False

if __name__ == "__main__":
    try:
        import time
        start = time.time()
        success = main()
        duration = time.time() - start
        print(f"\n⏱️ Processing time: {duration:.2f} seconds")
        
        if success:
            print("🎊 OpenVoice V2 processing completed successfully!")
        else:
            print("❌ OpenVoice V2 processing failed!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
