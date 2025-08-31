"""
Download OpenVoice V2 models from HuggingFace
"""
import os
import requests
from pathlib import Path

def download_file(url, filepath):
    """Download a file from URL"""
    print(f"Downloading {filepath.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded {filepath.name} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")

def download_openvoice_models():
    """Download OpenVoice V2 models"""
    print("Downloading OpenVoice V2 models...")
    
    # Create directories
    base_dir = Path("OpenVoice/checkpoints")
    converter_dir = base_dir / "converter"
    base_speakers_dir = base_dir / "base_speakers" / "EN"
    
    converter_dir.mkdir(parents=True, exist_ok=True)
    base_speakers_dir.mkdir(parents=True, exist_ok=True)
    
    # HuggingFace URLs for OpenVoice V2
    models = {
        # Converter models
        "converter/config.json": "https://huggingface.co/myshell-ai/OpenVoice-v2/resolve/main/converter/config.json",
        "converter/checkpoint.pth": "https://huggingface.co/myshell-ai/OpenVoice-v2/resolve/main/converter/checkpoint.pth",
        
        # Base speaker models
        "base_speakers/EN/config.json": "https://huggingface.co/myshell-ai/OpenVoice-v2/resolve/main/base_speakers/EN/config.json", 
        "base_speakers/EN/checkpoint.pth": "https://huggingface.co/myshell-ai/OpenVoice-v2/resolve/main/base_speakers/EN/checkpoint.pth",
        "base_speakers/EN/en_default_se.pth": "https://huggingface.co/myshell-ai/OpenVoice-v2/resolve/main/base_speakers/EN/en_default_se.pth"
    }
    
    for relative_path, url in models.items():
        filepath = base_dir / relative_path
        
        if filepath.exists():
            print(f"✅ {relative_path} already exists")
            continue
            
        try:
            download_file(url, filepath)
        except Exception as e:
            print(f"❌ Failed to download {relative_path}: {e}")
            return False
    
    print("✅ All models downloaded successfully!")
    return True

if __name__ == "__main__":
    download_openvoice_models()
