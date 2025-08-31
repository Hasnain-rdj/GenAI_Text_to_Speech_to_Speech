# Troubleshooting Guide

## Common Issues and Solutions

### 1. NumPy Compatibility Error

**Error:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2 as it may crash
```

**Solution:**
```bash
pip install "numpy<2.0"
```

### 2. Missing ipdb Module

**Error:**
```
ModuleNotFoundError: No module named 'ipdb'
```

**Solution:**
```bash
pip install ipdb
```

### 3. PyTorch Warning on Startup

**Error:**
```
Examining the path of torch.classes raised:
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
```

**Solution:**
This is handled automatically in the updated app.py with proper warning suppression.

### 4. Audio Format Issues

**Error:**
```
Error extracting source/reference embedding
```

**Solution:**
The app now automatically preprocesses audio files to the correct format:
- 16kHz sample rate
- Mono channel
- WAV format

### 5. Voice Cloning VAD Issues

**Error:**
```
RuntimeError: Numpy is not available
```

**Solution:**
Use the simplified voice cloning approach implemented in the app, which:
- Uses librosa for audio loading
- Handles format conversion automatically
- Includes fallback methods for embedding extraction

### 6. Git Large Files Error

**Error:**
```
error: GH001: Large files detected
```

**Solution:**
The updated .gitignore excludes all large files:
- Virtual environments
- Model files
- Audio files
- Executables

## Installation Steps

1. **Create Virtual Environment:**
```bash
python -m venv venv39
venv39\Scripts\activate  # Windows
source venv39/bin/activate  # Linux/Mac
```

2. **Install Dependencies:**
```bash
pip install "numpy<2.0"
pip install streamlit PyPDF2 gTTS pydub torch librosa soundfile ipdb
```

3. **Setup OpenVoice:**
```bash
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e .
cd ..
```

4. **Run Application:**
```bash
streamlit run app.py --logger.level=error
```

## Performance Tips

1. **Use shorter audio clips** for faster processing
2. **Ensure audio quality** is good for better cloning results
3. **Use WAV format** when possible for best compatibility
4. **Keep reference audio under 30 seconds** for optimal results

## Supported Formats

- **Input Audio:** MP3, WAV, M4A
- **Output Audio:** WAV (16kHz, mono)
- **PDF Files:** Standard PDF with extractable text
- **Text Files:** UTF-8 encoded

## Hardware Requirements

- **Minimum:** 8GB RAM, CPU-only processing
- **Recommended:** 16GB RAM, NVIDIA GPU with CUDA support
- **Storage:** At least 5GB free space for models and temporary files
