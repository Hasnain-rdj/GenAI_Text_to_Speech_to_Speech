# GenAI Text-to-Speech Voice Cloning Pipeline

A complete voice cloning pipeline that converts PDF text to speech and then clones it with a reference voice using OpenVoice V2.

## Features

- **PDF Text Extraction**: Extract text from PDF files (first 100k characters)
- **Text-to-Speech**: Convert extracted text to speech using Google TTS
- **Voice Cloning**: Clone the generated speech with a reference voice using OpenVoice V2
- **Streamlit Web Interface**: Easy-to-use web application for the complete pipeline

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Hasnain-rdj/GenAI_Text_to_Speech_to_Speech.git
cd GenAI_Text_to_Speech_to_Speech
```

### 2. Create Virtual Environment

```bash
python -m venv venv39
# On Windows:
venv39\Scripts\activate
# On Linux/Mac:
source venv39/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup OpenVoice

Since OpenVoice is not included in the repository (due to size), you need to clone it separately:

```bash
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e .
cd ..
```

### 5. Download FFmpeg

Download `ffmpeg.exe` and `ffprobe.exe` and place them in the project root directory.

### 6. Run Setup Script

```bash
python setup_complete.py
```

## Usage

### Option 1: Streamlit Web Application (Recommended)

```bash
streamlit run app.py
```

This will open a web interface where you can:
1. Upload PDF files and extract text
2. Convert text to speech
3. Clone voice with reference audio
4. Run the complete pipeline in one go

### Option 2: Individual Scripts

#### Extract Text from PDF
```bash
python main.py
```

#### Convert Text to Speech
```bash
python text_to_speech.py
```

#### Clone Voice
```bash
python openvoice_processor.py
```

## File Structure

```
├── app.py                 # Streamlit web application
├── main.py               # PDF text extraction
├── text_to_speech.py     # Text-to-speech conversion
├── openvoice_processor.py # Voice cloning
├── setup_complete.py     # Setup script
├── download_models.py    # Model downloading utility
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
└── OpenVoice/           # OpenVoice repository (clone separately)
```

## Requirements

- Python 3.9+
- PyTorch
- Streamlit
- OpenVoice V2
- FFmpeg
- Various audio processing libraries (see requirements.txt)

## Notes

- Large files (models, executables, virtual environments) are excluded from the repository
- You need to set up OpenVoice separately by cloning their repository
- FFmpeg executables need to be downloaded separately
- The application supports multiple audio formats (MP3, WAV, M4A)

## Troubleshooting

1. **Import Errors**: Make sure all dependencies are installed in the virtual environment
2. **FFmpeg Not Found**: Ensure ffmpeg.exe and ffprobe.exe are in the project directory
3. **OpenVoice Errors**: Verify OpenVoice is properly cloned and installed
4. **Memory Issues**: Reduce text portion size for large documents

## License

This project uses OpenVoice V2 which has its own license. Please check the OpenVoice repository for licensing terms.
