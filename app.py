"""
Complete Voice Cloning Pipeline - Streamlit Application
Combines PDF text extraction, text-to-speech, and voice cloning in one app
"""

import streamlit as st
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time
import tempfile
import shutil
import warnings

# Suppress PyTorch warnings and Streamlit file watcher warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set environment variables to suppress torch warnings
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

# Configure page
st.set_page_config(
    page_title="Voice Cloning Pipeline",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Install required packages if not present
@st.cache_resource
def install_dependencies():
    """Install required packages"""
    required_packages = {
        "PyPDF2": "PyPDF2",
        "gtts": "gTTS",
        "pydub": "pydub",
        "torch": "torch",
        "librosa": "librosa",
        "soundfile": "soundfile",
        "requests": "requests"
    }
    
    for package, install_name in required_packages.items():
        try:
            __import__(package.lower() if package != "PyPDF2" else "PyPDF2")
        except ImportError:
            with st.spinner(f"Installing {install_name}..."):
                subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

# Initialize dependencies
install_dependencies()

# Import after installation
import PyPDF2
from gtts import gTTS
from pydub import AudioSegment
import re

# Suppress torch warnings specifically for Streamlit
import logging
logging.getLogger("torch").setLevel(logging.ERROR)

# Import torch with warning suppression
try:
    import torch
    # Suppress PyTorch warnings without breaking functionality
    torch._C._set_print_stacktraces_on_fatal_signal(False)
except (ImportError, AttributeError):
    torch = None

# Constants
PYTHON_EXE = "venv39\\Scripts\\python.exe"

def clean_text(text):
    """Clean extracted text by removing extra whitespace and unwanted characters"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def extract_text_from_pdf(pdf_file, max_chars=100_000):
    """Extract and clean text from PDF file"""
    try:
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Get total pages
        total_pages = len(pdf_reader.pages)
        
        # Extract text
        extracted_text = ""
        chars_extracted = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num in range(total_pages):
            if chars_extracted >= max_chars:
                break
            
            # Extract text from current page
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            # Clean the text
            page_text = clean_text(page_text)
            
            # Add to total text
            remaining_chars = max_chars - chars_extracted
            extracted_text += page_text[:remaining_chars]
            chars_extracted = len(extracted_text)
            
            # Update progress
            progress = min(1.0, chars_extracted / max_chars)
            progress_bar.progress(progress)
            status_text.text(f"Progress: {progress*100:.1f}% ({chars_extracted}/{max_chars} characters)")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Text extraction complete!")
        
        return extracted_text[:max_chars]
        
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def text_to_speech_conversion(text, text_portion=1.0, lang='en'):
    """Convert text to speech using gTTS"""
    try:
        # Calculate text length based on portion
        text_length = int(len(text) * text_portion)
        text_to_convert = text[:text_length]
        
        with st.spinner("Converting text to speech..."):
            # Create gTTS object
            tts = gTTS(text=text_to_convert, lang=lang, slow=False)
            
            # Create temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = f"temp_audio_{timestamp}.mp3"
            
            # Save to file
            tts.save(temp_file)
            
            return temp_file
            
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None

def check_openvoice_setup():
    """Check if OpenVoice setup is complete"""
    # Check virtual environment
    if not Path(PYTHON_EXE).exists():
        return False, "Python 3.9 virtual environment not found!"
    
    # Check OpenVoice
    if not Path("OpenVoice").exists():
        return False, "OpenVoice directory not found!"
    
    return True, "Setup check passed"

def run_openvoice_cloning(source_file, reference_file):
    """Run OpenVoice V2 voice cloning"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"openvoice_cloned_{timestamp}.wav"
        
        # OpenVoice cloning script
        cloning_code = f'''
import sys
import os
sys.path.insert(0, "OpenVoice")

# Set environment for Unicode handling and FFmpeg
os.environ["PYTHONIOENCODING"] = "utf-8"
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
        
        # Extract embeddings
        print("Extracting source embedding...")
        source_se, _ = se_extractor.get_se("{source_file}", tone_color_converter)
        
        print("Extracting reference embedding...")
        reference_se, _ = se_extractor.get_se("{reference_file}", tone_color_converter)
        
        # Convert voice
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
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(cloning_code)
        
        # Set environment for Unicode handling and FFmpeg
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PATH"] = f"{os.getcwd()};{env.get('PATH', '')}"
        
        # Run the script
        result = subprocess.run([PYTHON_EXE, script_path], 
                              capture_output=True, text=True, timeout=600, env=env)
        
        # Clean up temp script
        if Path(script_path).exists():
            os.remove(script_path)
        
        if result.returncode == 0 and Path(output_file).exists():
            return output_file
        else:
            st.error(f"Voice cloning failed: {result.stderr}")
            return None
            
    except Exception as e:
        st.error(f"Error in voice cloning: {str(e)}")
        return None

# Streamlit App Layout
def main():
    st.title("🎭 Complete Voice Cloning Pipeline")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Choose Section:",
        ["📖 PDF Text Extraction", "🎤 Text to Speech", "🎭 Voice Cloning", "🔄 Complete Pipeline"]
    )
    
    if section == "📖 PDF Text Extraction":
        st.header("📖 PDF Text Extraction")
        st.markdown("Upload a PDF file to extract text (first 100,000 characters)")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Extract text
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
            
            if extracted_text:
                st.success(f"✅ Successfully extracted {len(extracted_text)} characters")
                
                # Show preview
                st.subheader("Text Preview")
                st.text_area("Extracted Text (first 500 characters)", 
                           extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                           height=200)
                
                # Download button
                st.download_button(
                    label="📥 Download Extracted Text",
                    data=extracted_text,
                    file_name=f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Store in session state for next steps
                st.session_state.extracted_text = extracted_text
    
    elif section == "🎤 Text to Speech":
        st.header("🎤 Text to Speech Conversion")
        st.markdown("Convert text to speech using Google Text-to-Speech")
        
        # Text input options
        text_source = st.radio("Text Source:", ["Upload Text File", "Use Extracted Text", "Enter Text Manually"])
        
        text_to_convert = None
        
        if text_source == "Upload Text File":
            uploaded_text_file = st.file_uploader("Choose a text file", type="txt")
            if uploaded_text_file is not None:
                text_to_convert = uploaded_text_file.read().decode("utf-8")
        
        elif text_source == "Use Extracted Text":
            if "extracted_text" in st.session_state:
                text_to_convert = st.session_state.extracted_text
                st.info(f"Using extracted text ({len(text_to_convert)} characters)")
            else:
                st.warning("No extracted text found. Please extract text from PDF first.")
        
        elif text_source == "Enter Text Manually":
            text_to_convert = st.text_area("Enter text to convert:", height=200)
        
        if text_to_convert:
            # Text portion selection
            portion_option = st.selectbox(
                "Select text portion:",
                ["Full text (100%)", "First 50%", "First 30%", "First 20%", "First 10%"]
            )
            
            portion_map = {
                "Full text (100%)": 1.0,
                "First 50%": 0.5,
                "First 30%": 0.3,
                "First 20%": 0.2,
                "First 10%": 0.1
            }
            
            text_portion = portion_map[portion_option]
            
            # Language selection
            language = st.selectbox("Select Language:", ["en", "es", "fr", "de", "it"])
            
            if st.button("🎵 Convert to Speech"):
                audio_file = text_to_speech_conversion(text_to_convert, text_portion, language)
                
                if audio_file and Path(audio_file).exists():
                    st.success("✅ Audio conversion completed!")
                    
                    # Audio player
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Audio File",
                        data=audio_bytes,
                        file_name=f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                        mime="audio/mp3"
                    )
                    
                    # Store in session state
                    st.session_state.generated_audio = audio_file
                    
                    # Clean up temp file
                    try:
                        os.remove(audio_file)
                    except:
                        pass
    
    elif section == "🎭 Voice Cloning":
        st.header("🎭 Voice Cloning with OpenVoice V2")
        st.markdown("Clone voice using OpenVoice V2 technology")
        
        # Check setup
        setup_ok, setup_msg = check_openvoice_setup()
        if not setup_ok:
            st.error(f"❌ {setup_msg}")
            st.info("💡 Please run setup_complete.py first")
            return
        else:
            st.success(f"✅ {setup_msg}")
        
        # File uploads
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Source Audio")
            source_option = st.radio("Source Audio:", ["Upload File", "Use Generated Audio"])
            
            source_file = None
            if source_option == "Upload File":
                uploaded_source = st.file_uploader("Choose source audio file", 
                                                 type=["mp3", "wav", "m4a"], key="source")
                if uploaded_source:
                    # Save uploaded file temporarily
                    source_file = f"temp_source_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{uploaded_source.name.split('.')[-1]}"
                    with open(source_file, "wb") as f:
                        f.write(uploaded_source.getbuffer())
            
            elif source_option == "Use Generated Audio":
                if "generated_audio" in st.session_state:
                    source_file = st.session_state.generated_audio
                    st.info("Using previously generated audio")
                else:
                    st.warning("No generated audio found. Please convert text to speech first.")
        
        with col2:
            st.subheader("🎤 Reference Audio")
            uploaded_reference = st.file_uploader("Choose reference voice file", 
                                                type=["mp3", "wav", "m4a"], key="reference")
            
            reference_file = None
            if uploaded_reference:
                # Save uploaded file temporarily
                reference_file = f"temp_reference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{uploaded_reference.name.split('.')[-1]}"
                with open(reference_file, "wb") as f:
                    f.write(uploaded_reference.getbuffer())
        
        if source_file and reference_file:
            if st.button("🎭 Start Voice Cloning"):
                with st.spinner("Cloning voice... This may take several minutes."):
                    cloned_file = run_openvoice_cloning(source_file, reference_file)
                
                if cloned_file and Path(cloned_file).exists():
                    st.success("✅ Voice cloning completed!")
                    
                    # Audio player
                    with open(cloned_file, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Cloned Audio",
                        data=audio_bytes,
                        file_name=cloned_file,
                        mime="audio/wav"
                    )
                    
                    # Clean up temp files
                    for temp_file in [source_file, reference_file]:
                        try:
                            if temp_file.startswith("temp_"):
                                os.remove(temp_file)
                        except:
                            pass
                else:
                    st.error("❌ Voice cloning failed!")
    
    elif section == "🔄 Complete Pipeline":
        st.header("🔄 Complete Pipeline")
        st.markdown("Run the entire pipeline from PDF to cloned voice")
        
        # Step 1: PDF Upload
        st.subheader("Step 1: Upload PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf", key="pipeline_pdf")
        
        # Step 2: Reference Audio
        st.subheader("Step 2: Upload Reference Voice")
        uploaded_ref = st.file_uploader("Choose reference voice file", 
                                      type=["mp3", "wav", "m4a"], key="pipeline_ref")
        
        # Step 3: Configuration
        st.subheader("Step 3: Configuration")
        col1, col2 = st.columns(2)
        with col1:
            text_portion = st.selectbox(
                "Text portion:",
                ["First 10%", "First 20%", "First 30%", "First 50%", "Full text"]
            )
        with col2:
            language = st.selectbox("Language:", ["en", "es", "fr", "de", "it"], key="pipeline_lang")
        
        if uploaded_pdf and uploaded_ref:
            if st.button("🚀 Run Complete Pipeline"):
                progress_container = st.container()
                
                with progress_container:
                    # Step 1: Extract text
                    st.info("📖 Step 1: Extracting text from PDF...")
                    extracted_text = extract_text_from_pdf(uploaded_pdf)
                    
                    if not extracted_text:
                        st.error("Failed to extract text from PDF")
                        return
                    
                    st.success(f"✅ Extracted {len(extracted_text)} characters")
                    
                    # Step 2: Convert to speech
                    st.info("🎤 Step 2: Converting text to speech...")
                    portion_map = {
                        "First 10%": 0.1, "First 20%": 0.2, "First 30%": 0.3, 
                        "First 50%": 0.5, "Full text": 1.0
                    }
                    
                    audio_file = text_to_speech_conversion(
                        extracted_text, 
                        portion_map[text_portion], 
                        language
                    )
                    
                    if not audio_file:
                        st.error("Failed to convert text to speech")
                        return
                    
                    st.success("✅ Audio conversion completed")
                    
                    # Step 3: Save reference file
                    reference_file = f"temp_reference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{uploaded_ref.name.split('.')[-1]}"
                    with open(reference_file, "wb") as f:
                        f.write(uploaded_ref.getbuffer())
                    
                    # Step 4: Voice cloning
                    st.info("🎭 Step 3: Cloning voice...")
                    
                    # Check setup first
                    setup_ok, setup_msg = check_openvoice_setup()
                    if not setup_ok:
                        st.error(f"❌ {setup_msg}")
                        st.info("💡 Please run setup_complete.py first")
                        return
                    
                    cloned_file = run_openvoice_cloning(audio_file, reference_file)
                    
                    if cloned_file and Path(cloned_file).exists():
                        st.success("🎉 Complete pipeline finished successfully!")
                        
                        # Show results
                        st.subheader("Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original TTS Audio:**")
                            with open(audio_file, "rb") as f:
                                st.audio(f.read(), format="audio/mp3")
                        
                        with col2:
                            st.write("**Cloned Voice Audio:**")
                            with open(cloned_file, "rb") as f:
                                cloned_audio = f.read()
                            st.audio(cloned_audio, format="audio/wav")
                            
                            # Download button for final result
                            st.download_button(
                                label="📥 Download Final Cloned Audio",
                                data=cloned_audio,
                                file_name=cloned_file,
                                mime="audio/wav"
                            )
                        
                        # Clean up temp files
                        for temp_file in [audio_file, reference_file]:
                            try:
                                if Path(temp_file).exists():
                                    os.remove(temp_file)
                            except:
                                pass
                    else:
                        st.error("❌ Voice cloning failed!")

if __name__ == "__main__":
    main()
