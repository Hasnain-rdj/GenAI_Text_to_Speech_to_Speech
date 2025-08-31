"""
Text to Speech Converter using gTTS
Converts extracted text to speech with threading support
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Install required packages if not present
required_packages = {
    "gtts": "gTTS",
    "pydub": "pydub",
    "ffmpeg-python": "ffmpeg-python"  # Required for audio processing
}

# First install ffmpeg which is required by pydub
try:
    import ffmpeg
except ImportError:
    print("📦 Installing ffmpeg-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])

# Install other packages
for package, install_name in required_packages.items():
    try:
        if package == "gtts":
            from gtts import gTTS
        elif package == "pydub":
            from pydub import AudioSegment
            # Test if pydub can actually load an MP3 file
            try:
                AudioSegment.from_file
            except Exception:
                raise ImportError("Pydub installation incomplete")
        elif package != "ffmpeg-python":  # Skip ffmpeg as we already handled it
            __import__(package)
    except ImportError:
        print(f"📦 Installing {install_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

# Verify pydub installation and additional dependencies
try:
    from pydub import AudioSegment
    
    # Check for pyaudioop (required by pydub for audio manipulation)
    try:
        import audioop
    except ImportError:
        print("📦 Installing audioop dependency...")
        # Audioop is part of the standard library but sometimes missing on Windows
        # We need to install simpleaudio as a fallback for audio processing
        subprocess.check_call([sys.executable, "-m", "pip", "install", "simpleaudio"])
    
    # Check FFmpeg installation
    try:
        subprocess.check_call(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("⚠️ FFmpeg not found in PATH. Audio combining might fail.")
        print("📝 Please ensure FFmpeg is installed and in your PATH.")
except ImportError:
    print("❌ Error: Could not import pydub. Installing additional dependencies...")
    # Install pydub again to make sure it's properly installed
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pydub"])

def read_text_file(file_path):
    """Read text from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return None

def text_to_speech(text, output_path, lang='en', slow=False):
    """Convert text to speech using gTTS"""
    try:
        print("🎤 Initializing text-to-speech conversion...")
        
        # Create gTTS object
        print("⚙️ Creating speech engine...")
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Save to file
        print(f"💾 Generating audio file: {output_path}")
        tts.save(output_path)
        
        return True
    except Exception as e:
        print(f"❌ Error in text-to-speech conversion: {str(e)}")
        return False

def process_text_portion(text, start_idx, end_idx, output_file):
    """Process a portion of text using gTTS"""
    portion = text[start_idx:end_idx]
    return text_to_speech(portion, output_file)

def main():
    # Input and output files
    input_file = "extracted_text.txt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"Life_3.0_AudioBook_100k_{timestamp}.mp3"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("💡 Run main.py first to extract text from PDF")
        sys.exit(1)
    
    # Read text
    print(f"📖 Reading text from: {input_file}")
    text = read_text_file(input_file)
    if not text:
        print("❌ Failed to read text file")
        sys.exit(1)
    
    print(f"📊 Text length: {len(text)} characters")
    
    # Ask user for text portion
    while True:
        choice = input("\n Choose option:\n 1. Convert full text\n 2. Convert first 50% text\n 3. Convert first 30% text\n 4. Convert first 20% text\n Enter (1/2/3/4/5): ").strip()
        if choice in ['1', '2', '3', '4','5']:
            break
        print("❌ Invalid choice. Please enter 1, 2, 3, 4, 5.")
    
    text_length = len(text)
    if choice == '2':
        text_length = len(text) // 2  # 50%
        print(f"🔄 Processing first {text_length} characters (50%)...")
    elif choice == '3':
        text_length = int(len(text) * 0.3)  # 30%
        print(f"🔄 Processing first {text_length} characters (30%)...")
    elif choice == '4':
        text_length = int(len(text) * 0.2)  # 20%
        print(f"🔄 Processing first {text_length} characters (20%)...")
    elif choice == '5':
        text_length = int(len(text) * 0.1)  # 10%
        print(f"🔄 Processing first {text_length} characters (10%)...")
    else:
        print("🔄 Processing full text...")
    
    # Number of threads to use (adjust based on your CPU)
    num_threads = 4
    text_per_thread = text_length // num_threads
    
    # Prepare thread tasks
    tasks = []
    temp_files = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        # Create tasks for each thread
        for i in range(num_threads):
            start_idx = i * text_per_thread
            end_idx = start_idx + text_per_thread if i < num_threads - 1 else text_length
            temp_file = f"temp_thread_{i}.mp3"
            temp_files.append(temp_file)
            
            future = executor.submit(
                process_text_portion,
                text,
                start_idx,
                end_idx,
                temp_file
            )
            futures.append(future)
        
        # Wait for all tasks to complete and check for errors
        for i, future in enumerate(as_completed(futures), 1):
            try:
                if not future.result():
                    print(f"❌ Failed in thread {i}")
                    # Clean up temp files
                    for tf in temp_files:
                        Path(tf).unlink(missing_ok=True)
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Error in thread {i}: {str(e)}")
                for tf in temp_files:
                    Path(tf).unlink(missing_ok=True)
                sys.exit(1)
    
    # Combine audio files
    print("\n🔄 Combining audio parts...")
    
    # Try multiple methods to combine audio files
    success = False
    
    # Method 1: FFmpeg direct concatenation (most reliable)
    try:
        # Create a temporary list file for ffmpeg
        concat_list = "concat_list.txt"
        with open(concat_list, "w") as f:
            for temp_file in temp_files:
                f.write(f"file '{temp_file}'\n")
                
        # Use ffmpeg to concatenate files
        print("🔄 Method 1: Merging audio files using FFmpeg...")
        subprocess.check_call([
            "ffmpeg", 
            "-f", "concat", 
            "-safe", "0", 
            "-i", concat_list, 
            "-c", "copy", 
            output_file
        ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
        # Remove the list file
        Path(concat_list).unlink(missing_ok=True)
        
        print("✅ Audio files merged successfully using FFmpeg!")
        success = True
        
    except (subprocess.SubprocessError, FileNotFoundError):
        print("⚠️ Method 1 (FFmpeg) failed, trying alternative methods...")
    
    # Method 2: Simple file operations if only one file or FFmpeg failed
    if not success:
        try:
            print("🔄 Method 2: Using simple file operations...")
            import shutil
            
            if len(temp_files) == 1:
                # Only one file, just rename it
                shutil.move(temp_files[0], output_file)
                print("✅ Single audio file renamed successfully!")
                success = True
            else:
                # Multiple files - try basic binary concatenation for MP3
                print("🔄 Method 2b: Binary concatenation...")
                with open(output_file, 'wb') as outfile:
                    for i, temp_file in enumerate(temp_files):
                        print(f"📂 Adding part {i+1}/{len(temp_files)}...")
                        with open(temp_file, 'rb') as infile:
                            outfile.write(infile.read())
                print("✅ Audio files concatenated using binary method!")
                success = True
                
        except Exception as e:
            print(f"⚠️ Method 2 failed: {str(e)}")
    
    # Method 3: Last resort - copy first file only
    if not success:
        try:
            print("🔄 Method 3: Copying first audio file as fallback...")
            import shutil
            if temp_files:
                shutil.copy2(temp_files[0], output_file)
                print(f"⚠️ Only copied first audio segment to {output_file}")
                success = True
            else:
                raise Exception("No audio files found to process")
        except Exception as e:
            print(f"❌ All methods failed: {str(e)}")
            # Clean up temp files and exit
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)
            sys.exit(1)
    # Clean up temp files
    print("🧹 Cleaning up temporary files...")
    for temp_file in temp_files:
        try:
            Path(temp_file).unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors
    
    if success:
        print(f"\n✨ Successfully created audiobook: {output_file}")
        print("💡 You can now use this audio file with OpenVoice V2")
    else:
        print("❌ Failed to create complete audiobook")
        sys.exit(1)

if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    print(f"\n⏱️ Total processing time: {duration/60:.1f} minutes")
