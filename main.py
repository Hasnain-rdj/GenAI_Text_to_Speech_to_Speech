"""
PDF Text Extractor
Extracts first 100k characters from a PDF file
"""
try:
    import PyPDF2
except ImportError:
    import subprocess
    print("📦 Installing PyPDF2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

from pathlib import Path
import re
import sys

def clean_text(text):
    """Clean extracted text by removing extra whitespace and unwanted characters"""
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def extract_text_from_pdf(pdf_path, max_chars=100_000):
    """Extract and clean text from PDF file"""
    try:
        print(f"📖 Reading PDF: {pdf_path}")
        
        # Open PDF file
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get total pages
            total_pages = len(pdf_reader.pages)
            print(f"📑 Total pages: {total_pages}")
            
            # Extract text
            extracted_text = ""
            chars_extracted = 0
            
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
                
                # Show progress
                progress = min(100, (chars_extracted / max_chars) * 100)
                print(f"📊 Progress: {progress:.1f}% ({chars_extracted}/{max_chars} characters)", end='\r')
            
            print("\n✅ Text extraction complete!")
            return extracted_text[:max_chars]
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def save_text_to_file(text, output_path):
    """Save extracted text to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"💾 Text saved to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}")
        return False

def main():
    # Input and output files
    pdf_file = "Life_3.0.pdf"
    output_file = "extracted_text.txt"
    
    # Check if PDF exists
    if not Path(pdf_file).exists():
        print(f"❌ PDF file not found: {pdf_file}")
        sys.exit(1)
    
    # Extract text
    extracted_text = extract_text_from_pdf(pdf_file)
    if not extracted_text:
        print("❌ Text extraction failed")
        sys.exit(1)
    
    # Save to file
    if save_text_to_file(extracted_text, output_file):
        print(f"✨ Successfully extracted {len(extracted_text)} characters")
        print(f"📝 Preview:\n{extracted_text[:200]}...")
    else:
        print("❌ Failed to save extracted text")
        sys.exit(1)

if __name__ == "__main__":
    main()
