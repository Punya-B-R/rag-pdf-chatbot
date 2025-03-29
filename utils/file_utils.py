import os
from pathlib import Path

def save_uploaded_file(uploaded_file, upload_dir="uploads"):
    """Save file with automatic cleanup registration"""
    os.makedirs(upload_dir, exist_ok=True)
    file_path = Path(upload_dir) / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)