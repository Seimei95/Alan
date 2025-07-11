# download_data.py
import gdown
import zipfile
import os

# Replace this with your actual Google Drive file ID
file_id = "1vkMdtp5NnTO4ZdobR98Nb75RB6OsdwgL"
url = f"https://drive.google.com/file/d/1vkMdtp5NnTO4ZdobR98Nb75RB6OsdwgL/view?usp=sharing"
output = "MICC-F220.zip"

print("📥 Downloading MICC-F220 dataset...")
gdown.download(url, output, quiet=False)

print("📂 Extracting to project root...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(".")

print("✅ Dataset ready at ./MICC-F220/")
