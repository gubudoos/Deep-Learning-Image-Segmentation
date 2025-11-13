import os
import subprocess
import zipfile
from pathlib import Path
import time

print("=" * 70)
print("DOWNLOADING FULL COCO 2017 DATASET")
print("=" * 70)
print("⏰ Estimated time: 30-60 minutes")
print("💾 Total size: ~19GB")
print("")

start_time = time.time()

# Setup
HOME = Path.home()
dataset_dir = HOME / 'coco2017'
dataset_dir.mkdir(exist_ok=True)
os.chdir(dataset_dir)

print(f"📁 Download location: {dataset_dir}\n")

# Download configuration
downloads = [
    ('train2017.zip', 'http://images.cocodataset.org/zips/train2017.zip', '18GB', 'Training images'),
    ('val2017.zip', 'http://images.cocodataset.org/zips/val2017.zip', '1GB', 'Validation images'),
    ('annotations_trainval2017.zip', 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip', '241MB', 'Annotations')
]

for filename, url, size, desc in downloads:
    folder = filename.replace('.zip', '')
    
    if os.path.exists(folder):
        print(f"✓ {desc} already exists - skipping")
        continue
    
    print(f"\n📥 Downloading {desc} ({size})...")
    print(f"   URL: {url}")
    
    # Download with progress bar
    result = subprocess.run(
        ['wget', '--progress=bar:force:noscroll', '-O', filename, url],
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"\n📦 Extracting {filename}...")
        
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            # Get total files for progress
            total_files = len(zip_ref.namelist())
            print(f"   Extracting {total_files} files...")
            
            for i, member in enumerate(zip_ref.namelist()):
                zip_ref.extract(member)
                if (i + 1) % 5000 == 0:
                    print(f"   Progress: {i+1}/{total_files} files")
        
        os.remove(filename)
        print(f"✓ {desc} ready!")
    else:
        print(f"✗ Failed to download {desc}")

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("✅ DATASET DOWNLOAD COMPLETE!")
print("=" * 70)
print(f"⏱️  Total time: {elapsed/60:.1f} minutes")

# Verify
train_count = len(list((dataset_dir / 'train2017').glob('*.jpg'))) if (dataset_dir / 'train2017').exists() else 0
val_count = len(list((dataset_dir / 'val2017').glob('*.jpg'))) if (dataset_dir / 'val2017').exists() else 0

print(f"\n📊 Dataset Summary:")
print(f"   Training images: {train_count:,}")
print(f"   Validation images: {val_count:,}")
print(f"   Annotations: {len(list((dataset_dir / 'annotations').glob('*.json')))}")
print(f"   Location: {dataset_dir}")