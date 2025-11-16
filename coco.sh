#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=coco2014_download
#SBATCH --output=coco_download_%j.out
#SBATCH --error=coco_download_%j.err

# Load modules
module load cuda/12.8.0

# --- IMPORTANT: Correct conda initialization ---
source /apps/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate msr_sam

echo "=============================================================="
echo "DOWNLOADING COCO 2014 DATASET"
echo "=============================================================="
echo ""

# Create directory
mkdir -p ~/coco2014
cd ~/coco2014

echo "📁 Download location: ~/coco2014"
echo ""

# Function to download and extract
download_and_extract() {
    local filename=$1
    local url=$2
    local desc=$3
    
    echo "📥 Downloading $desc..."
    echo "   URL: $url"
    
    # Download with retry logic
    if wget --progress=bar:force:noscroll -O "$filename" "$url"; then
        echo "📦 Extracting $filename..."
        unzip -q "$filename"
        rm "$filename"
        echo "✓ $desc ready!"
    else
        echo "✗ Failed to download $desc"
        return 1
    fi
    return 0
}

# Download training images
if [ ! -d "train2014" ]; then
    download_and_extract "train2014.zip" "http://images.cocodataset.org/zips/train2014.zip" "training images (13GB)"
else
    echo "✓ train2014 already exists - skipping"
fi

echo ""

# Download validation images
if [ ! -d "val2014" ]; then
    download_and_extract "val2014.zip" "http://images.cocodataset.org/zips/val2014.zip" "validation images (6GB)"
else
    echo "✓ val2014 already exists - skipping"
fi

echo ""

# Download annotations - CORRECTED URL for COCO 2014
if [ ! -d "annotations" ]; then
    download_and_extract "annotations_trainval2014.zip" "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "annotations (433MB)"
else
    echo "✓ annotations already exists - skipping"
fi

echo ""

# Verification
echo ""
echo "=============================================================="
echo "DOWNLOAD COMPLETE - VERIFYING DATASET"
echo "=============================================================="

python3 << 'EOF'
import os
from pathlib import Path

dataset_dir = Path.home() / 'coco2014'

print("📊 COCO 2014 Dataset Summary:")
print(f"Location: {dataset_dir}")

# Count files
train_count = len(list((dataset_dir / 'train2014').glob('*.jpg'))) if (dataset_dir / 'train2014').exists() else 0
val_count = len(list((dataset_dir / 'val2014').glob('*.jpg'))) if (dataset_dir / 'val2014').exists() else 0
test_count = len(list((dataset_dir / 'test2014').glob('*.jpg'))) if (dataset_dir / 'test2014').exists() else 0

print(f"Training images: {train_count:,}")
print(f"Validation images: {val_count:,}")
print(f"Test images: {test_count:,}")

# Check annotations
annotations_dir = dataset_dir / 'annotations'
if annotations_dir.exists():
    annotation_files = list(annotations_dir.glob('*.json'))
    print(f"Annotation files: {len(annotation_files)}")
    for f in annotation_files:
        print(f"  - {f.name}")
else:
    print("Annotations: NOT FOUND")

# Expected counts (approximate)
expected_train = 82783
expected_val = 40504
expected_test = 40775

print("\n✅ Expected counts:")
print(f"Training images: ~{expected_train:,}")
print(f"Validation images: ~{expected_val:,}")
print(f"Test images: ~{expected_test:,}")

if train_count >= expected_train * 0.95:
    print("✓ Training set looks complete")
else:
    print("⚠️  Training set may be incomplete")

if val_count >= expected_val * 0.95:
    print("✓ Validation set looks complete")
else:
    print("⚠️  Validation set may be incomplete")
EOF

echo ""
echo "=============================================================="
echo "COCO 2014 DOWNLOAD JOB COMPLETED"
echo "=============================================================="
