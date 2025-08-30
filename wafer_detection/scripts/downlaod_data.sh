#!/bin/bash
# download_data.sh - Script to download wafer defect dataset
# Author: Sarthak Aggarwal
# Date: 30-08-2025

# Create data folder if not exists
mkdir -p data
cd data

echo "============================================"
echo "📦 Wafer Defect Dataset Downloader"
echo "============================================"
echo "Choose a source to download from:"
echo "1) Kaggle"
echo "2) Hugging Face"
echo "============================================"
read -p "Enter choice [1 or 2]: " choice

if [ "$choice" -eq 1 ]; then
    echo "⬇️ Downloading dataset from Kaggle..."
    # Make sure kaggle API is configured with kaggle.json
    kaggle datasets download -d <sarthakaggarwal3010/wafer-detection> -p ./ --unzip
    echo "✅ Dataset downloaded from Kaggle!"

elif [ "$choice" -eq 2 ]; then
    echo "⬇️ Downloading dataset from Hugging Face..."
    # Requires 'git' and 'git-lfs' installed
    git lfs install
    git clone https://huggingface.co/datasets/<Sarthak123Agg/wafer_detection> hf_dataset
    mv hf_dataset/* .
    rm -rf hf_dataset
    echo "✅ Dataset downloaded from Hugging Face!"

else
    echo "❌ Invalid choice. Exiting."
    exit 1
fi

cd ..
echo "============================================"
echo "✅ Dataset ready in ./data/"
echo "============================================"
