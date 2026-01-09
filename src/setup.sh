#!/bin/bash
set -euo pipefail

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found. Install it with: pip install gdown"
    exit 1
fi

# Ensure models directory exists
if [[ ! -e $DIR/models ]]; then
    mkdir "$DIR/models"
else
    echo "'models' directory already exists."
fi

# Ensure input_videos directory exists
if [[ ! -e $DIR/input_videos ]]; then
    mkdir "$DIR/input_videos"
else
    echo "'input_videos' directory already exists."
fi

# Download the models to models/ directory
echo "Downloading models..."
gdown -O "$DIR/models/ball_detection.pt" "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
gdown -O "$DIR/models/player_detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
gdown -O "$DIR/models/pitch_detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

# Download the sample videos to input_videos/ directory
echo "Downloading sample videos..."
gdown -O "$DIR/input_videos/0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
gdown -O "$DIR/input_videos/2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"
gdown -O "$DIR/input_videos/08fd33_0.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"
gdown -O "$DIR/input_videos/573e61_0.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"
gdown -O "$DIR/input_videos/121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"

echo "Setup complete!"
