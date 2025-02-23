# Whiff Distance Analysis

A computer vision pipeline for analyzing baseball swings and calculating whiff distance metrics. This project uses advanced computer vision and machine learning techniques to track bat movement and analyze swing mechanics.

## Features

- Bat tracking and trajectory analysis
- Swing plane calculation
- Contact point detection
- Swing metrics computation (speed, angle, duration)
- Game context analysis using Florence2 model
- Robust error handling and retry mechanisms

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- YOLO (for bat detection)
- Florence2 (for game context analysis)
- BaseballCV package

## Installation

1. Clone the repository:
```bash
git clone https://github.com/joeysnclr/whiff-distance.git
cd whiff-distance
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from bat_tracking_pipeline import analyze_baseball_swing

# Analyze a single swing video
results = analyze_baseball_swing("path/to/swing_video.mp4")

# Access analysis results
print(f"Sequence length: {results['sequence_length']} frames")
print(f"Swing metrics: {results['swing_metrics']}")
```

## Project Structure

```
whiff-distance/
├── bat_tracking_pipeline.py   # Main pipeline implementation
├── requirements.txt          # Project dependencies
├── models/                   # Pre-trained models
├── data/                    # Sample data and datasets
└── output/                  # Analysis outputs
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BaseballCV library for core functionality
- Florence2 model for game context analysis
- YOLO for object detection 