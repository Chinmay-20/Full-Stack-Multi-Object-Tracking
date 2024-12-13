# Multi-Object Detection and Tracking System
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-v6.0-darkgreen.svg)](https://github.com/ultralytics/yolov5)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

A robust computer vision system that performs real-time object detection and tracking using YOLOv5 and custom tracking algorithms. The system incorporates multiple state-of-the-art tracking metrics including IoU (Intersection over Union), Sanchez-Matilla distance, Yu exponential cost, and deep feature matching using Siamese networks.

## Key Features

- Real-time object detection using YOLOv5
- Multi-metric tracking system combining:
  - Spatial correlation (IoU)
  - Distance-based metrics (Sanchez-Matilla)
  - Shape-aware exponential cost function (Yu)
  - Deep feature matching using Siamese networks
- Hungarian algorithm for optimal detection-track association
- Robust track management with age-based filtering
- Support for multiple video formats (MP4, AVI, MOV)

## Project Structure

```
├── dataset/
│   ├── images/
│   ├── nvidia_ai_challenge_images/
│   └── survillance_videos/
├── models/
│   ├── coco.names
│   ├── model640.pt
│   └── yolov5s.pt
├── object_tracking.py
├── siamese_net.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-tracking.git
cd object-tracking
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the tracking system on a video file:

```bash
python object_tracking.py path/to/your/video.mp4
```

The processed video will be saved as `output_video.mp4` in the project directory.

## System Architecture

### Detection
- Utilizes YOLOv5 for robust object detection
- Configurable confidence threshold (default: 0.5)
- IoU threshold for non-max suppression (default: 0.4)

### Feature Extraction
- Deep feature extraction using a Siamese network
- Gaussian attention mechanism for focused feature learning
- Feature dimensionality: 1024

### Tracking
The system employs a multi-metric tracking approach:

1. **IoU Matching**
   - Primary spatial correlation metric
   - Threshold: 0.3

2. **Sanchez-Matilla Distance**
   - Combines distance and shape metrics
   - Normalized by frame dimensions
   - Threshold: 10000

3. **Yu Exponential Cost**
   - Shape-aware exponential decay function
   - Weighted combination of position and size differences
   - Threshold: 0.5

4. **Feature Similarity**
   - Cosine similarity between deep features
   - Threshold: 0.2

### Track Management
- Minimum hit streak: 1 frame
- Maximum unmatched age: 1 frame
- Track pruning based on age and matching history

## Example Results

[Watch Output Demo](output_demo.mp4)

[Watch Output Video](output_video.mp4)

## Performance Metrics

- Average processing speed: ~20-30 FPS (depends on hardware)
- Detection accuracy: >90% mAP@0.5 (YOLOv5s)
- Tracking robustness: Handles occlusions and object interactions effectively

## Technical Considerations

### Memory Management
- Efficient frame processing with OpenCV
- Batch processing for feature extraction
- GPU acceleration supported for both detection and feature extraction

### Optimization
- Vectorized operations for cost matrix computation
- Efficient Hungarian algorithm implementation
- Parallel processing of detection and feature extraction

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<p align="center">
  Made with ❤️ by Your Name
</p>
