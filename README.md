# 🔍 Feature Detection & Matching in OpenCV
A complete benchmarking project demonstrating classic feature detection, description, and matching methods using OpenCV, including automatic visualization and PDF report generation. The project compares **SIFT, ORB, FAST, BRIEF, Harris**, and **Shi–Tomasi**, and generates match visualizations between two images.
## 🚀 Features
- Compare 6 classical feature detectors  
- Extract and visualize keypoints  
- Measure processing time for each algorithm  
- Automatic comparison grid & performance chart  
- Feature matching using **SIFT** & **ORB**  
- PDF report generation (images + charts + matches)  
- Clean and modular OpenCV functions  
## 📁 Included Script
| Script | Description |
|--------|-------------|
| **feature_detection.py** | Runs all detectors, matches features, saves charts & generates PDF. |
## 🔧 Installation
Install dependencies:  
`pip install opencv-python opencv-contrib-python numpy matplotlib reportlab`
## ▶️ How to Run
Run the main script:  
`python feature_detection.py`
Generates the following files automatically:  
- **comparison_grid.png** — side-by-side detector results  
- **performance.png** — keypoint count comparison  
- **sift_matches.png** — SIFT feature matches  
- **orb_matches.png** — ORB feature matches  
- **feature_report.pdf** — full summary report
## 🧠 Algorithms Compared
| Detector | Type | Notes |
|----------|------|--------|
| **SIFT** | Scale-Invariant Descriptor | Highly stable & accurate, slower |
| **ORB** | FAST + BRIEF | Very fast, good for real-time |
| **FAST** | Keypoint Detector | No descriptor; extremely fast |
| **BRIEF** | Descriptor | Requires STAR/FAST keypoints |
| **Harris** | Corner Detector | Classic, strong corners |
| **Shi–Tomasi** | Good Features to Track | Improved Harris |
## 💡 Enhancements
- Add SURF / AKAZE for more comparisons  
- Compute precision/recall of matches  
- Generate HTML dashboard instead of PDF  
- CLI flags for detector selection  
- Add RANSAC-based homography  
## 📜 License
Licensed under the **MIT License**.
