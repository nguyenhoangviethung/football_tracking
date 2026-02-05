Dưới đây là mẫu **README.md** chuyên nghiệp, được viết bằng tiếng Anh (chuẩn cho GitHub và CV quốc tế), dựa trên toàn bộ thông tin kỹ thuật trong báo cáo của bạn.

Bạn hãy copy nội dung này vào file `README.md` trong thư mục code của bạn.

---

# Football Analysis & Tracking System (AI-Powered)

## 📖 Introduction

This project focuses on building an automated system to analyze football broadcast footage using advanced Computer Vision and Deep Learning techniques. The system detects players, referees, and the ball, tracks their movements, classifies teams without prior knowledge (unsupervised), and projects gameplay onto a 2D tactical map for analytics.

Developed as part of **Project 3** at **Hanoi University of Science and Technology (HUST) - SOICT**.

## ✨ Key Features

* 
**Robust Player Tracking:** Utilizes **ByteTrack** to handle occlusion and maintain consistent IDs even when players cross paths or are partially hidden.


* 
**Small Object Detection (The Ball):** Implements **SAHI (Slicing Aided Hyper Inference)** to detect the ball (small object) by slicing frames, combined with trajectory-based filtering to remove false positives.


* 
**Unsupervised Team Classification:** Uses **SigLIP** for semantic embedding extraction, followed by **UMAP** dimensionality reduction and **K-Means** clustering to distinguish teams based on jersey features, robust against lighting changes.


* 
**Tactical 2D Mapping (Hologram View):** Maps player coordinates from camera view to a 2D pitch using **Homography** transformation and Keypoint Detection, enabling Voronoi diagrams and Heatmap visualization.


* 
**Performance Optimization:** Multithreaded architecture (Producer-Consumer pattern) to parallelize video reading, inference, and writing.



## 🛠️ Tech Stack & Methodology

### 1. Object Detection & Tracking

* **Players:** YOLOv8 (fine-tuned) + ByteTrack. ByteTrack is chosen over DeepSORT because it utilizes low-confidence detections to recover occluded objects.


* 
**Ball:** Custom model trained on `football-ball-detection` dataset + SAHI for high recall + Custom Deque Buffer for trajectory smoothing.



### 2. Team Assignment Pipeline

Instead of simple color histograms, we use a semantic approach:

1. **Crop** player images.
2. 
**Extract Features** using **SigLIP** (Google's Sigmoid Loss for Language Image Pre-Training).


3. 
**Reduce Dimensions** from 768D to 3D using **UMAP**.


4. 
**Cluster** using **K-Means** ().


5. 
**Goalkeeper Handling:** Spatial heuristic logic (assigns GK to the team with the closest centroid).



### 3. Tactical Projection

* **Keypoints Detection:** Identifies critical pitch landmarks (corners, center circle).
* 
**Homography:** Calculates the transformation matrix () using RANSAC to map pixel coordinates  to real-world metric coordinates .



## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/nguyenhoangviethung/football_tracking.git
cd football_tracking

```


2. **Install dependencies**
```bash
pip install -r requirements.txt

```



*Key requirements:* `ultralytics`, `supervision`, `roboflow`, `opencv-python`, `umap-learn`, `scikit-learn`, `torch`.


3. **Download Models**
* Place your trained YOLO weights (`best.pt`) in the `models/` directory.



## 💻 Usage

Run the main processing script on a video file:

```bash
python main.py --source_video "path/to/input.mp4" --target_video "path/to/output.mp4"

```

*Note: The system requires a GPU (e.g., NVIDIA T4 on Colab) for reasonable inference speed due to the complexity of SigLIP and SAHI.* 

## 📊 Results

### Tracking & Team Classification

*Handling occlusion and assigning team colors correctly.*

### Ball Tracking with SAHI

*Detecting the ball despite motion blur and small size.*

### 2D Map Projection (Voronoi & Heatmap)

*Real-time tactical analysis on a 2D pitch.*

## 🚧 Limitations & Future Work

* **Real-time constraints:** SAHI increases inference time significantly. Future work involves optimizing model export (TensorRT).


* 
**ID Switching:** Can still occur in extreme crowd scenarios (e.g., goal celebrations).


* 
**Event Spotting:** Plan to integrate LSTM/Transformers to detect events like passes, shots, and goals automatically.



## 🤝 Acknowledgements

* **Supervisors:** ThS. Lê Đức Trung.


* **References:**
* ByteTrack: Multi-Object Tracking by Associating Every Detection Box.


* SigLIP: Sigmoid Loss for Language Image Pre-Training.


* Roboflow Supervision.

