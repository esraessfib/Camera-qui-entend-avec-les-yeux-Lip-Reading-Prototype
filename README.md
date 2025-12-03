# LipNet Project – Lip Reading Using Deep Learning

##  Project Overview

LipNet is a deep learning model designed to read lips from video sequences and predict the corresponding text. This implementation reproduces and simplifies the original **LipNet (Assael et al.)** architecture using PyTorch, with optimizations for training on consumer GPUs.

This repository provides:

* A complete training pipeline
* Preprocessing scripts for dataset preparation
* A custom LipNet architecture (CNN + BiGRU + CTC Loss)
* A real-time inference script
* Evaluation utilities

The project is suitable for university work, research projects, or learning sequence-to-sequence modeling for video.

---

##  Objectives

* Build a deep-learning-based lip-reading system
* Train the model on a preprocessed dataset of lip-movement video sequences
* Use **CTC Loss** to align video frames with character sequences
* Enable real-time prediction from webcam or test videos

---

##  Repository Structure

```
project/
│
├── |── blue.py           # Collect the videos from the dataset where the word 'blue' is pronounced
│   ├── train.py          # Main training file
│   ├── prep_blue.py      # Preprocessing 
│   ├── realtime.py       # Real-time lip reading from webcam
│
├── README.md
├── requirements.txt

```

---

##  Installation

### 1. Clone the repository

```
git clone https://github.com/esraessfib/Camera-qui-entend-avec-les-yeux-Lip-Reading-Prototype
cd Camera-qui-entend-avec-les-yeux-Lip-Reading-Prototype
```

### 2. Create a virtual environment

```
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

Make sure you have **PyTorch + CUDA** installed according to your GPU.

---

##  Dataset

The model expects a dataset composed of:

* Short video clips of people speaking
* Frames extracted (112×112 or 128×128)
* Text labels for each clip

Example dataset used: part of the  **GRID Corpus** dataset, but you may use your own dataset after preprocessing.

### Preprocessing Includes:

* Face detection
* Lip region extraction
* Frame resizing
* Normalization
* Saving sequences as tensors

##  Training the Model

Run the optimized training script:

```
python src/train.py
```

The script includes:

* Mixed precision training (FP16)
* CTC loss computation
* Data augmentation
* Checkpoint saving

Training logs show:

* Loss evolution
* WER/CER metrics
* Learning rate schedule

---

##  Testing and Inference

### Real-time lip reading (webcam)

```
python src/realtime.py
```

Outputs the predicted text in real time.

---

##  Results

Typical results on a cleaned dataset:

* **Character Error Rate (CER):** ~5–15%
* **Word Error Rate (WER):** ~15–30%

Accuracy depends heavily on:

* Dataset size
* Quality of lip cropping
* Speaker variability

---

##  Model Architecture

LipNet is composed of:

* **3D Convolutional layers** for spatiotemporal feature extraction
* **Bidirectional GRUs** for sequence modeling
* **CTC Loss** for alignment-free training

```
Input Frames → CNN3D → BiGRU × 2 → Linear → CTC Decoder
```

---

##  Technologies Used

* Python 3.10+
* PyTorch
* OpenCV
* NumPy / Pandas
* TQDM
* CUDA (recommended)

---

##  Author

Project created by **isra safi bouteraa** as part of a deep-learning project at ENSIT.

---

##  License

This project is released under the **MIT License**.

---

##  Contributing

Pull requests are welcome! If you want to improve model performance, feel free to open an issue.

---

##  Acknowledgments

Inspired by the original LipNet paper:

* *“LipNet: End-to-End Sentence-level Lipreading” – Assael et al.*

And various open-source lipreading implementations.

---

