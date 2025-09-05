# Self-Driving Car – Behavioral Cloning 🚗

This project implements a deep learning model that learns to steer a car by observing human driving behavior. The model processes images from a front-facing camera and predicts steering angles, effectively cloning human driving behavior.

---

## ✨ Key Features

- **Data Augmentation**: Robust techniques for better generalization
- **CNN Architecture**: Custom network inspired by NVIDIA's self-driving car model
- **Real-time Prediction**: Socket.IO server for live steering angle predictions
- **Data Balancing**: Prevents model bias toward straight driving
- **Preprocessing Pipeline**: Optimized image preprocessing for better model performance

---

## 🎥 Demo & Presentation

- **[Video Demonstration](link-to-video)**: Watch the model in action
- **[Project Presentation](link-to-slides)**: View the PowerPoint slides

---

## 🏗️ Model Architecture

The network is based on NVIDIA's self-driving car architecture with modifications:

```
Input: (66, 200, 3) preprocessed image
    ↓
Conv2D (24 filters, 5x5, strides=2x2) + ELU
    ↓
Conv2D (36 filters, 5x5, strides=2x2) + ELU
    ↓
Conv2D (48 filters, 5x5, strides=2x2) + ELU
    ↓
Conv2D (64 filters, 3x3) + ELU
    ↓
Conv2D (64 filters, 3x3) + ELU
    ↓
Flatten
    ↓
Dense (100 neurons) + ELU
    ↓
Dense (50 neurons) + ELU
    ↓
Dense (10 neurons) + ELU
    ↓
Output: 1 neuron (steering angle)
```

---

## ⚙️ Installation & Usage

### ✅ Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Flask, SocketIO, Eventlet
- Pandas, scikit-learn, imgaug

### 🔽 Installation

```bash
git clone https://github.com/your-username/self-driving-car.git
cd self-driving-car
pip install tensorflow opencv-python socketio eventlet flask pandas scikit-learn imgaug
```

### 🚗 Training the Model

1. Place your training data inside the `data/` folder containing:
   - `IMG/` directory (images)
   - `driving_log.csv`

2. Run the training script:
   ```bash
   python train.py
   ```

3. The trained model will be saved as `model.h5`

### 🔌 Running the Prediction Server

Start the server for real-time steering predictions:

```bash
python test.py
```

- Runs on port 4567
- Waits for connections from the driving simulator

---

## 🔧 Technical Details

### 🖼️ Data Preprocessing

- Crop images to focus on the road area
- Convert RGB → YUV color space
- Apply Gaussian blur for noise reduction
- Resize images to 200×66 pixels
- Normalize pixel values (scale 0–1)

### 🧪 Data Augmentation

- Random panning and shifting
- Zoom variations
- Brightness adjustments
- Horizontal flipping (with steering angle correction)
- Random blurring
- Rotation transformations

### 📚 Training Approach

- Balanced dataset to avoid steering bias
- 80/20 train-validation split
- Batch generator for efficient training
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning rate = 1e-6)

---

## 📊 Results

- ✅ Low validation loss → strong generalization
- ✅ Smooth steering predictions under varied conditions
- ✅ Stable performance due to diverse augmentations

---

## 📁 Project Structure

```
self-driving-car/
│
├── data/
│   ├── IMG/                 # Training images
│   └── driving_log.csv      # Steering angles and image paths
│
├── train.py                 # Model training script
├── test.py                  # Prediction server script
├── model.h5                 # Trained model (generated)
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**
3. **Prepare your training data**
4. **Train the model**
5. **Run the prediction server**
6. **Connect your driving simulator**

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NVIDIA's End-to-End Deep Learning for Self-Driving Cars paper
- Udacity Self-Driving Car Nanodegree program
- The autonomous driving community
