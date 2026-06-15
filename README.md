# AgriVisonX: Deep Learning-Based Crop Disease Recognition

AgriVisonX is an advanced computer vision application designed to assist farmers, agronomists, and agricultural researchers in identifying and classifying plant diseases. Powered by a custom deep Convolutional Neural Network (CNN) built with Keras and TensorFlow, the application analyzes images of crop leaves and outputs precise diagnoses across 38 distinct crop-disease combinations.

The project features a sleek, interactive web interface built with Streamlit, enabling users to upload leaf photos and receive real-time, actionable predictions.

---

## 🌟 Key Features

* **38 Class Classification**: Supports identification of healthy leaves and common diseases for Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
* **State-of-the-Art Accuracy**: Achieves **~96.56% validation accuracy** and **~98.12% training accuracy** using deep feature extraction.
* **Interactive Web App**: A lightweight, easy-to-use Streamlit dashboard for real-time inference.
* **Robust CNN Architecture**: Implements a customized 10-layer Convolutional Neural Network optimized with regularization (Dropout) to prevent overfitting.
* **Pre-trained Weights Included**: Ships with ready-to-use weights saved in Keras format (`trained_plant_disease_model.keras`).

---

## 🧠 Model Architecture & Training

The core classifier is a deep Sequential CNN model designed specifically for plant pathology classification tasks:

* **Input Size**: `128 x 128 x 3` (RGB)
* **Convolutional Layers**:
  * 5 blocks of double-convolutional layers (`Conv2D` with ReLU activation)
  * Increasing filter sizes: `32 -> 64 -> 128 -> 256 -> 512`
  * Layer padding is balanced between `same` and `valid` to preserve spatial information while downsampling.
* **Pooling & Regularization**:
  * `MaxPooling2D` (`2x2` pool size, stride `2`) after each convolutional block.
  * Spatial Dropout (`0.25`) before flattening to reduce spatial correlation between features.
* **Fully Connected Network**:
  * Dense hidden layer with `1,500` units (ReLU activation)
  * Dropout (`0.40`) to prevent co-adaptation.
  * Softmax output layer with `38` units mapping to target classes.
* **Optimizer**: Adam with a custom learning rate of `1e-4` (`0.0001`).
* **Loss Function**: Categorical Crossentropy.

### 📈 Training History (10 Epochs)

Below is the validation and training performance logged during the model development process:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :---: | :-----------: | :---------------: | :-------------: | :-----------------: |
| **1** | 1.3959        | 58.80%            | 0.4835          | 85.04%              |
| **2** | 0.4442        | 85.94%            | 0.2758          | 91.00%              |
| **3** | 0.2665        | 91.41%            | 0.2617          | 91.62%              |
| **4** | 0.1811        | 94.08%            | 0.2042          | 93.61%              |
| **5** | 0.1366        | 95.56%            | 0.1818          | 94.30%              |
| **6** | 0.1049        | 96.59%            | 0.1811          | 94.92%              |
| **7** | 0.0900        | 97.06%            | 0.1432          | 95.86%              |
| **8** | 0.0775        | 97.45%            | 0.1875          | 94.78%              |
| **9** | 0.0615        | 97.97%            | 0.1662          | 95.05%              |
| **10**| 0.0579        | **98.13%**        | 0.1210          | **96.56%**          |

---

## 📂 Project Structure

```
├── main.py                          # Streamlit application entrypoint
├── Train_plant_disease.ipynb        # Jupyter notebook detailing dataset prep and model training
├── Test_plant_disease.ipynb         # Jupyter notebook detailing evaluation pipelines
├── config.json                      # Keras model layer configurations (JSON export)
├── training_hist.json               # Recorded metrics across training epochs
├── metadata.json                    # Metadata showing Keras version and model save date
├── requirements.txt                 # Python dependencies
├── trained_plant_disease_model.keras# Saved weights & architecture of the trained model (h5 model weights also present)
├── Diseases.png                     # Dashboard promotional illustration / display image
└── test/                            # A dataset directory containing 33 sample images for testing
```

---

## 🌿 Supported Classes (38 Crop-Disease Pairs)

The model evaluates leaves from **14 distinct crop species** and predicts one of the following states:

1. **Apple**: Scab, Black rot, Cedar apple rust, Healthy
2. **Blueberry**: Healthy
3. **Cherry**: Powdery mildew, Healthy
4. **Corn**: Cercospora leaf spot (Gray leaf spot), Common rust, Northern Leaf Blight, Healthy
5. **Grape**: Black rot, Esca (Black Measles), Leaf blight (Isariopsis Leaf Spot), Healthy
6. **Orange**: Huanglongbing (Citrus greening)
7. **Peach**: Bacterial spot, Healthy
8. **Pepper (Bell)**: Bacterial spot, Healthy
9. **Potato**: Early blight, Late blight, Healthy
10. **Raspberry**: Healthy
11. **Soybean**: Healthy
12. **Squash**: Powdery mildew
13. **Strawberry**: Leaf scorch, Healthy
14. **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (Two-spotted spider mite), Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### 1. Clone & Set Up Directory

```bash
git clone https://github.com/henry-n2/DroneProject.git
cd DroneProject
```

### 2. Install Dependencies

Install the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Note: If you run into issues installing `opencv-python-headless` on windows, you can install the standard `opencv-python` library instead.*

### 3. Run the Web Application

Launch the Streamlit dashboard:

```bash
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ How to Use the Application

1. **Home Page**: Displays the title and overview of the **Smart Disease Detection** system.
2. **Disease Recognition Page**:
   * Navigate using the sidebar.
   * Click **Choose an Image** and select a leaf image file (e.g., from the `test/` folder).
   * Click **Show Image** to preview.
   * Click **Predict** to run the image through the CNN classifier. The predicted crop/disease label will be displayed instantly.

---

## 🛠️ Tech Stack

* **Deep Learning Framework**: TensorFlow 2.x, Keras 3.x
* **Frontend**: Streamlit
* **Numerical & Data Processing**: NumPy, Pandas, Scikit-learn
* **Computer Vision & Image Loading**: Pillow (PIL), OpenCV
* **Data Visualization**: Matplotlib, Seaborn
