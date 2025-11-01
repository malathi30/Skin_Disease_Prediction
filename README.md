# 🧠 Skin Disease Prediction using CNN + Random Forest

A hybrid AI-based project that predicts **skin diseases** from images using a combination of **Convolutional Neural Networks (CNN)** and a **Random Forest classifier**. The system provides accurate classification along with disease information and treatment suggestions, all accessible through a **Streamlit web app**.

---

## 🚀 Features

✅ Image-based skin disease prediction
✅ Hybrid model: CNN (feature extraction) + Random Forest (classification)
✅ User-friendly Streamlit interface
✅ Real-time image upload and prediction
✅ Displays disease details and treatment suggestions
✅ Lightweight and fast model inference

---

## 🧩 Technologies Used

* **Python 3.x**
* **TensorFlow / Keras** – Deep learning for feature extraction
* **Scikit-learn** – Random Forest classification
* **NumPy & Pandas** – Data preprocessing and analysis
* **Pillow** – Image handling
* **Joblib** – Model serialization
* **Streamlit** – Interactive web interface

---

## 📂 Project Structure

```
Skin_Disease_Prediction/
│
├── app.py                        # Main Streamlit application
├── skin_disease_model.pkl        # Trained hybrid model file
├── disease_info.csv              # Dataset with disease details
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
│
├── models/                       # Contains training scripts or saved models
├── data/                         # Raw and processed image data (if applicable)
└── assets/                       # UI images, icons, etc.
```

---

## ⚙️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/malathi30/Skin_Disease_Prediction.git
cd Skin_Disease_Prediction
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate     # For Windows
source venv/bin/activate  # For macOS/Linux
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit app

```bash
streamlit run app.py
```

Then open the provided local URL (usually `http://localhost:8501/`) in your browser.

---

## 🧠 Model Overview

* **CNN (Convolutional Neural Network):** Extracts key visual features from input skin images.
* **Random Forest Classifier:** Utilizes the extracted features to make the final disease prediction.
* This hybrid approach improves accuracy and reduces overfitting compared to standalone CNN models.

---

## 📊 Output Example

After uploading an image, the app displays:

* **Predicted Disease:** e.g., *Benign Keratosis*
* **Confidence Score:** Model’s confidence in the prediction
* **Disease Description & Treatment:** Pulled from the information dataset

---

## 💡 Future Enhancements

* Expand dataset with more disease classes
* Integrate real-time camera input
* Deploy the model on cloud (e.g., AWS / Streamlit Cloud)
* Add multilingual voice assistance for accessibility

---

