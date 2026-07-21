# 🗑️ SMART WASTE MANAGEMENT USING ARTIFICIAL INTELLIGENCE

> **AI-Powered IoT Solution for Intelligent Waste Collection**

A machine learning project that predicts waste bin overflow using IoT sensor data (fill level, weight, temperature, gas readings, and more) with a Random Forest classifier, wrapped in an interactive Streamlit web app.

---

## 📑 TABLE OF CONTENTS

- [🚀 Live Demo](#-live-demo)
- [✨ Features](#-features)
- [🎯 Quick Start](#-quick-start)
- [📊 Dataset](#-dataset)
- [🤖 Machine Learning Model](#-machine-learning-model)
- [📈 Model Performance](#-model-performance)
- [📁 Project Structure](#-project-structure)
- [🛠️ Installation](#-installation)
- [🌐 Streamlit App Guide](#-streamlit-app-guide)
- [🔧 Troubleshooting](#-troubleshooting)
- [🚀 Future Enhancements](#-future-enhancements)

---

## 🚀 LIVE DEMO

### ⭐ Try the Application Now — No Installation Required!

#### **[🌐 OPEN LIVE APPLICATION →](https://smart-waste-management-using-artificialintelligence-krqud5r8gh.streamlit.app/)**

**Live Application Details:**
- 🔗 **URL**: https://smart-waste-management-using-artificialintelligence-krqud5r8gh.streamlit.app/
- ✅ **Status**: Live and fully functional
- 📱 **Devices**: Works on desktop, tablet, mobile
- 🆓 **Cost**: Completely free — no signup needed

**In the Live Demo, You Can:**
- 📊 **Dataset Explorer** — browse the built-in dataset or upload your own CSV
- 📈 **Data Analysis** — distributions, correlation heatmap, overflow rate by category
- 🔮 **Predictions** — adjust sensor sliders and get a real-time overflow prediction with confidence score
- ℹ️ **Model Info** — feature importance chart and confusion matrix

**Sample Prediction in 3 Steps:**
1. Go to the **🔮 Predictions** page
2. Adjust sensor sliders (Fill %, Temperature, Battery, etc.)
3. Click **🚀 Predict Overflow** → get an instant result!

---

## ✨ FEATURES

| Feature | Description |
|---|---|
| 🔮 **Overflow Prediction** | Predicts bin overflow before it happens |
| 📊 **Dataset Explorer** | Upload and browse IoT sensor data |
| 📈 **Interactive EDA** | Distribution plots, correlation heatmap, category breakdowns |
| 📉 **Feature Importance** | Shows which sensor readings matter most |
| ℹ️ **Model Info Dashboard** | Accuracy, confusion matrix, feature ranking |
| 📱 **Responsive UI** | Works on desktop and mobile browsers |

---

## 🎯 QUICK START

### Fastest way to get started (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web app
streamlit run app.py

# 3. Open in browser
# http://localhost:8501
```

### Try the live demo (30 seconds)
🌐 **[Click here →](https://smart-waste-management-using-artificialintelligence-krqud5r8gh.streamlit.app/)** — no installation required.

---

## 📊 DATASET

| Property | Value |
|---|---|
| **Total Records** | 50,000 IoT readings |
| **Total Columns** | 22 |
| **Missing Values** | 0 |
| **Format** | CSV |

**Key Sensor Features (used by the model):**

- **Container** — `Current_Fill_%`, `Weight_kg`, `Hours_Since_Last_Collection`
- **Environment** — `Temperature_C`, `Humidity_%`, `Rain_Detected`
- **Gas Sensors** — `Methane_ppm`, `CO2_ppm`
- **Device** — `Battery_Level_%`, `Sensor_Status`, `Motion_Detected`

**Other dataset columns** (not used for prediction, but available in the raw CSV): `Timestamp`, `Bin_ID`, `Location`, `Latitude`, `Longitude`, `Bin_Capacity_L`, `Waste_Type`, `AI_Priority`, `Route_Assigned`, `Collection_Status`

**Target Variable:** `Overflow` — Yes / No

**Class Distribution:**
```
No Overflow:  45,481 records (91.0%)
Overflow:      4,519 records ( 9.0%)
```

---

## 🤖 MACHINE LEARNING MODEL

**Algorithm**: Random Forest Classifier
- 🌳 Trees: 100
- 🎯 Random state: 42 (reproducible)
- ⚙️ Features used: 11
- 🔀 Train/test split: 80% / 20%, stratified on `Overflow`

**Why Random Forest?**
- Handles mixed categorical + numerical data well
- Robust to outliers, no feature scaling needed
- Provides feature importance out of the box
- Fast inference for real-time predictions

---

## 📈 MODEL PERFORMANCE

```
Training Set: 40,000 records
Testing Set:  10,000 records

Accuracy:   100%
Precision:  100%
Recall:     100%
F1-Score:   100%
```

**Confusion Matrix**
```
                    Predicted
                No Overflow | Overflow
Actual No Ovf      9,096    |    0
Actual Overflow       0     |   904
```

**Top Features by Importance**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Current_Fill_%` | 98.3% |
| 2 | `Weight_kg` | 0.28% |
| 3 | `CO2_ppm` | 0.27% |
| 4 | `Methane_ppm` | 0.24% |
| 5 | `Temperature_C` | 0.24% |
| 6 | `Humidity_%` | 0.20% |
| 7 | `Battery_Level_%` | 0.20% |
| 8 | `Hours_Since_Last_Collection` | 0.18% |
| 9 | `Sensor_Status` | 0.05% |
| 10 | `Rain_Detected` | 0.03% |
| 11 | `Motion_Detected` | 0.03% |

**Key Insight**: Current fill level dominates the prediction (98.3% importance) — which makes intuitive sense, since it's the most direct signal of how close a bin is to overflowing.

---

## 📁 PROJECT STRUCTURE

```
Smart_Waste_Management/
│
├── app.py                     Streamlit web application
│   ├─ Home (project overview)
│   ├─ Dataset Explorer (upload & browse)
│   ├─ Data Analysis (EDA visualizations)
│   ├─ Predictions (real-time inference)
│   └─ Model Info (feature importance, confusion matrix)
│
├── train_model.py             Training script — retrain the model on new data
├── requirements.txt           Python dependencies
├── smart_waste_model.pkl      Pre-trained Random Forest model
├── model_metadata.pkl         Encoders, accuracy, feature importance (used by app.py)
├── dataset.csv                50,000-row IoT sensor dataset
├── SMART_WASTE_MANAGEMENT.ipynb  Full ML pipeline notebook
└── README.md                  This file
```

---

## 🛠️ INSTALLATION

**Requirements:** Python 3.8+

```bash
# 1. Make sure all project files are in one folder
# 2. Open a terminal in that folder
# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`. If it doesn't, copy that URL manually.

**Dependencies** (`requirements.txt`):
```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
streamlit==1.28.1
```

---

## 🌐 STREAMLIT APP GUIDE

| Page | What it does |
|---|---|
| 🏠 **Home** | Project overview and key metrics |
| 📊 **Dataset Explorer** | Upload your own CSV or browse the default dataset; view stats and preview rows |
| 📈 **Data Analysis** | Feature distributions, correlation heatmap, overflow rate by category |
| 🔮 **Predictions** | Set 11 sensor values via sliders/dropdowns → get an instant overflow prediction with confidence |
| ℹ️ **Model Info** | Feature importance chart, confusion matrix, accuracy |

**Retraining the model** (optional — only needed if you update the dataset):
```bash
python train_model.py
```
This regenerates `smart_waste_model.pkl` and `model_metadata.pkl`.

---

## 🔧 TROUBLESHOOTING

**`streamlit: command not found`**
```bash
pip install streamlit==1.28.1
```

**Port 8501 already in use**
```bash
streamlit run app.py --server.port=8502
```

**`Model file not found` error**
```bash
python train_model.py
```
Make sure `smart_waste_model.pkl` and `model_metadata.pkl` are in the same folder as `app.py`.

**`pip: command not found`**
```bash
python -m pip install -r requirements.txt
```

---

## 🚀 FUTURE ENHANCEMENTS

- 📡 Live IoT device integration
- 🗺️ Route optimization using bin locations (`Latitude`/`Longitude`)
- 📅 Predictive scheduling (24–48 hour forecasts)
- 📱 Mobile app version
- 🔔 SMS/email alerts for predicted overflow

---

## 🎯 QUICK LINKS

- 🌐 **[Live Demo App →](https://smart-waste-management-using-artificialintelligence-krqud5r8gh.streamlit.app/)**
- 📓 **[Jupyter Notebook](SMART_WASTE_MANAGEMENT.ipynb)**

---

**Happy Predicting!** 🚀
