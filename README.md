# 🌞 Solar Power Generation Forecasting using Artificial Neural Networks (ANN)

## 📌 Project Overview
This project focuses on predicting future solar power generation using **Artificial Neural Networks (ANN)** trained on historical solar and weather data. Accurate solar power forecasting helps energy planners anticipate supply variations and efficiently allocate electrical resources across cities, ensuring grid stability and sustainable energy management.

The system integrates data preprocessing, statistical analysis, ANN-based prediction, and visualization into a unified workflow.

---

## 🎯 Objectives
- Predict short-term solar power generation using historical data  
- Analyze the influence of environmental parameters on power output  
- Support smart energy allocation for cities  
- Improve grid stability and renewable energy utilization  

---

## 🧠 Technologies Used
- **Python**
- **Artificial Neural Networks (ANN)**
- **Pandas & NumPy** – Data processing
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn / TensorFlow / Keras** – Model training
- **Streamlit** – Interactive visualization interface

---

## 📂 Dataset Description
The project uses structured tabular datasets containing meteorological and solar generation data.

### Files Used
- `solarpowergeneration.csv` – Hourly solar and weather data  
- `Solar per day con..xlsx` – Daily consolidated records  

### Key Features
- Temperature (°C)  
- Relative Humidity (%)  
- Wind Speed (m/s)  
- Solar Irradiance (W/m²)  
- Cloud Cover  
- Angle of Incidence  
- Generated Power (kW)  

### Sample Dataset (Shortened View)

| Temperature (°C) | Humidity (%) | Wind Speed (m/s) | Irradiance (W/m²) | Power (kW) |
|-----------------|--------------|------------------|------------------|------------|
| 2.17 | 31 | 6.37 | 0.00 | 454.10 |
| 2.31 | 27 | 5.15 | 1.78 | 1412.00 |
| 3.65 | 33 | 4.68 | 108.58 | 2214.85 |
| 5.82 | 30 | 3.60 | 258.10 | 2527.61 |

📎 **Dataset Source:**  
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

---

## 🌐 Live Deployment

[🚀 Solar Forecasting Intelligence Studio](https://solar-power-generation-forecasting-1.streamlit.app/)

---

## 👨‍💻 Author

Developed by:

**Dhavala V D M Adithya Naidu**

## ⚠️ Important Note (Before Running the Code)
Before running any **Jupyter Notebook (`.ipynb`) files**, ensure that you **update the dataset file paths** according to where the dataset files are stored on your local system.

Failure to update the file paths may result in *FileNotFoundError*.

**Example:**
```python
data = pd.read_csv("C:/Users/YourName/Documents/solarpowergeneration.csv")


---
