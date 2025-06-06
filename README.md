# Gold-Price-Forecasting-Using-Deep-Learning

# 🪙 Gold Price Forecasting Using RNN Models

## 📌 Overview

This project explores the use of **Recurrent Neural Network (RNN)** architectures—specifically **Simple RNN**, **LSTM**, and **GRU**—to forecast gold prices based on historical data. By leveraging deep learning, the project aims to improve the accuracy and robustness of gold price prediction compared to traditional time series models like ARIMA or GARCH.

---

## 📚 Abstract

Gold plays a vital role in the global financial ecosystem as both a hedge against economic instability and a core asset for central banks. Accurately forecasting gold prices is essential for investors, policymakers, and economists.

This study:

* Uses **RNN-based models** to capture long-term temporal dependencies in historical gold prices
* Compares RNN, LSTM, and GRU models using real-world financial data
* Evaluates models using key metrics such as **MSE**, **RMSE**, **MAE**, and **R²**

---

## 🎯 Objectives

* Analyze and preprocess 10+ years of gold price data (2013–2023)
* Develop and train RNN, LSTM, and GRU models using Keras and TensorFlow
* Compare model performance to identify the most effective architecture
* Explore the impact of hyperparameter tuning on model accuracy

---

## 📦 Dataset

* Source: [Investing.com - Gold Historical Data](https://www.investing.com)
* Period: Jan 2013 – Dec 2023
* Features:

  * `Date`, `Price`, `Open`, `High`, `Low`, `Volume`, `Change%`

---

## ⚙️ Methodology

### 🧹 Data Preprocessing

* Handling null/missing values
* Normalization using Min-Max or Z-score scaling
* Feature engineering and time windowing (30-day look-back)
* Conversion to 3D format for RNN input: `(samples, timesteps, features)`

### 🧠 Models Implemented

| Model    | Description                                                  |
| -------- | ------------------------------------------------------------ |
| **RNN**  | Baseline sequential model with SimpleRNN layers              |
| **LSTM** | Handles long-term dependencies and vanishing gradient issues |
| **GRU**  | Lightweight alternative to LSTM with competitive performance |

* All models use 2 stacked layers with dropout regularization
* Hyperparameters optimized using **Keras Tuner (Random Search)**

---

## 📈 Evaluation Metrics

* **MSE** (Mean Squared Error)
* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **R² Score**

### 📊 Example Results (LSTM Model)

```
MSE:   0.0006
RMSE:  0.0245
MAE:   0.0192
R²:    0.8334
```

> LSTM showed the best performance among the three models, followed by GRU and RNN.

---

## 🛠 Tools & Technologies

* Python
* Pandas, NumPy, Matplotlib
* TensorFlow & Keras
* Keras Tuner
* Google Colab / Jupyter Notebook

---

## 📂 Folder Structure (suggested)

```
.
├── data/
│   ├── Gold Price (2013–2022).csv
│   └── Gold Price (2023).csv
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── rnn_model.ipynb
│   ├── lstm_model.ipynb
│   └── gru_model.ipynb
├── models/
│   └── best_lstm_model.h5
├── results/
│   ├── plots/
│   └── metrics/
├── README.md
└── requirements.txt
```

## 📌 Conclusion

* Deep learning models, especially **LSTM**, provide promising results for gold price forecasting.
* With proper tuning and data preprocessing, RNN-based models can outperform traditional methods in volatility-heavy financial series.
* Future work can involve adding external macroeconomic features and using hybrid architectures like CNN-LSTM or Transformer-based models.

