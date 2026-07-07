# OracleXI – AI Sports Forecasting System

> An intelligent Python desktop application that forecasts **Football** and **Cricket** match outcomes using Machine Learning, Statistical Analysis, and Monte Carlo Simulation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Library Usage](#library-usage)
- [Algorithms](#algorithms)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Educational Outcomes](#educational-outcomes)

---

## 🎯 Overview

OracleXI is a complete Python desktop application built using **Tkinter** for the GUI, backed by a powerful prediction engine that combines:

- **Pandas** for data cleaning, preprocessing, and feature engineering
- **NumPy** for numerical computations, Poisson distribution, and Monte Carlo simulation
- **Statsmodels** for time-series analysis (ARIMA), regression, trend detection, and confidence intervals
- **TensorFlow** for neural network training and prediction
- **Matplotlib** for interactive chart visualization embedded inside the application

The application uses an MVC (Model-View-Controller) architecture with proper modular design.

---

## ✨ Features

### Core Capabilities
- ⚽ **Football Predictions** – Forecast international football match outcomes
- 🏏 **Cricket Predictions** – Predict IPL cricket match results
- 📊 **Analytics Dashboard** – Interactive charts with team performance analysis
- 📋 **Prediction History** – Search, filter, and export past predictions

### Technical Features
- 🧠 **Neural Network** – Lightweight TensorFlow Sequential model for ML prediction
- 📈 **ARIMA Forecasting** – Time-series analysis for team form prediction
- 🎲 **Monte Carlo Simulation** – 10,000+ Poisson-distributed match simulations
- 🔄 **Weighted Ensemble** – Combines Statsmodels (30%) + TensorFlow (40%) + Monte Carlo (30%)
- 💾 **Model Persistence** – Train once, auto-load saved models
- 📥 **CSV Export** – Export predictions and processed data

### GUI Features
- 🌙 **Dark Theme** – Premium dark UI with gradient accents
- 📊 **Embedded Charts** – 6 Matplotlib charts per sport
- ⏳ **Loading Animation** – Spinner during prediction computation
- 🔍 **Search & Filter** – Find predictions by team or sport
- 📱 **Responsive Layout** – Scrollable content with proper resizing

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

1. **Clone/Download the project:**
   ```bash
   cd OracleXI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate datasets:**
   ```bash
   python generate_datasets.py
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

### Requirements
```
numpy>=1.24.0
pandas>=2.0.0
tensorflow>=2.13.0
statsmodels>=0.14.0
matplotlib>=3.7.0
```

---

## 📁 Folder Structure

```
OracleXI/
│
├── data/                           # Generated CSV datasets
│   ├── football_results.csv        # ~2,000 football match records
│   ├── cricket_match_info.csv      # ~800 cricket match records
│   └── cricket_ball_by_ball.csv    # ~80,000 ball-by-ball records
│
├── core/                           # Core prediction engine (Model)
│   ├── __init__.py
│   ├── data_engine.py              # Pandas data loading & querying
│   ├── preprocessing.py            # Data cleaning & feature engineering
│   ├── statistics.py               # Statsmodels analysis & forecasting
│   ├── tensorflow_model.py         # TensorFlow neural network
│   ├── simulation.py               # NumPy Monte Carlo simulation
│   └── prediction_engine.py        # Orchestrator (Controller)
│
├── gui/                            # Tkinter GUI (View)
│   ├── __init__.py
│   ├── widgets.py                  # Custom widget library
│   ├── home.py                     # Home page
│   ├── prediction.py               # Prediction page
│   ├── analytics.py                # Analytics dashboard
│   └── history.py                  # Prediction history
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── constants.py                # Configuration & constants
│   └── helper.py                   # Logging, I/O, validation
│
├── models/                         # Saved TensorFlow models (auto-generated)
├── history/                        # Prediction history JSON (auto-generated)
├── exports/                        # CSV exports (auto-generated)
│
├── main.py                         # Application entry point
├── generate_datasets.py            # Synthetic dataset generator
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📚 Library Usage

### 1. Pandas (Data Processing)

| Operation | File | Description |
|-----------|------|-------------|
| `pd.read_csv()` | `data_engine.py` | Import CSV datasets |
| `fillna()`, `dropna()` | `data_engine.py` | Handle missing values |
| `drop_duplicates()` | `data_engine.py` | Remove duplicate records |
| `pd.to_datetime()` | `data_engine.py` | Date conversion |
| `groupby().agg()` | `data_engine.py` | GroupBy analysis |
| `rolling().mean()` | `data_engine.py` | Rolling averages |
| `expanding().mean()` | `data_engine.py` | Moving averages |
| `pd.merge()` | `data_engine.py` | Dataset joining |
| `pd.concat()` | `data_engine.py` | DataFrame concatenation |
| `to_csv()` | `preprocessing.py` | Export processed data |
| Feature engineering | `preprocessing.py` | Goal diff, win rate, momentum |

### 2. NumPy (Numerical Computing)

| Operation | File | Description |
|-----------|------|-------------|
| `np.random.poisson()` | `simulation.py` | Poisson distribution goals |
| `np.random.normal()` | `simulation.py` | Normal distribution runs |
| Monte Carlo Simulation | `simulation.py` | 10,000+ match simulations |
| `np.outer()` | `simulation.py` | Probability matrix |
| `np.tril()`, `np.triu()` | `simulation.py` | Matrix operations |
| `np.mean()`, `np.std()` | `statistics.py` | Statistical calculations |
| `np.clip()` | `simulation.py` | Value clamping |
| `np.average()` (weighted) | `preprocessing.py` | Momentum scoring |
| `np.concatenate()` | `preprocessing.py` | Feature vector assembly |
| Array creation & indexing | `preprocessing.py` | Feature vectors |

### 3. Statsmodels (Statistical Analysis)

| Operation | File | Description |
|-----------|------|-------------|
| `sm.OLS()` | `statistics.py` | Ordinary Least Squares regression |
| `ARIMA()` | `statistics.py` | Time-series forecasting |
| Rolling Mean | `statistics.py` | Smoothed averages |
| Moving Average (EWMA) | `statistics.py` | Exponential weighting |
| Trend Detection | `statistics.py` | OLS coefficient analysis |
| Confidence Intervals | `statistics.py` | Statistical bounds |
| Residual Analysis | `statistics.py` | Prediction error analysis |
| `get_forecast()` | `statistics.py` | Future value prediction |

### 4. TensorFlow (Machine Learning)

| Operation | File | Description |
|-----------|------|-------------|
| `tf.keras.Sequential()` | `tensorflow_model.py` | Model architecture |
| `Dense()` layers | `tensorflow_model.py` | Fully connected layers |
| `BatchNormalization()` | `tensorflow_model.py` | Training stability |
| `Dropout()` | `tensorflow_model.py` | Regularization |
| `model.compile()` | `tensorflow_model.py` | Optimizer & loss config |
| `model.fit()` | `tensorflow_model.py` | Model training |
| `model.predict()` | `tensorflow_model.py` | Inference |
| `model.save()` | `tensorflow_model.py` | Model persistence |
| `load_model()` | `tensorflow_model.py` | Model restoration |
| `EarlyStopping` | `tensorflow_model.py` | Training callback |
| Softmax output | `tensorflow_model.py` | Probability distribution |

### 5. Tkinter (GUI)

| Widget/Feature | File | Description |
|----------------|------|-------------|
| `tk.Tk()` root window | `main.py` | Application window |
| `tk.Frame` | `widgets.py` | Layout containers |
| `tk.Canvas` | `widgets.py` | Custom drawing (buttons, bars) |
| `ttk.Combobox` | `prediction.py` | Dropdown selectors |
| `tk.Entry` | `history.py` | Search input |
| `tk.Label` | All GUI files | Text display |
| `FigureCanvasTkAgg` | `analytics.py` | Embedded Matplotlib charts |
| `tk.Toplevel` | `history.py` | Popup windows |
| `threading.Thread` | `prediction.py` | Background prediction |
| `after()` scheduling | `widgets.py` | Animation timing |
| Custom scrolling | `widgets.py` | `ScrollableFrame` |
| Dark theme styling | `widgets.py` | `apply_dark_theme()` |

---

## 🧮 Algorithms

### Monte Carlo Simulation
The Monte Carlo method simulates 10,000+ matches by:
1. Computing expected goals/runs using Poisson distribution (`np.random.poisson`)
2. Running vectorized simulations using NumPy arrays
3. Counting win/draw/loss outcomes from simulated scores
4. Computing probability distributions and confidence intervals

### Poisson Distribution
For football: `P(X=k) = (λ^k × e^(-λ)) / k!`
- λ (lambda) = average goals per match
- Used for both exact probability calculation and simulation

### ARIMA Forecasting
Time-series model with parameters (p, d, q) = (1, 0, 1):
- Analyzes historical win/loss patterns
- Forecasts future team form
- Provides confidence intervals for predictions

### Neural Network Architecture
```
Input(16) → Dense(64, ReLU) → BatchNorm → Dropout(0.2)
         → Dense(32, ReLU) → BatchNorm → Dropout(0.2)
         → Dense(16, ReLU)
         → Dense(3/2, Softmax) → Output
```

### Weighted Ensemble
Final prediction = 30% × Statsmodels + 40% × TensorFlow + 30% × Monte Carlo

---

## 📸 Screenshots

*Screenshots will be added after running the application.*

| Page | Description |
|------|-------------|
| Home | Landing page with sport cards and overview |
| Prediction | Team selection and AI prediction results |
| Analytics | Interactive charts and visualizations |
| History | Searchable prediction history table |

---

## 🔮 Future Enhancements

- [ ] Real-time data feeds from sports APIs
- [ ] Player-level analysis and predictions
- [ ] League table simulation
- [ ] Custom model training parameters
- [ ] PDF report generation
- [ ] Database integration (SQLite)
- [ ] Multi-language support
- [ ] Cloud model deployment

---

## 🎓 Educational Outcomes

### Bloom's Taxonomy Alignment

**BL3 – Applying:**
- Apply Pandas for real-world data processing tasks
- Apply NumPy for numerical computations and simulations
- Apply Statsmodels for statistical analysis
- Apply TensorFlow for machine learning prediction
- Apply Tkinter for GUI development

**BL4 – Analyzing:**
- Analyze sports data to identify patterns and trends
- Analyze model performance through confidence metrics
- Analyze statistical significance through p-values and regression
- Analyze time-series data for forecasting
- Compare ensemble predictions from multiple methods

### Course Outcomes Demonstrated
1. **CO1:** Use Python libraries for data analysis (Pandas, NumPy)
2. **CO2:** Implement machine learning models (TensorFlow)
3. **CO3:** Apply statistical methods for analysis (Statsmodels)
4. **CO4:** Build desktop applications with GUI (Tkinter)
5. **CO5:** Integrate multiple libraries in a cohesive project

---

## 📄 License

This project is created for educational purposes as part of Semester 3 Python coursework.

---

*Built with ❤️ using Python, TensorFlow, Pandas, NumPy, Statsmodels, and Tkinter*
# OracleXI
