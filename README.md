# ⚡ FindIT: National Electricity Demand Forecasting (XAI Hybrid AI)

Welcome to the **FindIT Electricity Demand Forecasting** project. This repository contains a State-of-the-Art (SOTA) Hybrid AI architecture designed to forecast national electricity demand by synthesizing long-term macroeconomic trends with short-term environmental and cultural fluctuations.

---

## 🏗️ System Architecture: The "Hybrid Trinity"

Unlike traditional forecasting models, FindIT utilizes a decoupled architecture to handle different aspects of energy load complexity. The system is split into three core engines:

1.  **The Forecaster (Prophet by Meta)**: Captures calendar-based patterns, long-term trends, and the "shock effects" of Indonesian National Holidays (Lunar/Hijriah shifts).
2.  **The Regressor (LightGBM by Microsoft)**: Learns the mathematical residuals (errors) from the Prophet baseline by correlating them with exogenous weather data (Avg_Temp, Rainfall) and Macroeconomic factors.
3.  **The Guardrail (Isolation Forest)**: A spatial-isolation tree model that detects grid anomalies/corrupted spikes and replaces them via 7-day rolling mean imputation to ensure a gap-free time series.

> [!TIP]
> **Joint Bayesian Optimization**: The entire pipeline's hyperparameters are tuned simultaneously using **Optuna**, finding the perfect synergy between Prophet, LightGBM, and the Isolation Forest contamination levels.

---

## 📖 Detailed Documentation

| Document | Description |
| :--- | :--- |
| 🏛️ [**Master Architecture**](./Documentation/hybrid_model_architecture.md) | High-level technical summary of the Hybrid Engine and why we chose this specific stack. |
| 📄 [**Full Technical Whitepaper**](./Documentation/AI_Project_Documentation.md) | In-depth breakdown (Bahasa Indonesia) of the algorithmic logic and data philosophy. |
| 📘 [**Comprehensive Tech Doc**](./Documentation/full_technical_documentation.md) | Detailed technical specifications and implementation details. |
| 📊 [**Data Dictionary**](./Documentation/dataset_documentation.md) | Definitions for features like `Lag_1`, `Rolling_7`, and exogenous variable interactions. |

---

## 🚀 Getting Started

### 1. Installation & Environment Setup

It is recommended to use a virtual environment.

```bash
# Create and activate venv (macOS/Linux)
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r Notebook/requirements.txt

# Install Dashboard & UI dependencies
pip install streamlit plotly holidays
```

### 2. Training the Model
The core logic resides in [`Scripts/hybrid_model.py`](./Scripts/hybrid_model.py). Running this script will:
1.  Perform Pre-processing & Anomaly Imputation.
2.  Execute **30 trials of Optuna Bayesian Optimization**.
3.  Train the final Prophet + LightGBM champion models.
4.  Export serialized models to `/Models` and visualizations to `/Outputs`.

```bash
python Scripts/hybrid_model.py
```

### 3. Launching the XAI Dashboard
Once the models are trained, you can launch the interactive Streamlit dashboard to explore predictions and Explainable AI (XAI) impacts.

```bash
streamlit run dashboard.py
```

---

## 🛠️ Key Program Modules

*   **[`Scripts/hybrid_model.py`](./Scripts/hybrid_model.py)**: The heartbeat of the project. Contains the model definition, tuning logic, and training pipeline.
*   **[`dashboard.py`](./dashboard.py)**: The user interface. Features historical validation, anomaly marking, and a "What-If" future forecaster with Local SHAP explanations.
*   **[`Scripts/build_real_datasets.py`](./Scripts/build_real_datasets.py)**: Script for aggregating raw data into the final processed formats.

---

## ⚖️ Explainable AI (XAI)
This project prioritizes transparency over "Black-Box" predictions. By using **SHAP (SHapley Additive exPlanations)**, we provide:
- **Global Impact**: Understanding which features (e.g., Temperature vs. Holidays) drive the overall model.
- **Local Explanation**: For every single prediction, the dashboard shows exactly *why* the AI predicted that specific MWh value.

---

> *"Predicting the future of human electricity consumption by balancing mathematical trends with environmental chaos."*
