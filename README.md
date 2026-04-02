# FindIT Electricity Demand Forecasting

Welcome to the **FindIT Electricity Demand Forecasting** repository! This project houses a state-of-the-art Hybrid AI architecture (Prophet + LightGBM + Isolation Forest) designed to scientifically forecast national electricity grids by balancing macroeconomic trending with micro-anomalous weather and holiday fluctuations.

## 📖 Project Documentation

To truly understand how this AI architecture operates, please refer to our extensive documentation suite located in the `/Documentation` folder.

This project uses mathematically robust **Joint Bayesian Optimization (Optuna)**, strict chronological quarantining to prevent data leakage, and Isolation Forests to protect the ML algorithm from corrupted spikes. 

**Please read the following documents to dive deep into our logic:**

### 1. The Master Architecture
**File:** [`/Documentation/AI_Project_Documentation.md`](./Documentation/AI_Project_Documentation.md)
* **What's Inside:** This is the core whitepaper of the project (available in Bahasa Indonesia). It explains the entire algorithmic pipeline, covering:
  - Why we use an **Isolation Forest** instead of standard mathematical anomaly detection.
  - The philosophy behind the **Prophet** Base layer and the **LightGBM** micro-corrector layer.
  - The **Optuna Bayesian Optimization** engine that automatically finds hyperparameter synergies across all 3 algorithms simultaneously.
  - Explanation of our evaluation metrics (MAPE, MAE, RMSE) and how we strictly avoid "Target Leakage".

### 2. The Data Dictionary
**File:** [`/Documentation/dataset_documentation.md`](./Documentation/dataset_documentation.md)
* **What's Inside:** If you want to know exactly what the physical data represents, read this document. It covers:
  - Feature Engineering definitions (`Lag_1`, `Rolling_7`).
  - How exogenous variables (`Avg_Temp`, `Rainfall`, `Is_Holiday`) interact with the Grid.
  - Why the independent variables are entirely organic while the Demand targets utilize calculated chronological allocation. 

---

## 🚀 How to Run the Pipeline

If you want to run the model locally, all dependencies and execution states have been preserved for you.

1. **Install Dependencies:**
   ```bash
   pip install -r Notebook/requirements.txt
   ```

2. **Execute Interactive Training (Jupyter):**
   Open `Notebook/training.ipynb` to explore the 30-trial Optuna Bayesian Optimization live in your browser and serialize your intelligent weights.

3. **Execute Production Inference:**
   Open `Notebook/inference.ipynb` to effortlessly load the frozen `.joblib` models and generate mathematical demand predictions off unlabelled data.

---

> *"Predicting the future of human electricity consumption by synthesizing mathematical trends and environmental chaos."*
