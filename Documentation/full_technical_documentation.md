# Dokumentasi Teknis Lengkap: Castricity — Hybrid AI Electricity Demand Forecasting

> **Proyek:** FindIT 2026 Hackathon · Tim Nekat Aja  
> **Versi Terakhir Diperbarui:** 10 April 2026  
> **Bahasa:** Bahasa Indonesia (dengan terminologi teknis Inggris)

---

## Daftar Isi

1. [Ringkasan Proyek](#1-ringkasan-proyek)
2. [Struktur Direktori Proyek](#2-struktur-direktori-proyek)
3. [Sumber Data Mentah & Asal Usulnya](#3-sumber-data-mentah--asal-usulnya)
4. [Pipeline Pembangunan Dataset (`build_real_datasets.py`)](#4-pipeline-pembangunan-dataset-build_real_datasetspy)
5. [Fitur-Fitur Dataset (Feature Dictionary)](#5-fitur-fitur-dataset-feature-dictionary)
6. [Daftar Pustaka / Library yang Digunakan & Alasannya](#6-daftar-pustaka--library-yang-digunakan--alasannya)
7. [Arsitektur Model Hybrid (`hybrid_model.py`)](#7-arsitektur-model-hybrid-hybrid_modelpy)
8. [Deteksi Anomali & Strategi Imputasi (Bukan Penghapusan)](#8-deteksi-anomali--strategi-imputasi-bukan-penghapusan)
9. [Joint Bayesian Optimization (Optuna)](#9-joint-bayesian-optimization-optuna)
10. [Anti-Overfitting Architecture: Train-Only Strategy](#10-anti-overfitting-architecture-train-only-strategy)
11. [Evaluasi Kinerja Model](#11-evaluasi-kinerja-model)
12. [Explainable AI (XAI) — SHAP Values](#12-explainable-ai-xai--shap-values)
13. [Dashboard Interaktif (`dashboard.py`)](#13-dashboard-interaktif-dashboardpy)
14. [Artefak Output yang Dihasilkan](#14-artefak-output-yang-dihasilkan)
15. [Cara Menjalankan Proyek](#15-cara-menjalankan-proyek)

---

## 1. Ringkasan Proyek

Castricity adalah platform prediksi permintaan listrik regional Indonesia berbasis Machine Learning dengan pendekatan Explainable AI (XAI). Alih-alih menggunakan satu model tunggal, proyek ini menggunakan **Arsitektur Hybrid 3-Komponen** yang masing-masing menangani dimensi permasalahan yang berbeda:

| Komponen | Library | Peran |
|:---|:---|:---|
| **Forecaster** | Prophet (Meta) | Menangkap pola waktu: tren tahunan, musiman mingguan. Dilatih **hanya pada train data**. |
| **Regressor** | LightGBM (Microsoft) | Mempelajari residual (kesalahan) Prophet dari 18 fitur eksogen. Dilatih **hanya pada train data**, val = held-out early-stop. |
| **Guardrail** | Isolation Forest (scikit-learn) | Mendeteksi anomali & **mengimputasi** data abnormal agar dataset tetap utuh tanpa lubang |

Seluruh hyperparameter ketiga komponen di atas dioptimasi secara **simultan** menggunakan **Optuna Bayesian Optimization**, bukan secara terpisah.

**Prinsip Desain Utama:**
- **Train-Only Architecture**: Tidak ada model yang melihat data validasi atau test saat training.
- **Regularisasi Berlapis**: L1/L2, shallow trees, extra_trees, min_child_samples.
- **3-Split Evaluation**: Memisahkan diagnosa overfitting (Train→Val) dari distribution shift (Val→Test).

---

## 2. Struktur Direktori Proyek

```
Dataset/
├── Raw Data/                         # Data mentah asli dari berbagai sumber
│   ├── BPS_Electricity/              # CSV distribusi listrik per provinsi (BPS)
│   ├── Kaggle_Climate/               # Data cuaca historis Indonesia (Kaggle/BMKG)
│   │   ├── climate_data.csv          # ~589.265 baris data cuaca harian
│   │   └── Json-Indonesia-holidays/  # API hari libur nasional Indonesia
│   └── World_Bank_Macro/             # Data makroekonomi Indonesia (World Bank)
│       └── API_IDN_DS2_en_csv_v2_8804.csv
│
├── Scripts/
│   ├── build_real_datasets.py        # Membangun dataset harian & bulanan dari raw data
│   └── hybrid_model.py              # Pipeline utama: training + evaluasi + export
│
├── Notebook/
│   ├── training.ipynb                # Notebook training (mirror dari hybrid_model.py)
│   └── inference.ipynb               # Notebook inferensi + SHAP Explainability
│
├── Documentation/                    # Dokumentasi proyek
│   ├── full_technical_documentation.md   # ← FILE INI
│   ├── AI_Project_Documentation.md       # Metodologi AI (Bab 2)
│   ├── dataset_documentation.md          # Feature Dictionary
│   └── hybrid_model_architecture.md      # Ringkasan arsitektur hybrid
│
├── Models/                           # Model tersimpan (output training)
│   ├── prophet_model.joblib          # Model Prophet terlatih
│   ├── lgbm_model.joblib             # Model LightGBM terlatih
│   ├── iso_forest.joblib             # Model Isolation Forest terlatih
│   ├── knn_imputer.joblib            # KNN Imputer (fitted on train)
│   └── best_hybrid_params.json       # Hyperparameter terbaik dari Optuna
│
├── Outputs/                          # Dataset olahan & visualisasi
│   ├── dataset_daily_processed.csv       # Dataset harian siap model (dari build script)
│   ├── dataset_monthly_processed.csv     # Dataset bulanan siap model
│   ├── dataset_daily_with_predictions.csv # Dataset + kolom prediksi (dari hybrid_model)
│   ├── fig0_anomalies_detected.png       # Visualisasi anomali
│   ├── fig1_actual_vs_predicted.png      # Aktual vs Prediksi (full timeline)
│   ├── fig2_model_comparison.png         # Bar chart MAE/RMSE/MAPE
│   ├── fig3_residual_distribution.png    # Distribusi residual
│   └── fig4_shap_summary.png            # SHAP feature importance
│
├── dashboard.py                      # Streamlit dashboard interaktif
├── output_xai.png                    # SHAP plot untuk dashboard
└── README.md                         # Pengantar proyek
```

---

## 3. Sumber Data Mentah & Asal Usulnya

### 3.1 BPS — Listrik yang Didistribusikan Menurut Provinsi (GWh)

- **Lokasi:** `Raw Data/BPS_Electricity/Listrik yang Didistribusikan Menurut Provinsi (GWh), YYYY.csv`
- **Sumber Asli:** Badan Pusat Statistik (BPS) Republik Indonesia
- **Periode:** 2011–2024 (yang digunakan: 2018–2023)
- **Isi:** Total energi listrik yang didistribusikan per provinsi dalam satuan GWh per tahun.
- **Cara Parsing:** Script membaca baris yang mengandung kata "Indonesia" atau "Total" untuk mendapatkan angka agregat nasional tahunan. Jika parsing gagal, digunakan nilai fallback hardcoded:

  | Tahun | Fallback (GWh) |
  |:---:|:---:|
  | 2018 | 232.000 |
  | 2019 | 243.000 |
  | 2020 | 240.000 |
  | 2021 | 254.000 |
  | 2022 | 270.000 |
  | 2023 | 285.000 |

- **Digunakan untuk:** Menghitung `Demand_MWh` harian melalui alokasi proporsional.

### 3.2 World Bank — Data Makroekonomi Indonesia

- **Lokasi:** `Raw Data/World_Bank_Macro/API_IDN_DS2_en_csv_v2_8804.csv`
- **Sumber Asli:** World Bank Open Data API — Indonesia country profile
- **Data yang Diekstrak:**
  - **GDP** (Indicator Code: `NY.GDP.MKTP.CD`) — dalam Miliar USD
  - **Populasi** (Indicator Code: `SP.POP.TOTL`) — jumlah penduduk

- **Cara Parsing:** File CSV dari World Bank memiliki 4 baris header. Script melewatkan 4 baris pertama (`skiprows=4`), lalu memfilter baris berdasarkan `Indicator Code` dan membaca kolom tahun (`"2018"`, `"2019"`, dst.).
- **Digunakan untuk:** Dataset bulanan (`dataset_monthly_processed.csv`) sebagai fitur eksogen makro.

### 3.3 Kaggle/BMKG — Data Cuaca Historis Indonesia

- **Lokasi:** `Raw Data/Kaggle_Climate/climate_data.csv`
- **Sumber Asli:** Dataset Kaggle yang berasal dari stasiun BMKG di seluruh Indonesia
- **Volume:** ~589.265 baris, 19 fitur cuaca
- **Data yang Diekstrak:**
  - **`Tavg`** → Suhu rata-rata harian (°C)
  - **`RR`** → Curah hujan harian (mm)
- **Cara Parsing:** Kolom `date` diparse dengan format `dd-mm-YYYY`, lalu diagregasi rata-rata nasional per hari (`groupby('date').agg({'Tavg': 'mean', 'RR': 'mean'})`).
- **Penanganan Data Kosong:** Jika suatu tanggal tidak memiliki data cuaca, diisi dengan rata-rata musiman (seasonal average) berdasarkan `DayOfYear`. Jika tetap kosong, diisi default: Suhu = 27.5°C, Curah Hujan = 5.0 mm.

### 3.4 Json Indonesia Holidays — Hari Libur Nasional

- **Lokasi:** `Raw Data/Kaggle_Climate/Json-Indonesia-holidays/api.json`
- **Sumber Asli:** Repository open-source `guangrei/Json-Indonesia-holidays`
- **Format:** JSON dengan kunci berupa tanggal dan nilai yang memiliki field `libur: true/false`
- **Cara Parsing:** Membaca semua entri di mana `libur == True` dan mengonversi ke list `pd.Timestamp`.
- **Digunakan untuk:** Kolom `Is_Holiday` (bernilai 1 pada hari libur nasional, 0 lainnya).

---

## 4. Pipeline Pembangunan Dataset (`build_real_datasets.py`)

Script ini bertanggung jawab mengubah seluruh data mentah menjadi 2 dataset siap model.

### 4.1 Langkah-Langkah Pipeline

#### Langkah 1: Parsing BPS Yearly Demand
Membaca seluruh file CSV distribusi listrik per provinsi, mengekstrak total nasional per tahun (2018–2023) dalam satuan GWh.

#### Langkah 2: Parsing Makroekonomi
Membaca GDP dan Populasi dari file World Bank, disimpan dalam dictionary `macro_dict[year]`.

#### Langkah 3: Parsing Kalender & Cuaca
- Membaca `climate_data.csv` → agregasi harian nasional (`Avg_Temp`, `Rainfall`)
- Membaca `api.json` → list tanggal libur nasional
- Membuat kerangka harian 1 Januari 2018 – 31 Desember 2023

#### Langkah 4: Distribusi Permintaan Tahunan ke Harian
Total demand GWh tahunan dari BPS didistribusikan ke level harian menggunakan **bobot proporsional**:

```
Daily_Weight = 1.0
  - 0.2 jika akhir pekan (Is_Weekend)
  - 0.3 jika hari libur (Is_Holiday)
  + (Avg_Temp - 27) × 0.05   ← hari lebih panas = lebih banyak AC

Demand_MWh_hari = (Daily_Weight_hari / Total_Weight_tahun) × Total_MWh_tahun
```

#### Langkah 5: Noise Stokastik (Anti Target Leakage)
Tanpa noise, `Demand_MWh` adalah fungsi deterministik murni dari `Is_Weekend`, `Is_Holiday`, dan `Avg_Temp` — artinya model hanya perlu merekayasa balik formula alokasi untuk mendapat akurasi "sempurna". Ini adalah **Target Leakage**.

Untuk mencegahnya, dua lapisan noise ditambahkan:

| Tipe Noise | Formula | Efek Simulasi |
|:---|:---|:---|
| **Multiplikatif ±5%** | `Demand × (1 + N(0, 0.05))` | Volatilitas harian tak terduga (pabrik lembur, AC acak) |
| **Aditif ~3%** | `Demand + N(0, mean_demand × 0.03)` | Ketidakpastian pengukuran, baseline error |

Total variasi gabungan ~6–8% memaksa model belajar generalisasi dari pola nyata.

#### Langkah 6: Feature Engineering
- **`Lag_1`**: Demand 1 hari sebelumnya (H-1)
- **`Lag_7`**: Demand 7 hari sebelumnya (minggu lalu, hari yang sama)
- **`Lag_30`**: Demand 30 hari sebelumnya (bulan lalu)
- **`Rolling_7`**: Rata-rata bergerak 7 hari terakhir

30 baris pertama dibuang agar semua fitur lag terisi (tidak `NaN`).

#### Langkah 7: Export
- `Outputs/dataset_daily_processed.csv` — dataset harian
- `Outputs/dataset_monthly_processed.csv` — dataset bulanan (agregasi + makro)

---

## 5. Fitur-Fitur Dataset (Feature Dictionary)

### 5.1 Dataset Harian (18 Fitur untuk LightGBM)

| Kolom | Tipe | Deskripsi | Asal |
|:---|:---|:---|:---|
| `Date` | Datetime | Tanggal dalam format `YYYY-MM-DD` | Generated index |
| `Demand_MWh` | Float | **TARGET PREDIKSI** — beban listrik harian nasional (MWh) | BPS yearly → alokasi proporsional + noise |
| `Day_of_Week` | Integer | 0=Senin … 6=Minggu | Dihitung dari `Date` |
| `Is_Weekend` | Integer | 1=Sabtu/Minggu, 0=hari kerja | Filter dari `Date` |
| `Is_Holiday` | Integer | 1=Libur Nasional, 0=bukan | JSON holidays (guangrei) |
| `Month`, `DayOfYear`, `WeekOfYear` | Integer | Siklus Temporal & Kalenderisasi | Dihitung dari `Date` |
| `Trend` | Integer | Hari ke-N sejak 2018-01-01 (pertumbuhan linier) | Dihitung dari `Date` |
| `Avg_Temp` | Float | Suhu rata-rata nasional harian (°C) | Kaggle/BMKG `climate_data.csv` |
| `Rainfall` | Float | Curah hujan rata-rata harian (mm) | Kaggle/BMKG `climate_data.csv` |
| `Temp_Lag_1` | Float | Suhu historis H-1 | Feature engineering |
| `Lag_1` ... `Lag_30` | Float | Demand 1, 2, 7, 14, 30 hari sebelumnya | Feature engineering |
| `Rolling_7`, `14`, `30` | Float | Moving average demand berbagai batas waktu | Feature engineering |

### 5.2 Dataset Bulanan

| Kolom | Tipe | Deskripsi | Asal |
|:---|:---|:---|:---|
| `Year` | Integer | Tahun pencatatan | Generated index |
| `Month` | Integer | Bulan (1–12) | Generated index |
| `Demand_GWh` | Float | **TARGET** — total demand bulanan (GWh) | Agregasi dari `Demand_MWh` |
| `GDP` | Float | PDB Indonesia (Miliar USD) | World Bank (`NY.GDP.MKTP.CD`) |
| `Population` | Float | Populasi Indonesia | World Bank (`SP.POP.TOTL`) |
| `Avg_Temp` | Float | Rata-rata suhu bulanan | Agregasi dari harian |
| `Lag_1` | Float | Demand 1 bulan sebelumnya | Feature engineering |
| `Lag_12` | Float | Demand 12 bulan sebelumnya (YoY) | Feature engineering |
| `Rolling_12` | Float | Moving average 12 bulan | Feature engineering |

---

## 6. Daftar Pustaka / Library yang Digunakan & Alasannya

### 6.1 Library Inti Model

| Library | Alasan Penggunaan |
|:---|:---|
| **`prophet`** (Meta/Facebook) | Secara native memahami pola musiman (weekly, yearly seasonality). Tidak memerlukan feature engineering manual untuk tren waktu. Kuat terhadap pergeseran tanggal Hijriah. Menggunakan `Avg_Temp` sebagai exogenous regressor. |
| **`lightgbm`** (Microsoft) | Algoritma gradient boosting tree tercepat untuk dataset tabular berukuran sedang. Mendukung `extra_trees` mode untuk regularisasi tambahan. Kompatibel penuh dengan SHAP untuk explainability. Digunakan untuk mempelajari residual Prophet. |
| **`scikit-learn`** | Menyediakan `IsolationForest` untuk deteksi anomali, `KNNImputer` untuk imputasi missing values, serta metrik evaluasi (`mean_squared_error`, `mean_absolute_error`, `precision_score`, `recall_score`, `f1_score`). |
| **`optuna`** | Framework Bayesian Optimization state-of-the-art — `TPESampler` (Tree-structured Parzen Estimator). Jauh lebih efisien dari grid search. Digunakan untuk mengoptimasi hyperparameter **semua 3 model secara simultan** dalam satu objective function. |
| **`shap`** | Library Explainable AI (XAI) yang menghitung kontribusi setiap fitur. `TreeExplainer` khusus dioptimalkan untuk model tree-based. Menghasilkan summary plot, waterfall plot, force plot, dan dependence plot. |

### 6.2 Library Pemrosesan Data

| Library | Alasan Penggunaan |
|:---|:---|
| **`pandas`** | Manipulasi DataFrame: membaca CSV, merge, groupby, feature engineering lag/rolling, export CSV. |
| **`numpy`** | Operasi numerik: noise stokastik Gaussian, masking boolean, perhitungan metrik MAPE manual, IQR bounds. |

### 6.3 Library Visualisasi

| Library | Alasan Penggunaan |
|:---|:---|
| **`matplotlib`** | Backend utama visualisasi statis: anomaly plot, comparison bar chart, residual distribution, SHAP summary. Diatur dengan `Agg` backend (non-interactive) agar bisa berjalan di server tanpa GUI. |
| **`plotly`** | Visualisasi interaktif di dashboard Streamlit: grafik time series dengan hover, bar chart SHAP lokal. |

### 6.4 Library Infrastruktur

| Library | Alasan Penggunaan |
|:---|:---|
| **`joblib`** | Serialisasi model (save/load) ke format `.joblib`. Lebih cepat dari `pickle` untuk objek numpy/sklearn besar. Menyimpan `prophet_model`, `lgbm_model`, `iso_forest`, dan `knn_imputer`. |
| **`streamlit`** | Framework dashboard web interaktif. Memungkinkan pembuatan UI prediksi tanpa menulis HTML/CSS/JS. |
| **`holidays`** | Library Python yang menyediakan daftar hari libur nasional per negara. `holidays.Indonesia()` dipakai di dashboard untuk auto-deteksi hari libur. |

---

## 7. Arsitektur Model Hybrid (`hybrid_model.py`)

### 7.1 Alur Pipeline End-to-End

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: DATA INGESTION                           │
│   Baca train/val/test CSV → parse tanggal                              │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: KNN IMPUTATION (ZERO-LEAKAGE)                │
│   KNNImputer fit STRICTLY on train → transform all splits              │
│   Export knn_imputer.joblib untuk inference                            │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      STEP 3: FEATURE ENGINEERING (18 FEATURES)          │
│   Month, DayOfYear, WeekOfYear, Trend, Lag_2, Lag_14,                  │
│   Rolling_14, Rolling_30, Temp_Lag_1                                    │
│   (ditambahkan ke fitur yang sudah ada dari build script)              │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│         STEP 4: JOINT BAYESIAN OPTIMIZATION (OPTUNA, 50 Trials)        │
│   TPESampler seed=0 — optimasi SIMULTAN 12 hyperparameter:             │
│   • contamination (IF)                                                  │
│   • changepoint_prior_scale, seasonality_prior_scale, n_changepoints   │
│   • learning_rate, max_depth, num_leaves, subsample, colsample_bytree  │
│   • min_child_samples, reg_alpha, reg_lambda                           │
│   Objective: minimize MAE pada validation set                          │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│            STEP 5: FINAL ARCHITECTURE TRAINING (TRAIN-ONLY)             │
│                                                                         │
│   5a. ANOMALY DETECTION + IMPUTATION                                    │
│       IsolationForest(n_estimators=300) + IQR → detect anomalies       │
│       Anomali DIIMPUTASI with 7-day clean mean (NOT removed)           │
│                                                                         │
│   5b. PROPHET TRAINING (TRAIN-ONLY)                                    │
│       Input: train_df_clean ONLY                                        │
│       Regressor: Avg_Temp                                               │
│       Output: Prophet_Pred for ALL splits (train, val, test)           │
│       Val & Test = FULLY OUT-OF-SAMPLE                                  │
│                                                                         │
│   5c. LIGHTGBM RESIDUAL TRAINING (TRAIN-ONLY, Val = held-out)          │
│       Train on: train_df_clean residuals ONLY                           │
│       Early stop on: val_df residuals (NEVER trained on)                │
│       n_estimators=800, extra_trees=True                                │
│       Regularised: reg_alpha, reg_lambda, min_child_samples            │
│                                                                         │
│   5d. FINAL PREDICTION                                                  │
│       Final_Pred = Prophet_Pred + LGBM_Residual_Pred                   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              STEP 6: 3-SPLIT EVALUATION & DIAGNOSTICS                   │
│   Train (in-sample) / Val (out-of-sample) / Test (out-of-sample)       │
│   Diagnostics: Overfitting (Train→Val Gap) vs Dist Shift (Val→Test)    │
│   Export: model .joblib, predictions CSV, visualisasi PNG              │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Mengapa Hybrid? Mengapa Bukan 1 Model Saja?

Konsumsi listrik di Indonesia dipengaruhi oleh **dua poros yang saling bertolak-belakang**:

1. **Poros Temporal (Waktu Kalender):** Listrik sangat patuh terhadap jam kerja, hari libur nasional (Idul Fitri, Natal), dan musim. **Prophet** secara spesifik didesain untuk menangkap pola ini.

2. **Poros Kausalitas (Lingkungan & Eksogen):** Suhu panas memicu pemakaian AC ekstrem. Curah hujan mempengaruhi kebutuhan pendinginan. **LightGBM** menangkap variasi ini melalui 18 fitur eksogen.

Tidak ada satu model yang bisa menangani keduanya secara optimal. Oleh karena itu, kita memisah tugas (**decoupling**):
- Prophet menangani baseline temporal → menghasilkan prediksi dasar
- LightGBM menangani residual (selisih antara aktual dan prediksi Prophet) → koreksi mikro
- **Prediksi Akhir = Prophet + LightGBM Residual**

### 7.3 Detail Komponen Prophet

**Library:** `prophet` (oleh Meta/Facebook)

**Konfigurasi:**
```python
Prophet(
    yearly_seasonality=True,    # Menangkap pola tahunan (musim kemarau/hujan)
    weekly_seasonality=True,    # Menangkap pola hari kerja vs akhir pekan
    daily_seasonality=False,    # TIDAK diaktifkan — data kita sudah level harian
    changepoint_prior_scale=...,  # Dari Optuna (0.001–0.1) — tren SANGAT halus
    seasonality_prior_scale=...,  # Dari Optuna (0.01–1.0) — musiman terkontrol
    n_changepoints=...,           # Dari Optuna (5–20) — sedikit perubahan tren
)
prophet_model.add_regressor('Avg_Temp')  # Suhu sebagai regressor eksogen
```

**Training Data:** `train_df_clean` ONLY (tidak pernah melihat val atau test).

**Output:** Kolom `yhat` — prediksi baseline temporal. Prediksi dihasilkan untuk semua split (train, val, test), namun val dan test adalah **fully out-of-sample**.

### 7.4 Detail Komponen LightGBM

**Library:** `lightgbm` (oleh Microsoft)

**Tugas:** Memprediksi `Prophet_Residual = Demand_MWh - Prophet_Pred`

**Konfigurasi:**
```python
LGBMRegressor(
    learning_rate=...,      # Dari Optuna (0.001–0.05) — konvergensi lambat & stabil
    max_depth=...,          # Dari Optuna (2–4) — pohon SANGAT dangkal
    num_leaves=...,         # Dari Optuna (4–15) — daun SANGAT sedikit
    subsample=...,          # Dari Optuna (0.4–0.8) — stochastic subsampling
    colsample_bytree=...,   # Dari Optuna (0.4–0.8) — feature subsampling
    min_child_samples=...,  # Dari Optuna (15–60) — minimum sampel per leaf
    reg_alpha=...,          # Dari Optuna (0.01–10.0) — L1 regularisasi
    reg_lambda=...,         # Dari Optuna (0.01–10.0) — L2 regularisasi
    n_estimators=800,       # Cap jumlah pohon (early stopping biasanya berhenti lebih awal)
    extra_trees=True,       # Random threshold splitting untuk generalisasi
    random_state=42,        # Reproducibility
    n_jobs=-1,              # Gunakan semua CPU cores
    verbose=-1,             # Tidak ada output log
)
```

**Training:** `train_df_clean` ONLY. `val_df` hanya untuk early-stopping (50 rounds).

**Fitur Input yang Digunakan (18 Fitur):**
| Kategori | Fitur | Mengapa Penting |
|:---|:---|:---|
| Kalender | `Day_of_Week` | Pola hari kerja vs weekend |
| Kalender | `Is_Weekend` | Penurunan beban weekend (industri tutup) |
| Kalender | `Is_Holiday` | Penurunan beban hari libur nasional |
| Temporal | `Month` | Pola musiman bulanan |
| Temporal | `DayOfYear` | Posisi dalam siklus tahunan |
| Temporal | `WeekOfYear` | Posisi dalam siklus mingguan |
| Temporal | `Trend` | Pertumbuhan demand jangka panjang |
| Cuaca | `Avg_Temp` | Suhu tinggi → AC → demand naik |
| Cuaca | `Rainfall` | Hujan → suhu turun → demand turun |
| Cuaca | `Temp_Lag_1` | Retensi panas inersia bangunan |
| Autoregresif | `Lag_1`, `Lag_2` | Inersia konsumsi harian |
| Autoregresif | `Lag_7` | Pola mingguan berulang |
| Autoregresif | `Lag_14`, `Lag_30` | Pola dua-mingguan dan bulanan |
| Rolling | `Rolling_7` | Tren halus 1 minggu |
| Rolling | `Rolling_14`, `Rolling_30` | Tren halus 2/4 minggu |

---

## 8. Deteksi Anomali & Strategi Imputasi (Bukan Penghapusan)

### 8.1 Masalah Awal: Penghapusan Data Menciptakan Lubang

Implementasi awal mendeteksi anomali lalu **menghapus** baris anomali. Ini menyebabkan **lubang (gaps)** dalam time series yang merusak fitur-fitur lag dan rolling.

### 8.2 Solusi: Imputasi dengan Rata-rata 7 Hari Terakhir

Pendekatan saat ini **tidak menghapus** baris apa pun. Nilai anomali **diganti** dengan rata-rata dari data bersih dalam jendela 7 hari ke belakang:

```python
is_anomaly = (train_anomalies == -1) | (iqr_anomalies == -1)
train_df_clean = train_df.copy()

for idx in np.where(is_anomaly)[0]:
    lookback_start = max(0, idx - 7)
    lookback_mask = ~is_anomaly[lookback_start:idx]
    clean_window = train_df[target_col].iloc[lookback_start:idx][lookback_mask]
    if len(clean_window) > 0:
        train_df_clean.iloc[idx, ...] = clean_window.mean()
    else:
        train_df_clean.iloc[idx, ...] = train_df[target_col].mean()
```

### 8.3 Perbandingan Pendekatan

| Aspek | Penghapusan (Lama) | Imputasi (Sekarang) |
|:---|:---|:---|
| Jumlah baris setelah cleaning | Berkurang | **Tetap sama** |
| Integritas fitur lag | ❌ Rusak (gaps) | ✅ Terjaga |
| Informasi temporal | ❌ Hilang | ✅ Terjaga |
| Pengaruh ke Prophet | Data pelatihan lebih sedikit | **Data pelatihan lengkap** |

---

## 9. Joint Bayesian Optimization (Optuna)

### 9.1 Mengapa Tidak Grid Search?

Grid search mengevaluasi **semua kombinasi** parameter secara brute-force. Dengan 12 hyperparameter, grid search memerlukan jutaan evaluasi. Optuna menggunakan **TPESampler** yang memfokuskan pencarian di area yang menjanjikan.

### 9.2 Mengapa Optimasi "Joint" (Simultan)?

Hyperparameter dari 3 komponen **saling mempengaruhi**:
- Jika `contamination` tinggi → lebih banyak data diimputasi → Prophet melihat data berbeda → residual berubah → LightGBM harus beradaptasi
- Jika Prophet terlalu fleksibel (`changepoint_prior_scale` tinggi) → residual lebih kecil → LightGBM kurang berguna

Satu `objective()` function menjalankan **seluruh pipeline** dan mengukur MAE akhir pada validation set.

### 9.3 Parameter yang Dioptimasi & Justifikasi Setiap Range

#### Isolation Forest

| Parameter | Range | Skala | Justifikasi |
|:---|:---|:---|:---|
| `contamination` | 0.001–0.05 | Log | Anomali listrik jarang terjadi (0.1–5% data). Skala log karena sensitivitas tinggi di angka kecil — perbedaan antara 0.001 dan 0.01 jauh lebih signifikan daripada antara 0.01 dan 0.02. |

#### Prophet (Diperketat untuk Anti-Overfitting)

| Parameter | Range | Skala | Justifikasi |
|:---|:---|:---|:---|
| `changepoint_prior_scale` | **0.001–0.1** | Log | **Jauh lebih ketat dari default Prophet (0.05) dan range umum (0.01–0.5).** Nilai rendah memaksa tren bergerak sangat halus. Jika > 0.1, Prophet terlalu fleksibel — mengikuti noise harian alih-alih tren sesungguhnya, menyebabkan overfitting. Dengan noise stokastik ±6-8% di data kami, tren halus adalah pilihan yang tepat. |
| `seasonality_prior_scale` | **0.01–1.0** | Log | **Jauh lebih ketat dari default Prophet (10.0).** Demand listrik memiliki seasonality yang konsisten dan relatif stabil. Amplitudo besar (>1.0) menyebabkan Prophet menangkap variasi noise sebagai "musiman", bukan pola sejati. |
| `n_changepoints` | **5–20** | Linear | **Dikurangi dari range umum (25–50).** Semakin sedikit changepoint = tren lebih mulus. Dalam 6 tahun data (2018–2023), hanya ada ~3-4 perubahan struktural nyata per tahun (musim, Ramadan, COVID). 5–20 changepoint sudah cukup. |

#### LightGBM (Regularisasi Berat untuk Anti-Overfitting)

| Parameter | Range | Skala | Justifikasi |
|:---|:---|:---|:---|
| `learning_rate` | **0.001–0.05** | Log | **Sangat rendah** (umum: 0.01–0.3). Learning rate rendah berarti setiap pohon baru hanya membuat koreksi kecil. Dikombinasikan dengan early stopping (50 rounds), model berhenti di titik optimal — bukan di titik overfit. Skala log karena perbedaan antara 0.001 dan 0.01 lebih signifikan dari 0.01 dan 0.05. |
| `max_depth` | **2–4** | Linear | **Sangat dangkal** (default: tidak terbatas, umum: 3–10). Pohon kedalaman 2 hanya bisa menangkap interaksi 2-fitur (misal: "jika hari kerja DAN suhu > 30°C"). Kedalaman 4 menangkap interaksi 4-fitur. Ini memaksa model fokus pada pola dominan dan mencegah menghafal noise kompleks. |
| `num_leaves` | **4–15** | Linear | **Sangat sedikit** (default LightGBM: 31, umum: 15–127). Dengan hanya 4–15 daun per pohon, setiap "bin keputusan" mengandung banyak data → keputusan lebih robust. Mencegah daun kecil yang menangkap noise. |
| `subsample` | **0.4–0.8** | Linear | **Cukup agresif** (umum: 0.7–1.0). Setiap pohon hanya melihat 40–80% data training secara random. Ini memaksa setiap pohon menjadi "ahli" di subset data berbeda — meningkatkan diversitas dan mengurangi overfitting. Mirip efek bagging di Random Forest. |
| `colsample_bytree` | **0.4–0.8** | Linear | **Cukup agresif** (umum: 0.7–1.0). Setiap pohon hanya menggunakan 40–80% dari 18 fitur. Mencegah model terlalu bergantung pada 1-2 fitur dominan (misal: `Lag_1`). Mendorong model memanfaatkan fitur cuaca dan temporal juga. |
| `min_child_samples` | **15–60** | Linear | Setiap daun harus memiliki minimal 15–60 data point. Ini mencegah daun yang mengandung terlalu sedikit data — yang biasanya menangkap noise unik, bukan pola umum. Angka 15 dipilih agar ~2 minggu data minimum; 60 agar ~2 bulan. |
| `reg_alpha` (L1) | **0.01–10.0** | Log | **Regularisasi Lasso** — mendorong bobot fitur non-informatif menuju nol. Secara efektif melakukan _implicit feature selection_. Range log karena efek L1 sangat non-linear. |
| `reg_lambda` (L2) | **0.01–10.0** | Log | **Regularisasi Ridge** — menekan semua bobot secara proporsional. Mencegah bobot yang terlalu besar pada fitur tertentu. Bekerja komplementer dengan L1: L1 menghilangkan fitur lemah, L2 menekan fitur kuat yang terlalu dominan. |

#### Parameter Tetap (Non-Tuned)

| Parameter | Nilai | Justifikasi |
|:---|:---|:---|
| `n_estimators` | **800** | Cap jumlah pohon maksimal. Dengan early stopping 50 rounds, training biasanya berhenti di 100–400 pohon. Cap 800 memberikan ruang untuk learning rate sangat rendah yang membutuhkan lebih banyak iterasi. Tidak 10000 karena itu mengundang overfitting. |
| `extra_trees` | **True** | Menggunakan **random split threshold** alih-alih threshold optimal. Efek: setiap pohon sedikit lebih "acak" → variance lebih rendah → generalisasi lebih baik. Mirip konsep Extra Trees Classifier di scikit-learn. |
| `early_stopping` | **50 rounds** (final) / **20 rounds** (Optuna) | Jika val error tidak membaik selama 50 iterasi berturut-turut, training dihentikan otomatis. Ini adalah pencegah overfitting paling fundamental — model berhenti begitu mulai "menghafal" alih-alih "belajar". Optuna menggunakan 20 rounds untuk efisiensi waktu pencarian. |

### 9.4 Caching & Retune Policy

Hasil Optuna disimpan di `Models/best_hybrid_params.json` dengan metadata:
- `last_tuned_at`: timestamp kapan terakhir di-tune
- `n_trials`: jumlah trial yang dijalankan
- `retune_every_days`: batas usia parameter (default: 30 hari)

**Retune otomatis terjadi jika:**
- File parameter tidak ditemukan
- Parameter sudah lebih tua dari `RETUNE_EVERY_DAYS` (30 hari)
- `FORCE_RETUNE = True` diset manual

---

## 10. Anti-Overfitting Architecture: Train-Only Strategy

### 10.1 Masalah Sebelumnya

Arsitektur sebelumnya melatih Prophet dan/atau LightGBM pada **gabungan train+val data**. Ini menyebabkan:

1. **Val MAPE terlihat sangat baik** — karena model sudah MELIHAT data val saat training.
2. **Test MAPE jauh lebih tinggi** — karena model belum pernah melihat data test.
3. **Gap Val→Test besar** — yang *tampak* seperti overfitting, padahal sebagian besar adalah konsekuensi dari val yang tidak jujur.

### 10.2 Solusi: Train-Only Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  TRAIN-ONLY ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│  Split      │ Prophet Status    │ LightGBM Status           │
│─────────────│───────────────────│───────────────────────────│
│  Train      │ ✅ In-sample      │ ✅ In-sample              │
│  Val        │ ❌ Out-of-sample  │ ❌ Out-of-sample          │
│  Test       │ ❌ Out-of-sample  │ ❌ Out-of-sample          │
└─────────────────────────────────────────────────────────────┘
```

- **Prophet** dilatih HANYA pada `train_df_clean`. Val dan Test tidak pernah disentuh.
- **LightGBM** dilatih HANYA pada residual `train_df_clean`. Val hanya digunakan sebagai _early-stopping monitor_ — data val DIBACA tapi TIDAK dilatih.
- **Arsitektur final identik dengan arsitektur Optuna** — menghilangkan mismatch antara tuning dan deployment.

### 10.3 Diagnosa 3-Split

Dengan arsitektur ini, evaluasi terpisah pada 3 split memberikan diagnosa presisi:

| Gap | Mengukur | Sehat Jika |
|:---|:---|:---|
| **Train→Val** | **Overfitting** — seberapa banyak performa turun pada data unseen | < 1.0 pp |
| **Val→Test** | **Distribution Shift** — seberapa berbeda pola data test vs val | < 2.0 pp |

Jika Train→Val kecil tapi Val→Test besar → bukan overfitting, melainkan **pola demand listrik di periode test genuinely berbeda** (COVID, perubahan ekonomi, dll). Ini bukan masalah model — ini realitas data.

---

## 11. Evaluasi Kinerja Model

### 11.1 Data Split

| Split | Proporsi | Tujuan | Status untuk Prophet | Status untuk LightGBM |
|:---|:---|:---|:---|:---|
| **Train** | 70% pertama | Melatih kedua model | In-sample | In-sample |
| **Validation** | 15% berikutnya | Early stopping LightGBM + Optuna objective | Out-of-sample | Out-of-sample (hanya monitor) |
| **Test** | 15% terakhir | Evaluasi akhir — TIDAK pernah disentuh | Out-of-sample | Out-of-sample |

### 11.2 Metrik yang Digunakan

| Metrik | Formula | Interpretasi |
|:---|:---|:---|
| **MAE** | `mean(|actual - predicted|)` | "Model meleset rata-rata X MWh per hari." |
| **RMSE** | `sqrt(mean((actual - predicted)²))` | Seperti MAE tapi mengutamakan kesalahan besar. |
| **MAPE** | `mean(|actual - predicted| / actual) × 100%` | Persentase kesalahan relatif. Target: < 5%. |

### 11.3 Output Perbandingan (3-Row Format)

```
==================================================================================
  MODEL COMPARISON: Prophet-Only vs Hybrid (Prophet + LightGBM)
==================================================================================
Metric       | Prophet-Only (Train)   | Hybrid (Train)         | Note
----------------------------------------------------------------------------------
MAE          |         X,XXX.XX MWh   |           XXX.XX MWh   | In-sample
RMSE         |         X,XXX.XX MWh   |         X,XXX.XX MWh   |
MAPE         |            X.XX%       |            X.XX%       |
----------------------------------------------------------------------------------
Metric       | Prophet-Only (Val)     | Hybrid (Val)           | Note
----------------------------------------------------------------------------------
MAE          |         X,XXX.XX MWh   |         X,XXX.XX MWh   | Out-of-sample
RMSE         |        XX,XXX.XX MWh   |        XX,XXX.XX MWh   |
MAPE         |            X.XX%       |            X.XX%       |
----------------------------------------------------------------------------------
Metric       | Prophet-Only (Test)    | Hybrid (Test)          | Note
----------------------------------------------------------------------------------
MAE          |        XX,XXX.XX MWh   |        XX,XXX.XX MWh   | Out-of-sample
RMSE         |        XX,XXX.XX MWh   |        XX,XXX.XX MWh   |
MAPE         |            X.XX%       |            X.XX%       |
==================================================================================

  >> Train→Val Gap:  X.XX pp  (Overfitting indicator)
  >> Val→Test Gap:   X.XX pp  (Distribution Shift indicator)
  >> OVERFIT CHECK:  ✅ Healthy / ⚠️ Mild / ❌ Significant
  >> SHIFT CHECK:    ✅ Stable / ⚠️ Moderate / ❌ Large
```

---

## 12. Explainable AI (XAI) — SHAP Values

### 12.1 Apa itu SHAP?

**SHAP (SHapley Additive exPlanations)** berasal dari teori permainan kooperatif. SHAP mengukur **kontribusi marginal setiap fitur** terhadap prediksi model:

`Prediksi = Base Value + SHAP(Fitur₁) + SHAP(Fitur₂) + ... + SHAP(Fiturₙ)`

### 12.2 Mengapa Penting?

Tanpa XAI, model hanya berupa "black box". Dengan SHAP:
- Membuktikan bahwa suhu 35°C berkontribusi +12% beban
- Menjelaskan bahwa hari libur menurunkan prediksi -15% beban
- Juri hackathon dapat **memvalidasi** model secara transparan

### 12.3 Implementasi di Proyek

**Inference Notebook (`inference.ipynb`) menyediakan 7 tipe analisis SHAP:**

| Tipe | Fungsi |
|:---|:---|
| **Global Summary Plot** | Fitur paling berpengaruh secara keseluruhan (beeswarm plot) |
| **Mean \|SHAP\| Ranking** | Importance ranking numerik per fitur |
| **Tabel Korelasi Bisnis** | Terjemahan setiap fitur ke narasi bisnis |
| **Single-Point Narrative** | Top-5 faktor penentu prediksi hari terakhir |
| **Waterfall Plot** | Dekomposisi prediksi dari base value ke akhir |
| **Force Plot** | Visualisasi tarik-menarik fitur (merah=naik, biru=turun) |
| **Dependence Plot** | Korelasi non-linear fitur ↔ dampak SHAP (top-3 fitur) |

### 12.4 Cara Membaca SHAP Summary Plot

- **Sumbu Y:** Fitur diurutkan dari paling berpengaruh (atas) ke paling kecil (bawah)
- **Sumbu X:** Nilai SHAP — positif = menaikkan prediksi, negatif = menurunkan
- **Warna titik:** Merah = nilai fitur tinggi, biru = nilai fitur rendah

---

## 13. Dashboard Interaktif (`dashboard.py`)

### 13.1 Teknologi
- **Framework:** Streamlit
- **Visualisasi:** Plotly (interaktif), Matplotlib (statis/SHAP)
- **Deteksi Libur:** Library `holidays` (Indonesia)

### 13.2 Tab 1: Validasi Sistem (Historis)

**Tujuan:** Mengevaluasi apakah model bekerja stabil pada data harian historis yang sudah diketahui.

**Konten:**
- KPI cards: jumlah hari, observasi, anomali terdeteksi, MAE, RMSE
- Time series chart: Actual Demand vs Hybrid Prediction + marker anomali
- Prophet Components: dekomposisi tren + seasonality
- Global SHAP: feature importance seluruh dataset
- Penjelasan interpretatif dalam bahasa Indonesia

### 13.3 Tab 2: Future Forecaster & Local XAI

**Tujuan:** Membuat prediksi untuk tanggal masa depan dengan input kustom.

**Input pengguna:**
- Tanggal prediksi (auto-detect hari libur nasional via `holidays.Indonesia()`)
- Suhu udara (°C)
- Curah hujan (mm)
- Override hari libur (untuk simulasi what-if)

**Output:**
1. **Prediksi akhir** dalam MWh (= Prophet baseline + LightGBM residual)
2. **Flag anomali** jika kombinasi fitur dianggap tidak lazim oleh Isolation Forest
3. **Narasi Eksekutif** — Top 3 faktor penentu prediksi
4. **Chart SHAP lokal** — dampak setiap variabel (horizontal bar)
5. **Tabel detail variabel** — nilai input, dampak AI, dan arti fisik

---

## 14. Artefak Output yang Dihasilkan

### 14.1 Model Files (`Models/`)

| File | Deskripsi |
|:---|:---|
| `prophet_model.joblib` | Model Prophet terlatih (tren + seasonality + Avg_Temp regressor) |
| `lgbm_model.joblib` | Model LightGBM terlatih (residual corrector, regularised) |
| `iso_forest.joblib` | Model Isolation Forest (anomaly detector) |
| `knn_imputer.joblib` | KNN Imputer (fitted on train data only) |
| `best_hybrid_params.json` | 12 hyperparameter terbaik + metadata tuning |

### 14.2 Dataset Files (`Outputs/`)

| File | Deskripsi |
|:---|:---|
| `dataset_daily_processed.csv` | Dataset harian siap model (output `build_real_datasets.py`) |
| `dataset_monthly_processed.csv` | Dataset bulanan + makro (output `build_real_datasets.py`) |
| `dataset_daily_with_predictions.csv` | Dataset harian + kolom prediksi (output `hybrid_model.py`) |

### 14.3 Visualisasi (`Outputs/`)

| File | Konten |
|:---|:---|
| `fig0_anomalies_detected.png` | Scatter plot anomali terdeteksi pada data training |
| `fig1_actual_vs_predicted.png` | Time series full: Actual vs Prophet vs Hybrid + split zones |
| `fig2_model_comparison.png` | Bar chart MAE, RMSE, MAPE: Prophet-Only vs Hybrid |
| `fig3_residual_distribution.png` | Histogram residual Prophet vs Hybrid (tighter = better) |
| `fig4_shap_summary.png` | SHAP summary beeswarm plot (global feature importance) |

---

## 15. Cara Menjalankan Proyek

### 15.1 Install Dependencies

```bash
pip install -r Notebook/requirements.txt
```

Atau install secara manual:
```bash
pip install pandas numpy matplotlib prophet lightgbm scikit-learn shap optuna joblib streamlit plotly holidays
```

### 15.2 Bangun Dataset dari Raw Data

```bash
python Scripts/build_real_datasets.py
```

Output: `Outputs/dataset_daily_processed.csv` dan `Outputs/dataset_monthly_processed.csv`

### 15.3 Training Model Hybrid

```bash
python Scripts/hybrid_model.py
```

**Environment variables opsional:**
- `OPTUNA_TRIALS=50` — jumlah trial Optuna (default: 50)
- `RETUNE_EVERY_DAYS=30` — usia maksimal parameter sebelum retune (default: 30)
- `FORCE_RETUNE=True` — paksa retune meskipun parameter masih segar

### 15.4 Jalankan Dashboard

```bash
streamlit run dashboard.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`.

---

> *"Meramalkan masa depan tidak sekadar memutar rata-rata masa lalu, melainkan menyeimbangkan pola cuaca alam bebas dengan agenda kultural masyarakat di atasnya."*
