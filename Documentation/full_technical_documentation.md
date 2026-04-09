# Dokumentasi Teknis Lengkap: Castricity — Hybrid AI Electricity Demand Forecasting

> **Proyek:** FindIT 2026 Hackathon · Tim Nekat Aja  
> **Versi Terakhir Diperbarui:** 6 April 2026  
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
10. [Evaluasi Kinerja Model](#10-evaluasi-kinerja-model)
11. [Explainable AI (XAI) — SHAP Values](#11-explainable-ai-xai--shap-values)
12. [Dashboard Interaktif (`dashboard.py`)](#12-dashboard-interaktif-dashboardpy)
13. [Artefak Output yang Dihasilkan](#13-artefak-output-yang-dihasilkan)
14. [Cara Menjalankan Proyek](#14-cara-menjalankan-proyek)

---

## 1. Ringkasan Proyek

Castricity adalah platform prediksi permintaan listrik regional Indonesia berbasis Machine Learning dengan pendekatan Explainable AI (XAI). Alih-alih menggunakan satu model tunggal, proyek ini menggunakan **Arsitektur Hybrid 3-Komponen** yang masing-masing menangani dimensi permasalahan yang berbeda:

| Komponen | Library | Peran |
|:---|:---|:---|
| **Forecaster** | Prophet (Meta) | Menangkap pola waktu: tren tahunan, musiman mingguan, & efek hari libur nasional |
| **Regressor** | LightGBM (Microsoft) | Mempelajari residual (kesalahan) Prophet dari faktor eksogen: cuaca, lag, dll. |
| **Guardrail** | Isolation Forest (scikit-learn) | Mendeteksi anomali & **mengimputasi** data abnormal agar dataset tetap utuh tanpa lubang |

Seluruh hyperparameter ketiga komponen di atas dioptimasi secara **simultan** menggunakan **Optuna Bayesian Optimization**, bukan secara terpisah.

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

### 5.1 Dataset Harian

| Kolom | Tipe | Deskripsi | Asal |
|:---|:---|:---|:---|
| `Date` | Datetime | Tanggal dalam format `YYYY-MM-DD` | Generated index |
| `Demand_MWh` | Float | **TARGET PREDIKSI** — beban listrik harian nasional (MWh) | BPS yearly → alokasi proporsional + noise |
| `Day_of_Week` | Integer | 0=Senin … 6=Minggu | Dihitung dari `Date` |
| `Is_Weekend` | Integer | 1=Sabtu/Minggu, 0=hari kerja | Filter dari `Date` |
| `Is_Holiday` | Integer | 1=Libur Nasional, 0=bukan | JSON holidays (guangrei) |
| `Month`, `DayOfYear`, `WeekOfYear` | Integer | Siklus Temporal & Kalenderisasi | Dihitung dari `Date` |
| `Trend` | Integer | Penanda hari linier berjalannya waktu sejak hari pertama dataset | Dihitung dari `Date` |
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
| `Industrial_Index` | Float | Indeks industri sintetis | Distribusi normal sintetis |
| `Avg_Temp` | Float | Rata-rata suhu bulanan | Agregasi dari harian |
| `Lag_1` | Float | Demand 1 bulan sebelumnya | Feature engineering |
| `Lag_12` | Float | Demand 12 bulan sebelumnya (YoY) | Feature engineering |
| `Rolling_12` | Float | Moving average 12 bulan | Feature engineering |

---

## 6. Daftar Pustaka / Library yang Digunakan & Alasannya

### 6.1 Library Inti Model

| Library | Versi | Alasan Penggunaan |
|:---|:---|:---|
| **`prophet`** (Meta/Facebook) | — | Secara native memahami pola musiman (weekly, yearly seasonality) dan efek hari libur. Tidak memerlukan feature engineering manual untuk tren waktu. Sangat kuat terhadap data kalender Indonesia yang memiliki sistem Hijriah (tanggal Lebaran bergeser tiap tahun). |
| **`lightgbm`** (Microsoft) | — | Algoritma gradient boosting tree tercepat untuk dataset tabular berukuran sedang. Lebih cepat dari XGBoost, lebih sensitif terhadap angka desimal kecil (penting untuk cuaca), dan kompatibel penuh dengan SHAP. Digunakan untuk mempelajari residual (kesalahan) Prophet. |
| **`scikit-learn`** | — | Menyediakan `IsolationForest` untuk deteksi anomali, serta metrik evaluasi (`mean_squared_error`, `mean_absolute_error`, `precision_score`, `recall_score`, `f1_score`). Juga `TimeSeriesSplit` untuk cross-validation time-series. |
| **`optuna`** | — | Framework Bayesian Optimization state-of-the-art yang menggunakan `TPESampler` (Tree-structured Parzen Estimator). Jauh lebih efisien dari grid search karena memfokuskan pencarian di area parameter space yang menjanjikan. Digunakan untuk mengoptimasi hyperparameter **semua 3 model secara simultan** dalam satu objective function. |
| **`shap`** | — | Library Explainable AI (XAI) yang menghitung kontribusi setiap fitur terhadap prediksi model. `TreeExplainer` khusus dioptimalkan untuk model tree-based seperti LightGBM. Menghasilkan SHAP summary plot untuk interpretasi global dan local explanation per prediksi. |

### 6.2 Library Pemrosesan Data

| Library | Alasan Penggunaan |
|:---|:---|
| **`pandas`** | Manipulasi DataFrame: membaca CSV, merge, groupby, interpolasi temporal, feature engineering lag/rolling, export CSV. |
| **`numpy`** | Operasi numerik: noise stokastik Gaussian, masking boolean, perhitungan metrik MAPE manual, IQR bounds. |

### 6.3 Library Visualisasi

| Library | Alasan Penggunaan |
|:---|:---|
| **`matplotlib`** | Backend utama visualisasi statis: anomaly plot, comparison bar chart, residual distribution, SHAP summary. Diatur dengan `Agg` backend (non-interactive) agar bisa berjalan di server tanpa GUI. |
| **`plotly`** | Visualisasi interaktif di dashboard Streamlit: grafik time series dengan hover, bar chart SHAP lokal. |

### 6.4 Library Infrastruktur

| Library | Alasan Penggunaan |
|:---|:---|
| **`joblib`** | Serialisasi model (save/load) ke format `.joblib`. Lebih cepat dari `pickle` untuk objek numpy/sklearn besar. Dipakai untuk menyimpan `prophet_model`, `lgbm_model`, dan `iso_forest`. |
| **`streamlit`** | Framework dashboard web interaktif. Memungkinkan pembuatan UI prediksi tanpa menulis HTML/CSS/JS. Digunakan untuk `dashboard.py`. |
| **`holidays`** | Library Python yang menyediakan daftar hari libur nasional per negara. `holidays.Indonesia()` dipakai di dashboard untuk auto-deteksi apakah tanggal input adalah libur nasional. |

### 6.5 Library Utilitas

| Library | Alasan Penggunaan |
|:---|:---|
| **`os`**, **`json`**, **`glob`** | File system: path resolution, baca/tulis JSON, pencarian file wildcard. |
| **`warnings`** | Menyembunyikan warning yang tidak relevan (convergence, deprecation). |
| **`logging`** | Menekan output verbose dari `cmdstanpy` (backend Prophet) agar log tidak penuh. |
| **`itertools`** | Imported tapi tidak secara aktif digunakan di versi terkini. |

---

## 7. Arsitektur Model Hybrid (`hybrid_model.py`)

### 7.1 Alur Pipeline End-to-End

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: DATA INGESTION                           │
│   Baca dataset_daily_processed.csv → parse tanggal                     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: KNN IMPUTATION (ZERO-LEAKAGE)                │
│   Pengisian nilai kosong menggunakan Machine Learning KNNImputer yang  │
│   hanya dilatih pada data historis Train. Mengisi pola cuaca dan gaps  │
│   secara presisi.                                                      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      STEP 3: FEATURE ENGINEERING MODULE                 │
│   Ekspansi 18 fitur: Month, DayOfYear, IsHoliday, Avg_Temp, Rainfall,  │
│   Temp_Lag_1, Lags (1, 2, 7, 14, 30), Rolling_Means (7, 14, 30)        │
│   Hapus observasi yang mengandung NaN akibat Auto-Regressive cut-off   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 STEP 4: TRAIN/VAL/TEST SPLIT (70/15/15)                 │
│   Kronologis murni — TIDAK di-shuffle                                  │
│   Train: 70% data pertama                                              │
│   Validation: 15% berikutnya                                           │
│   Test: 15% terakhir                                                   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│          STEP 5: JOINT BAYESIAN OPTIMIZATION (OPTUNA)                   │
│   30 trials TPESampler — optimasi simultan:                            │
│   • contamination (Isolation Forest)                                    │
│   • changepoint_prior_scale & seasonality_prior_scale (Prophet)        │
│   • learning_rate, max_depth, num_leaves, subsample, colsample (LGBM) │
│   Objective: minimize MAE pada validation set                          │
│   (Lihat Bagian 9 untuk detail lengkap)                                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│            STEP 6: FINAL ARCHITECTURE TRAINING                          │
│                                                                         │
│   6a. ANOMALY DETECTION + IMPUTATION                                    │
│       IsolationForest + IQR → mendeteksi anomali                       │
│       Anomali TIDAK dihapus, melainkan DIIMPUTASI                       │
│       dengan rata-rata 7 hari terakhir data bersih                     │
│       (Lihat Bagian 8 untuk detail lengkap)                             │
│                                                                         │
│   6b. PROPHET TRAINING                                                  │
│       Input: data yang sudah diimputasi (gap-free)                     │
│       Konfigurasi: yearly + weekly seasonality, no daily               │
│       Output: Prophet_Pred untuk semua split                           │
│                                                                         │
│   6c. LIGHTGBM RESIDUAL TRAINING                                       │
│       Residual = Demand_MWh - Prophet_Pred                             │
│       LightGBM belajar memprediksi residual dari fitur eksogen         │
│       Early stopping: 50 rounds pada validation set                    │
│       Output: LGBM_Residual_Pred                                       │
│                                                                         │
│   6d. FINAL PREDICTION                                                  │
│       Final_Pred = Prophet_Pred + LGBM_Residual_Pred                   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              STEP 7: EVALUATION & EXPORT                                │
│   Metrik: MAE, RMSE, MAPE pada Validation & Test set                  │
│   Export: model .joblib, predictions CSV, visualisasi PNG              │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Mengapa Hybrid? Mengapa Bukan 1 Model Saja?

Konsumsi listrik di Indonesia dipengaruhi oleh **dua poros yang saling bertolak-belakang**:

1. **Poros Temporal (Waktu Kalender):** Listrik sangat patuh terhadap jam kerja, hari libur nasional (Idul Fitri, Natal), dan musim. **Prophet** secara spesifik didesain untuk menangkap pola ini.

2. **Poros Kausalitas (Lingkungan & Makro):** Suhu panas akibat El Niño memicu pemakaian AC ekstrem. Pertumbuhan GDP memicu pabrik baru. **LightGBM** menangkap variasi ini melalui fitur eksogen.

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
    changepoint_prior_scale=...,  # Dari Optuna — mengontrol fleksibilitas tren
    seasonality_prior_scale=...,  # Dari Optuna — mengontrol amplitudo musiman
)
```

**Input:** DataFrame 2 kolom: `ds` (tanggal) dan `y` (demand MWh)  
**Output:** Kolom `yhat` — prediksi baseline temporal

**Hyperparameter yang Di-tune Optuna:**
- `changepoint_prior_scale` (0.01–0.5): Semakin besar = tren lebih fleksibel (bisa berubah tajam). Semakin kecil = tren lebih mulus.
- `seasonality_prior_scale` (0.1–10.0): Semakin besar = amplitudo musiman lebih besar.

### 7.4 Detail Komponen LightGBM

**Library:** `lightgbm` (oleh Microsoft)

**Tugas:** Memprediksi `Prophet_Residual = Demand_MWh - Prophet_Pred`

**Konfigurasi:**
```python
LGBMRegressor(
    learning_rate=...,      # Dari Optuna (0.01–0.1)
    max_depth=...,          # Dari Optuna (3–7)
    num_leaves=...,         # Dari Optuna (15–63)
    subsample=...,          # Dari Optuna (0.7–1.0)
    colsample_bytree=...,   # Dari Optuna (0.7–1.0)
    n_estimators=5000,      # Jumlah pohon maksimal (dikontrol early stopping)
    random_state=42,        # Reproducibility
    n_jobs=-1,              # Gunakan semua CPU cores
    verbose=-1,             # Tidak ada output log
)
```

**Early Stopping:** Training berhenti jika selama 50 iterasi berturut-turut tidak ada peningkatan pada validation set. Ini mencegah **overfitting**.

**Fitur Input yang Digunakan:**
| Fitur | Mengapa Penting |
|:---|:---|
| `Day_of_Week` | Pola hari kerja vs weekend |
| `Is_Weekend` | Penurunan beban weekend |
| `Is_Holiday` | Penurunan beban hari libur |
| `Avg_Temp` | Suhu tinggi → AC → demand naik |
| `Rainfall` | Hujan → suhu turun → AC berkurang |
| `Lag_1` | Inersia konsumsi 1 hari sebelumnya |
| `Lag_7` | Pola mingguan berulang |
| `Lag_30` | Pola bulanan |
| `Rolling_7` | Tren halus jangka pendek |

---

## 8. Deteksi Anomali & Strategi Imputasi (Bukan Penghapusan)

### 8.1 Masalah Awal: Penghapusan Data Menciptakan Lubang

Implementasi awal mendeteksi anomali menggunakan IsolationForest + IQR, lalu **menghapus** baris yang dianggap anomali dari dataset training:

```python
# ❌ PENDEKATAN LAMA — Menghapus baris anomali
train_df_clean = train_df[(train_anomalies != -1) & (iqr_anomalies != -1)].copy()
```

**Masalah:** Menghapus baris menyebabkan **lubang (gaps)** dalam time series. Ini merusak fitur-fitur lag (`Lag_1`, `Lag_7`, `Lag_30`) dan `Rolling_7` yang bergantung pada data kontinu. Jika baris tanggal 15 Maret dihapus, maka `Lag_1` untuk tanggal 16 Maret menunjuk ke nilai yang salah.

### 8.2 Solusi: Imputasi dengan Rata-rata 7 Hari Terakhir

Pendekatan saat ini **tidak menghapus** baris apa pun. Sebaliknya, nilai anomali **diganti (diimputasi)** dengan rata-rata dari data bersih dalam jendela 7 hari ke belakang:

```python
# ✅ PENDEKATAN BARU — Imputasi, bukan penghapusan
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

### 8.3 Cara Kerja Imputasi Step-by-Step

1. **Deteksi Anomali (2 Metode Gabungan):**
   - **IsolationForest:** Algoritma berbasis pohon keputusan yang mengisolasi titik data. Data yang mudah diisolasi (sedikit split pohon) dianggap anomali (output: `-1`).
   - **IQR (Interquartile Range):** Metode statistik klasik. Data di luar rentang `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` dianggap outlier.
   - Sebuah titik dianggap anomali jika **salah satu** metode menandainya (OR logic).

2. **Iterasi Setiap Titik Anomali:** Untuk setiap data point yang ditandai anomali:
   - Cari 7 data point sebelumnya yang **bukan** anomali (clean data window)
   - Hitung rata-rata (mean) dari data bersih tersebut
   - Ganti nilai demand anomali dengan nilai rata-rata ini

3. **Fallback:** Jika dalam jendela 7 hari ke belakang tidak ada data bersih sama sekali (kasus sangat jarang misalnya 7+ anomali berturut-turut), digunakan rata-rata global seluruh data training.

### 8.4 Mengapa Rata-rata 7 Hari?

- **7 hari = 1 siklus mingguan.** Konsumsi listrik mengikuti pola mingguan yang kuat (hari kerja vs akhir pekan). Menggunakan rata-rata 1 minggu menangkap siklus ini.
- **Hanya data bersih** yang dimasukkan ke perhitungan mean — data anomali lain dalam jendela di-exclude agar noise tidak menyebar.
- **Menjaga integritas temporal** — tidak ada baris yang dihapus, sehingga fitur `Lag_1`, `Lag_7`, `Lag_30`, dan `Rolling_7` tetap menunjuk ke tanggal yang benar.

### 8.5 Perbandingan Pendekatan

| Aspek | Penghapusan (Lama) | Imputasi (Sekarang) |
|:---|:---|:---|
| Jumlah baris setelah cleaning | Berkurang | **Tetap sama** |
| Integritas fitur lag | ❌ Rusak (gaps) | ✅ Terjaga |
| Informasi temporal | ❌ Hilang | ✅ Terjaga |
| Pengaruh ke Prophet | Data pelatihan lebih sedikit | **Data pelatihan lengkap** |
| Kompleksitas | Sederhana (filter boolean) | Sedang (loop + lookback) |

---

## 9. Joint Bayesian Optimization (Optuna)

### 9.1 Mengapa Tidak Grid Search?

Grid search mengevaluasi **semua kombinasi** parameter secara brute-force. Dengan 8 hyperparameter yang perlu di-tune, grid search memerlukan ribuan evaluasi. Optuna menggunakan **TPESampler** (Tree-structured Parzen Estimator) yang secara cerdas memfokuskan pencarian di area yang menjanjikan — jauh lebih efisien.

### 9.2 Mengapa Optimasi "Joint" (Simultan)?

Hyperparameter dari 3 komponen **saling mempengaruhi**:
- Jika `contamination` tinggi → lebih banyak data diimputasi → Prophet melihat data berbeda → residual berubah → LightGBM harus beradaptasi
- Jika Prophet terlalu fleksibel (`changepoint_prior_scale` tinggi) → residual lebih kecil → LightGBM kurang berguna

Optimasi terpisah (tune Prophet sendiri, lalu tune LightGBM sendiri) **tidak menangkap interaksi ini**. Oleh karena itu, satu `objective()` function menjalankan **seluruh pipeline** (anomaly detection → Prophet → LightGBM) dan mengukur MAE akhir pada validation set.

### 9.3 Parameter yang Di-optimasi

| Parameter | Milik | Range | Skala | Dampak |
|:---|:---|:---|:---|:---|
| `contamination` | Isolation Forest | 0.001–0.05 | Log | Proporsi data yang dianggap anomali |
| `changepoint_prior_scale` | Prophet | 0.01–0.5 | Log | Fleksibilitas tren (tinggi = lebih responsif) |
| `seasonality_prior_scale` | Prophet | 0.1–10.0 | Log | Amplitudo efek musiman |
| `learning_rate` | LightGBM | 0.01–0.1 | Log | Kecepatan belajar (rendah = stabil, lambat) |
| `max_depth` | LightGBM | 3–7 | Linear | Kedalaman pohon (tinggi = lebih kompleks) |
| `num_leaves` | LightGBM | 15–63 | Linear | Jumlah daun pohon |
| `subsample` | LightGBM | 0.7–1.0 | Linear | Fraksi data per iterasi (regularisasi) |
| `colsample_bytree` | LightGBM | 0.7–1.0 | Linear | Fraksi fitur per iterasi (regularisasi) |

### 9.4 Caching & Retune Policy

Hasil Optuna disimpan di `Models/best_hybrid_params.json` dengan metadata:
- `last_tuned_at`: timestamp kapan terakhir di-tune
- `n_trials`: jumlah trial yang dijalankan
- `retune_every_days`: batas usia parameter (default: 30 hari)

**Retune otomatis terjadi jika:**
- File parameter tidak ditemukan
- Parameter sudah lebih tua dari `RETUNE_EVERY_DAYS` (30 hari)
- Environment variable `FORCE_RETUNE=1` diset manual

**Jika parameter masih segar:**
Optuna di-skip sepenuhnya dan parameter langsung di-load dari file — menghemat waktu eksekusi signifikan.

---

## 10. Evaluasi Kinerja Model

### 10.1 Data Split

| Split | Proporsi | Tujuan |
|:---|:---|:---|
| **Train** | 70% pertama | Melatih Prophet + LightGBM |
| **Validation** | 15% berikutnya | Early stopping LightGBM + Optuna objective |
| **Test** | 15% terakhir | Evaluasi akhir — TIDAK pernah dilihat saat training |

Split dilakukan **kronologis** (tanpa shuffle) — penting untuk time series agar model tidak "melihat masa depan".

### 10.2 Metrik yang Digunakan

| Metrik | Formula | Interpretasi |
|:---|:---|:---|
| **MAE** (Mean Absolute Error) | `mean(\|actual - predicted\|)` | Rata-rata selisih absolut dalam MWh. Mudah dipahami: "rata-rata model meleset X MWh per hari." |
| **RMSE** (Root Mean Squared Error) | `sqrt(mean((actual - predicted)²))` | Seperti MAE tapi memberi bobot lebih besar terhadap kesalahan besar. Sensitif terhadap outlier prediksi. |
| **MAPE** (Mean Absolute Percentage Error) | `mean(\|actual - predicted\| / actual) × 100%` | Persentase kesalahan relatif. Target industri: MAPE < 10%. |

### 10.3 Output Perbandingan

Pipeline mencetak tabel perbandingan **Prophet-Only vs Hybrid** pada kedua split (Validation & Test):

```
========================================================================
  MODEL COMPARISON: Prophet-Only vs Hybrid (Prophet + LightGBM)
========================================================================
Metric       | Prophet-Only (Test)    | Hybrid (Test)
------------------------------------------------------------------------
MAE          |         XX,XXX MWh    |         XX,XXX MWh
RMSE         |         XX,XXX MWh    |         XX,XXX MWh
MAPE         |            X.XX%      |            X.XX%
========================================================================
```

Serta persentase peningkatan Hybrid dibanding Prophet-Only.

---

## 11. Explainable AI (XAI) — SHAP Values

### 11.1 Apa itu SHAP?

**SHAP (SHapley Additive exPlanations)** berasal dari teori permainan kooperatif. SHAP mengukur **kontribusi marginal setiap fitur** terhadap prediksi model. Untuk setiap data point:

`Prediksi = Base Value + SHAP(Fitur₁) + SHAP(Fitur₂) + ... + SHAP(Fiturₙ)`

### 11.2 Mengapa Penting?

Tanpa XAI, model hanya berupa "black box" — kita tahu hasilnya tapi tidak tahu **mengapa**. Dengan SHAP:
- Kita bisa membuktikan bahwa suhu 35°C berkontribusi +12% beban
- Kita bisa menjelaskan bahwa hari libur menurunkan prediksi -15% beban
- Juri hackathon bisa **mempercayai** model karena transparan

### 11.3 Implementasi

**Global XAI** (di `hybrid_model.py`):
```python
explainer = shap.TreeExplainer(model_lgb)  # TreeExplainer khusus LGBM (cepat & exact)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, ...)  # Beeswarm plot semua fitur
```

**Local XAI** (di `dashboard.py`):
```python
explainer = shap.TreeExplainer(lgbm_model)
shap_vals = explainer.shap_values(custom_data)[0]  # SHAP untuk 1 data point
```

Dashboard menampilkan **Top 3 faktor penentu** dengan narasi bahasa Indonesia dan grafik dampak horizontal.

### 11.4 Cara Membaca SHAP Summary Plot

- **Sumbu Y:** Fitur diurutkan dari paling berpengaruh (atas) ke paling kecil (bawah)
- **Sumbu X:** Nilai SHAP — positif = menaikkan prediksi, negatif = menurunkan
- **Warna titik:** Merah = nilai fitur tinggi, biru = nilai fitur rendah

---

## 12. Dashboard Interaktif (`dashboard.py`)

### 12.1 Teknologi
- **Framework:** Streamlit
- **Visualisasi:** Plotly (interaktif), Matplotlib (statis/SHAP)
- **Deteksi Libur:** Library `holidays` (Indonesia)

### 12.2 Tab 1: Validasi Sistem (Historis)

**Tujuan:** Mengevaluasi apakah model bekerja stabil pada data harian historis yang sudah diketahui.

**Konten:**
- KPI cards: jumlah hari, observasi, anomali terdeteksi, MAE, RMSE
- Time series chart: Actual Demand vs Hybrid Prediction + marker anomali
- Prophet Components: dekomposisi tren + seasonality
- Global SHAP: feature importance seluruh dataset
- Penjelasan interpretatif dalam bahasa Indonesia

### 12.3 Tab 2: Future Forecaster & Local XAI

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

## 13. Artefak Output yang Dihasilkan

### 13.1 Model Files (`Models/`)

| File | Ukuran | Deskripsi |
|:---|:---|:---|
| `prophet_model.joblib` | ~144 KB | Model Prophet terlatih (tren + seasonality) |
| `lgbm_model.joblib` | ~154 KB | Model LightGBM terlatih (residual corrector) |
| `iso_forest.joblib` | ~4.5 MB | Model Isolation Forest (anomaly detector) |
| `best_hybrid_params.json` | ~620 B | Hyperparameter terbaik + metadata tuning |

### 13.2 Dataset Files (`Outputs/`)

| File | Deskripsi |
|:---|:---|
| `dataset_daily_processed.csv` | Dataset harian siap model (output `build_real_datasets.py`) |
| `dataset_monthly_processed.csv` | Dataset bulanan + makro (output `build_real_datasets.py`) |
| `dataset_daily_with_predictions.csv` | Dataset harian + kolom prediksi (output `hybrid_model.py`) |

### 13.3 Visualisasi (`Outputs/`)

| File | Konten |
|:---|:---|
| `fig0_anomalies_detected.png` | Scatter plot anomali terdeteksi pada data training |
| `fig1_actual_vs_predicted.png` | Time series full: Actual vs Prophet vs Hybrid + split zones |
| `fig2_model_comparison.png` | Bar chart MAE, RMSE, MAPE: Prophet-Only vs Hybrid |
| `fig3_residual_distribution.png` | Histogram residual Prophet vs Hybrid (tighter = better) |
| `fig4_shap_summary.png` | SHAP summary beeswarm plot (global feature importance) |
| `output_xai.png` (root dir) | SHAP plot duplicate untuk dashboard |

---

## 14. Cara Menjalankan Proyek

### 14.1 Install Dependencies

```bash
pip install -r Notebook/requirements.txt
```

Atau install secara manual:
```bash
pip install pandas numpy matplotlib prophet lightgbm scikit-learn shap optuna joblib streamlit plotly holidays
```

### 14.2 Bangun Dataset dari Raw Data

```bash
python Scripts/build_real_datasets.py
```

Output: `Outputs/dataset_daily_processed.csv` dan `Outputs/dataset_monthly_processed.csv`

### 14.3 Training Model Hybrid

```bash
python Scripts/hybrid_model.py
```

**Environment variables opsional:**
- `OPTUNA_TRIALS=50` — jumlah trial Optuna (default: 30)
- `RETUNE_EVERY_DAYS=7` — usia maksimal parameter sebelum retune (default: 30)
- `FORCE_RETUNE=1` — paksa retune meskipun parameter masih segar

### 14.4 Jalankan Dashboard

```bash
streamlit run dashboard.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`.

---

> *"Meramalkan masa depan tidak sekadar memutar rata-rata masa lalu, melainkan menyeimbangkan pola cuaca alam bebas dengan agenda kultural masyarakat di atasnya."*
