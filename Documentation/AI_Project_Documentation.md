# Dokumentasi Utama: Peramalan Permintaan Listrik AI Hibrida (Hybrid AI Electricity Demand Forecasting)

## 1. Ringkasan Eksekutif
Proyek ini menguraikan pipeline prediktif mutakhir untuk analitik permintaan listrik nasional menggunakan **Arsitektur Hibrida Prophet + LightGBM**. Metodologi inti menghindari tebakan teoritis dengan membungkus keseluruhan arsitektur di dalam **Optuna Bayesian Optimization Engine**, mengintegrasikan tren makroekonomi dengan pelacakan fluktuasi mikro secara sempurna, sambil tetap dilindungi secara matematis oleh filter anomali **Isolation Forest**.

---

## 2. Prapemrosesan Dataset & Validasi Fisika
Memprediksi jaringan listrik nasional yang ekstrem memerlukan perlindungan validitas statistik dataset sejak awal.

### Perbandingan Data Mentah Awal vs. Dataset Prapemrosesan Akhir
Sebelum dataset dapat digunakan oleh arsitektur, dataset ini mengalami restrukturisasi besar-besaran untuk menjamin bahwa AI mempelajari tren fisik daripada menghafal rumus matematika yang rusak.

**1. Keadaan Awal (Data Mentah):**
- **Ketersebaran (Sparsity):** Dataset terisolasi yang tersebar di tabel cuaca Kaggle (`climate_data.csv`), pelacak hari libur API open-source, dan output total statistik tahunan BPS.
- **Fitur yang Hilang:** Tidak memiliki konteks untuk "cuaca kemarin" atau "tren minggu lalu".
- **Risiko Target Leakage (Kebocoran Target):** Karena permintaan tahunan BPS dipecah secara retrospektif menjadi 365 bagian harian melalui rumus matematika deterministik, model berbasis pohon (tree-based) dapat dengan mudah merekayasa balik rumus pasti tersebut tanpa mempelajari apa pun tentang jaringan fisik kelistrikan.

**2. Keadaan Prapemrosesan (Dataset yang Direkayasa):**
Untuk meresolusi masalah ini, `dataset_daily_processed.csv` dihasilkan menggunakan teknik berikut:
- **Injeksi Noise Gaussian Stokastik:** Selama kompilasi, lapisan noise multiplikatif (±5%) dan aditif (~3%) disisipkan secara paksa ke dalam nilai `Demand_MWh`. *Alasan:* Hal ini bertindak sebagai perisai fisik terhadap Target Leakage. Ia mensimulasikan kekacauan matematis mutlak dari dunia nyata—pemadaman listrik yang tak terduga, kecelakaan pabrik, atau perilaku iklim mikro yang tidak sesuai musim. Hal ini memaksa algoritma ML untuk belajar *generalisasi luas* alih-alih menghafal rute (route-memorization).
- **Rekayasa Fitur Autoregresif:** Kami memetakan kolom baru bernama `Lag_1` (permintaan kemarin), `Lag_7` (hari yang sama minggu lalu), dan rata-rata bergerak kontinu `Rolling_7`. *Alasan:* Jaringan saraf (neural nets) dan pohon penambah (boost-trees) secara inheren tidak memahami "waktu". Membuat fitur ini memberi algoritma memori jangka pendek numerik, yang memungkinkannya membaca momentum secara matematis.
- **Penandaan Biner (Binary Flagging):** Titik data dipetakan ke dalam `Is_Weekend` dan `Is_Holiday`. *Alasan:* Jaringan listrik industri berperilaku sangat berbeda pada hari libur nasional dibandingkan hari kerja biasa. Ini membekali algoritma dengan konteks kalender deterministik.

### Pemisahan Kronologis (The Tri-Split)
Arsitektur ini sepenuhnya menghindari pemisahan Train/Test secara acak (`shuffle=True`). Waktu hanya bergerak ke satu arah. Matriks secara kronologis dipecah menjadi **70% Pelatihan / 15% Validasi / 15% Pengujian**. Langkah ini benar-benar mengkarantina model, mencegahnya menggunakan titik data apa pun dari 'masa depan' fisik saat mencoba memahami 'masa lalu'.

---

## 3. Ekstraksi Anomali Matematis (Isolation Forest)
Detektor anomali standar (seperti ambang batas z-score 3-Sigma) mengalami kesulitan dalam perkiraan listrik karena pergeseran musiman majemuk yang masif (misalnya, puncak permintaan musim panas vs penurunan musim dingin). Sangatlah mudah bagi matematika standar untuk secara keliru menandai "Puncak Musim Panas" sebagai sebuah anomali.

- **Metodologi:** Kami menggunakan `Isolation Forest` dengan `n_estimators=300` yang sangat ditingkatkan. Ini mengukur secara persis seberapa mudah aliran logika permintaan harian spesifik dapat diisolasi secara struktural dari keseluruhan dataset berdasarkan lintasan internal algoritmanya.
- **Perlindungan Data:** Isolation Forest dipicu untuk `.fit()` secara ketat *hanya* pada dataset Pelatihan 70% untuk memastikan bahwa definisi "Pola Bersih" tidak bocor ke dalam pengujian di masa depan.
- **Kartografi Visual:** Karena visualisasi pencilan (outliers) diperlukan untuk memvalidasi filter algoritmik, pipeline ini menghasilkan `fig0_anomalies_detected.png`, secara aktif menandai titik ambang batas matematis peluruhan sebagai koordinat fisik yang dipetakan kembali ke dalam timeline.

---

## 4. Arsitektur Hibrida Tiga Lapis (The Tri-Layer Hybrid Architecture)
Kegagalan utama dari prakiraan algoritma tunggal (single-model) mana pun adalah ketidakmampuannya untuk mendeteksi secara simultan keseimbangan tren ekonomi makro yang sangat besar seraya secara cerdas menangkap fluktuasi mikro yang tidak terduga (seperti perubahan cuaca mendadak atau hari libur mendadak). Kami memanfaatkan dua algoritma berbeda yang dijembatani oleh ekstraksi Residual:

### Lapisan 1: Prophet (Matriks Tren Dasar)
Prophet bertindak sebagai tulang punggung struktural dari logika tersebut. Ia mengabaikan volatilitas harian dan memetakan secara eksklusif lintasan fisik penggunaan listrik skala tahunan, mengatur persis pergeseran struktural (`changepoint_prior_scale`) serta mengakomodasi siklus kebiasaan penduduk selama rentang tujuh hari (`seasonality_prior_scale`).

### Lapisan 2: Ekstraksi Residual
Alih-alih langsung dijadikan hasil akhir, timeline perkiraan yang dihasilkan oleh Prophet secara matematis dikurangi dengan metrik kenyataan sebenarnya (`Demand_MWh - Prophet_Pred`). Selisih kesalahan matematis pasti (yakni the **Residuals**) yang ditorehkan oleh Prophet kini berubah wujud menjadi target murni/sasaran satu-satunya untuk pembelajaran lapisan berikutnya.

### Lapisan 3: LightGBM (Gradient Boosting)
LightGBM *tidak* berusaha memprediksi listrik. Model ini hanya disuapi murni dengan fitur eksogen (Cuaca, Hari Libur, Lags/Tren Mundur) dan dilatih secara eksklusif untuk mendeteksi *di mana persisnya Prophet akan meleset*. Dengan mencoba memprediksi persis skala ukuran dan bentuk celah logika `Prophet_Residuals` tersebut, LightGBM bertindak sebagai dewa pengoreksi mikro (ultimate micro-corrector).
- **Rumus Output Akhir:** `Final_Prediction = Prophet Base + LightGBM Residual Prediction`.

---

## 5. Joint Bayesian Optimization (Optuna)
Alih-alih membuang waktu mengandalkan pencarian grid (grid search) primitif atau sistem tebakan berulang buta, perombakan masif arsitektur ini membungkus segalanya secara tunggal di dalam mesin pencarian probabilistik **Optuna TPESampler**.

- **Sinergi Bersama (Joint Synergies):** Optuna secara matematis merancang perbatasan optimasi yang beradaptasi secara tersinkronisasi murni di sekeliling 3 model tersebut secara serempak sekaligus (`contamination` bagi sistem anomali Forest + skala `changepoint/seasonality` perbekalan Prophet + titik persimpangan `Learning Rate / Max Depth` perbekalan LightGBM). Model ini menyimulasikan 30 titik pencarian raksasa mandiri. Jika model mengetahui bahwa Ambang Anomali 2% meledak parah dan gagal apabila diduetkan dengan Learning Rate yang dalam, model *akan benar-benar membaca hal itu, memperlajarinya, dan menyimpannya*, untuk memastikan ke depannya mesin tersebut menghindari kombinasi keliru secara akurat demi menghantam puncak optimal dari ketiga kolaborasi tersebut.
- **Replikabilitas Deterministik:** TPESampler mengakar keras pada setelan penguncian matematika acak status (`seed=0`) tunggal. Metode ini mencegah bias uji coba dan menjamin 100% replikabilitas deterministik ketat taraf akademis formal; hal ini membuat siapapun _data scientist_ profesional independen yang mengetes model source-code lokal ini ke depannya akan mengekstrak susunan sinergis sempurna tanpa keraguan kemelesetan hasil evaluasi.
- **Perlindungan Overfeeding (Overfitting):** Saat Optuna bekerja agresif memonitor persentase timeline `val_df`, LightGBM dirancang dengan protokol pemutus otomatis pengereman keras (`early_stopping`). Andai kata model LightGBM terpantau malah secara buta menghafalkan huruf demi huruf lintasan evaluasi alih-alih sanggup merumuskan makna di balik pergeseran titik tersebut bagi data eksternal, jaringan kelistrikan latihan akan langsung diblokir dan pemrosesan dimatikan darurat.

---

## 6. Metrik Evaluasi & Perbandingan Baseline
Arsitektur dinilai secara eksklusif lewat karantina eksekusi sepihak pada seksi Test dataset acak 15%.

1. **Root Mean Squared Error (RMSE):** Berfungsi mutlak untuk menjatuhkan pinalti pada kemelesetan deteksi yang mematikan. Karena rumus matematis model RMSE pada dasarnya memangkat-duakan nilai luput/error model, kemelesetan sedikit saja atas perkiraan arah prediksi akan melempar nilai denda metrik secara fatal menjadi sangat tinggi.
2. **Mean Absolute Error (MAE):** Mengkalkulasi jarak mutlak nilai yang didapatkan versus realita titik target sempurnanya.
3. **Mean Absolute Percentage Error (MAPE):** Evaluator Standar Skala Internasional. Fokus mengalkulasi wujud error murni berbasis skala persentase. Ketimbang mendikte jarak blok, MAPE mengecam apakah tingkat error 50 Megawatt itu sudah sangat membahayakan kapasitas listrik kota, atau angka 50 tersebut ternyata hanyalah sekadar fraksi debu jika dibandingkan total 50.000 kapasitas listrik.

### Perbandingan Pra dan Pasca Penyetelan (Pre and Post Comparison)
`fig2_model_comparison.png` memvisualisasikan residual dari algoritma dasar **Prophet Baseline** dan menghantamkannya secara konfrontasi langsung melawan gabungan **Hybrid AI**. Di seantero chart performa grafis pasca modifikasi, hasil pencerahan LightGBM ini dibuktikan telah secara gamblang me-"netralisir" lonjakan cacat data anomali ekstrem yang mendustai baseline, plus menyuntikkan teknik "compression" (merapatkan semua data error menuju target pusat), mengamankan klaim absolut supremasi peramalan kelistrikan dengan meratakan bias persebaran sejauh mungkin dengan titik NOL mutlak.
