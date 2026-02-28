# Arsitektur Hybrid AI: Prophet (Meta) + LightGBM ⚡

Dokumen ini adalah ringkasan teknis tingkat tinggi _(High-Level Architecture)_ yang dirancang untuk menjelaskan keputusan strategis Model Machine Learning kita pada dewan juri Hackathon. Berbeda dengan pendekatan tradisional yang hanya menggunakan satu model konvensional, proyek ini menggunakan metodologi **Ensemble Hybrid SOTA (State-of-the-Art)**.

---

## 🏗️ Mengapa Harus Hybrid?

Tidak ada satu pun model AI di dunia yang sempurna di segala kondisi. Konsumsi listrik di Indonesia dipengaruhi oleh dua poros utama yang sifatnya saling bertolak-belakang:

1.  **Poros Temporal (Waktu Kalender)**: Listrik sangat patuh terhadap Jam Kerja, Hari Libur Nasional (Idul Fitri, Natal), dan Musim.
2.  **Poros Kausalitas (Lingkungan & Makro)**: Suhu panas akibat El Nino memicu pemakaian AC ekstrem. Pertumbuhan GDP memicu pabrik baru. Varian ini tidak mengikuti tanggal, melainkan mengikuti kondisi meteorologi dan ekonomi.

Pendekatan Tradisional (misal: XGBoost saja) akan kebingungan saat _"Hari Raya Idul Fitri jatuh di tanggal yang berbeda setiap tahunnya"_. Oleh karena itu, kita **memisahkan tugas (Decoupling)** ke dalam 3 Neural/Algorithmic Engine khusus:

---

## 🛠️ Trinitas Komponen Utama Kita

### 1. The Forecaster: Prophet (Oleh Meta/Facebook) 📈

**Tugas Utama**: Menangkap Pola Waktu, Tren Jangka Panjang, dan _Efek Kejut_ Hari Libur Nasional.

- **Cara Kerja**: Prophet memecah kurva konsumsi menjadi 3 fondasi: Garis Tren (Tahunan), _Seasonality_ (Siklus Mingguan/Bulanan), dan secara spesifik diberikan input tabel **Libur Nasional Indonesia riil (2018-2024)**.
- **Alasan Penggunaan**: Prophet secara _native_ dan mulus memahami bahwa jika besok adalah Lebaran pertama, maka konsumsi beban nasional dijamin akan terjun bebas—tidak peduli apa pun yang terjadi pada cuaca.

### 2. The Regressor: LightGBM (Oleh Microsoft) 🧠

**Tugas Utama**: Mempelajari Residu Matematika (Kesalahan/Sisa) dari apa yang tidak bisa diprediksi oleh Prophet (Kejadian Eksternal / _Exogenous Variables_).

- **Cara Kerja**: Setelah Prophet membuat kerangka dasarnya, nilai kekurangannya (_Residual_ = Aktual - Prophet) diserahkan pada algoritma pohon gradient berkinerja tinggi, LightGBM. LightGBM fokus mencocokkan anomali ini dengan parameter Cuaca (Tavg, Rainfall) dan Makroekonomi (GDP, Populasi).
- **Alasan Penggunaan**: Kinerjanya jauh lebih cepat untuk dieksekusi dibandingkan XGBoost di lingkungan lokal (Offline), lebih sensitif terhadap angka desimal cuaca ekstrem, dan kompatibel penuh dengan SHAP Values.

### 3. The Guardrail (Deteksi Anomali): Isolation Forest 🚨

**Tugas Utama**: Memberikan lapisan validasi tambahan. Model tidak hanya menebak _"Berapa megawatt esok hari?"_, tetapi juga mengeluarkan peringatan otomatis jika angka prediksi maupun aktual berada di zona _"Tidak Wajar"_.

- **Cara Kerja**: Model berbasis isolasi spasial pohon _decision tree_ yang mengklasifikasi apakah relasi antara Cuaca yang turun melawan Konsumsi Listrik hari itu termasuk sebagai `Inlier (Normal: 1)` atau `Outlier (Anomali: -1)`. Sangat krusial bagi operator PLN untuk deteksi pemadaman besar-besaran atau lonjakan ekstrem.

---

## ⚖️ Keunggulan Strategis Untuk Pitching Juri

1.  _Transparent / Explainable AI_ (XAI): Model Hybrid kita bisa direntangkan menggunakan **SHAP Summary Plot**. Alih-alih berupa _"Kotak Hitam" (Black-Box)_ seperti Autoencoder (Deep Learning), kita bisa membuktikan kepada juri: _"Di tanggal 12 Mei, suhu 35C berkontribusi +12% beban, sedangkan kelembapan menyumbang -2% beban"_.
2.  _Robustness_ terhadap _Shift_ Tanggal Hijriah/Sistem Penanggalan Lunar.
3.  Desainnya sangat efisien dan seluruh set model dilatih **100% Offline (Lokal)** menggunakan _Laptop_ standard, menjadikannya sistem yang berdaya tahan tinggi jika server pemerintahan/cloud putus akses.

---

`"Meramalkan masa depan tidak sekadar memutar rata-rata masa lalu, melainkan menyeimbangkan pola cuaca alam bebas dengan agenda kultural masyarakat di atasnya."`
