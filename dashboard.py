import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
import shap
import holidays
from prophet import Prophet

st.set_page_config(page_title="Electricity Demand XAI Dashboard", layout="wide")
st.title("⚡ Grid Demand Forecasting & Anomaly Detection")
st.markdown("*Hybrid Architecture: Prophet (Baseline) + LightGBM (Residuals) + Isolation Forest (Anomaly)*")

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'Models')
data_path = os.path.join(script_dir, 'Outputs', 'dataset_daily_with_predictions.csv')
xai_candidates = [
    os.path.join(script_dir, 'Outputs', 'fig4_shap_summary.png'),
    os.path.join(script_dir, 'output_xai.png'),
]

@st.cache_resource
def load_models(_prophet_mtime, _lgbm_mtime, _iso_mtime):
    prophet_model = joblib.load(os.path.join(models_dir, 'prophet_model.joblib'))
    lgbm_model = joblib.load(os.path.join(models_dir, 'lgbm_model.joblib'))
    iso_forest = joblib.load(os.path.join(models_dir, 'iso_forest.joblib'))
    return prophet_model, lgbm_model, iso_forest

@st.cache_data
def load_and_prep_data(_data_mtime):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    prophet_path = os.path.join(models_dir, 'prophet_model.joblib')
    lgbm_path = os.path.join(models_dir, 'lgbm_model.joblib')
    iso_path = os.path.join(models_dir, 'iso_forest.joblib')

    prophet_model, lgbm_model, iso_forest = load_models(
        os.path.getmtime(prophet_path),
        os.path.getmtime(lgbm_path),
        os.path.getmtime(iso_path),
    )
    data_mtime = os.path.getmtime(data_path)
    df = load_and_prep_data(data_mtime)
except FileNotFoundError as e:
    st.error(f"Missing required file: {e}. Please run `Scripts/hybrid_model.py` first.")
    st.stop()

st.caption(
    "Loaded artifacts: "
    f"data={pd.to_datetime(data_mtime, unit='s'):%Y-%m-%d %H:%M:%S}, "
    f"models={pd.to_datetime(max(os.path.getmtime(prophet_path), os.path.getmtime(lgbm_path), os.path.getmtime(iso_path)), unit='s'):%Y-%m-%d %H:%M:%S}"
)

# Features must match what the model was trained on (daily dataset only, no macro columns)
model_features = [
    'Day_of_Week', 'Is_Weekend', 'Is_Holiday',
    'Avg_Temp', 'Rainfall',
    'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_7',
]

# Only keep features that actually exist in the loaded data
available_features = [f for f in model_features if f in df.columns]

df_clean = df.dropna(subset=available_features + ['Demand_MWh', 'Hybrid_Prediction']).copy()
df_clean['Anomaly_Score'] = iso_forest.predict(df_clean[available_features])

feature_dict = {
    'Day_of_Week': 'Siklus Hari (Day_of_Week)',
    'Is_Weekend': 'Status Akhir Pekan (Is_Weekend)',
    'Is_Holiday': 'Status Hari Libur (Is_Holiday)',
    'Avg_Temp': 'Suhu Rata-Rata (Avg_Temp)',
    'Rainfall': 'Curah Hujan (Rainfall)',
    'Lag_1': 'Beban H-1 (Lag_1)',
    'Lag_7': 'Beban Minggu Lalu (Lag_7)',
    'Lag_30': 'Beban Bulan Lalu (Lag_30)',
    'Rolling_7': 'Tren Rata-rata 7 Hari (Rolling_7)',
}

feature_desc = {
    'Day_of_Week': 'Siklus operasional mingguan (Senin-Minggu) yang membedakan ritme aktivitas masyarakat.',
    'Is_Weekend': 'Mengurangi beban dasar secara signifikan karena perkantoran dan pasar tutup.',
    'Is_Holiday': 'Penurunan drastis beban listrik akibat libur nasional/tanggal merah secara serentak.',
    'Avg_Temp': 'Suhu harian; suhu tinggi memicu penggunaan AC (Cooling Demand) secara masif.',
    'Rainfall': 'Curah hujan memengaruhi visibilitas aktivitas luar ruang dan menekan suhu lokal.',
    'Lag_1': 'Inersia konsumsi dari 1 hari sebelumnya (Efek memori jangka pendek jaringan).',
    'Lag_7': 'Cerminan historis pola beban pada hari yang sama persis di minggu sebelumnya.',
    'Lag_30': 'Cerminan historis pola beban pada siklus bulanan sebelumnya.',
    'Rolling_7': 'Garis pergerakan rata-rata beban selama seminggu terakhir untuk melihat tren halus.',
}

tab1, tab2 = st.tabs(["📊 Validasi Sistem (Historis)", "🔮 Future Forecaster & Local XAI"])

with tab1:
    st.header("1. System Validation & Anomaly Detection")
    anomalies = df_clean[df_clean['Anomaly_Score'] == -1]

    total_rows = len(df_clean)
    anomaly_count = len(anomalies)
    anomaly_ratio = (anomaly_count / total_rows * 100) if total_rows else 0.0

    actual_hist = df_clean['Demand_MWh']
    hybrid_hist = df_clean['Hybrid_Prediction']
    abs_err_hybrid = (actual_hist - hybrid_hist).abs()
    mae_hybrid = abs_err_hybrid.mean()
    rmse_hybrid = ((actual_hist - hybrid_hist) ** 2).mean() ** 0.5
    non_zero_mask = actual_hist != 0
    mape_hybrid = ((abs_err_hybrid[non_zero_mask] / actual_hist[non_zero_mask]).mean() * 100) if non_zero_mask.any() else float('nan')

    has_prophet_pred = 'Prophet_Pred' in df_clean.columns
    if has_prophet_pred:
        prophet_hist = df_clean['Prophet_Pred']
        abs_err_prophet = (actual_hist - prophet_hist).abs()
        mae_prophet = abs_err_prophet.mean()
        rmse_prophet = ((actual_hist - prophet_hist) ** 2).mean() ** 0.5
        mape_prophet = ((abs_err_prophet[non_zero_mask] / actual_hist[non_zero_mask]).mean() * 100) if non_zero_mask.any() else float('nan')

        rmse_gain = ((rmse_prophet - rmse_hybrid) / rmse_prophet * 100) if rmse_prophet != 0 else 0.0
        mae_gain = ((mae_prophet - mae_hybrid) / mae_prophet * 100) if mae_prophet != 0 else 0.0
    else:
        rmse_gain = None
        mae_gain = None

    st.markdown(
        """
        **Tujuan panel historis ini** adalah mengevaluasi apakah sistem bekerja stabil pada data harian aktual.

        Arsitektur yang divalidasi:
        1. **Prophet** memodelkan pola utama jangka panjang (tren + musiman).
        2. **LightGBM** mengoreksi residual Prophet menggunakan variabel eksogen/cuaca + fitur lag.
        3. **Isolation Forest** menandai hari dengan pola fitur yang sangat tidak lazim sebagai anomali.

        Hasil yang perlu diperhatikan:
        - Seberapa dekat garis prediksi ke demand aktual.
        - Seberapa sering muncul anomali relatif terhadap total hari.
        - Seberapa besar error historis (MAE, RMSE, MAPE).
        """
    )

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
    kpi_1.metric("Periode Data", f"{df_clean['Date'].min():%Y-%m-%d} s/d {df_clean['Date'].max():%Y-%m-%d}")
    kpi_2.metric("Jumlah Observasi", f"{total_rows:,}")
    kpi_3.metric("Anomali Terdeteksi", f"{anomaly_count:,}", delta=f"{anomaly_ratio:.2f}% dari data")
    kpi_4.metric("MAE Hybrid", f"{mae_hybrid:,.0f} MWh")
    kpi_5.metric("RMSE Hybrid", f"{rmse_hybrid:,.0f} MWh")

    st.caption(
        f"MAPE Hybrid historis: {mape_hybrid:.2f}%"
        + (
            f" | Gain vs Prophet: MAE {mae_gain:.2f}%, RMSE {rmse_gain:.2f}%"
            if has_prophet_pred else
            " | Kolom Prophet_Pred tidak tersedia, sehingga perbandingan baseline Prophet tidak ditampilkan."
        )
    )

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=df_clean['Date'], y=df_clean['Demand_MWh'], mode='lines', name='Actual Demand', line=dict(color='blue')))
    fig_main.add_trace(go.Scatter(x=df_clean['Date'], y=df_clean['Hybrid_Prediction'], mode='lines', name='Hybrid Prediction', line=dict(color='orange', dash='dash')))
    fig_main.add_trace(go.Scatter(
        x=anomalies['Date'], y=anomalies['Demand_MWh'], mode='markers', name='Grid Anomaly Detected',
        marker=dict(color='red', size=10, symbol='x', line=dict(color='black', width=1)),
        hovertemplate="<b>Date:</b> %{x}<br><b>Demand:</b> %{y}<extra>Anomaly</extra>"
    ))
    fig_main.update_layout(height=400, template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig_main, use_container_width=True)

    with st.expander("Cara membaca grafik utama (Actual vs Hybrid + Anomali)", expanded=True):
        st.markdown(
            """
            1. **Garis biru (Actual Demand)** adalah beban listrik aktual harian.
            2. **Garis oranye putus-putus (Hybrid Prediction)** adalah hasil prediksi sistem hybrid.
            3. **Marker merah (Grid Anomaly Detected)** adalah hari yang oleh Isolation Forest dianggap tidak normal secara pola fitur.

            Interpretasi praktis:
            - Jika garis oranye menempel pada garis biru, model memiliki error rendah.
            - Jika marker merah muncul berkelompok, kemungkinan ada kejadian sistemik (event besar, gangguan, cuaca ekstrem, atau perubahan pola konsumsi).
            - Tidak semua anomali adalah kesalahan data; sebagian bisa menjadi sinyal operasional penting untuk investigasi.
            """
        )
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Prophet Components")
        df_prophet = df_clean[['Date']].rename(columns={'Date': 'ds'})
        forecast = prophet_model.predict(df_prophet)
        fig_prophet = prophet_model.plot_components(forecast)
        st.pyplot(fig_prophet)
        with st.expander("Penjelasan Prophet Components", expanded=False):
            st.markdown(
                """
                Plot komponen Prophet menjelaskan dari mana baseline terbentuk:
                1. **Trend**: arah pertumbuhan/penurunan demand jangka panjang.
                2. **Weekly Seasonality**: pola berulang per hari dalam seminggu (hari kerja vs akhir pekan).
                3. **Yearly Seasonality**: pola musiman tahunan.

                Komponen ini belum memuat koreksi residual LightGBM, sehingga dipakai sebagai baseline struktural.
                """
            )
    with col_b:
        st.subheader("Global Feature Impact")
        xai_path = next((path for path in xai_candidates if os.path.exists(path)), None)
        if xai_path:
            st.image(xai_path, use_container_width=True)
            with st.expander("Penjelasan Global Feature Impact (SHAP)", expanded=False):
                st.markdown(
                    """
                    Grafik SHAP global merangkum kontribusi fitur terhadap koreksi residual LightGBM.

                    Cara membaca:
                    1. **Sumbu Y**: urutan fitur dari paling berpengaruh ke paling kecil.
                    2. **Sumbu X (SHAP value)**:
                       - Nilai positif mendorong prediksi naik.
                       - Nilai negatif mendorong prediksi turun.
                    3. **Warna titik**:
                       - Warna lebih "panas" menandakan nilai fitur tinggi.
                       - Warna lebih "dingin" menandakan nilai fitur rendah.

                    Nilai bisnis:
                    - Mengidentifikasi pendorong demand utama (mis. suhu, lag konsumsi).
                    - Membantu validasi bahwa perilaku model konsisten dengan logika sistem tenaga.
                    """
                )
        else:
            st.info("No XAI figure found. Run Scripts/hybrid_model.py to generate Outputs/fig4_shap_summary.png.")

    with st.expander("Kesimpulan Validasi Historis", expanded=True):
        st.markdown(
            """
            Checklist cepat sebelum model dipakai operasional:
            1. **Akurasi**: MAE/RMSE/MAPE berada dalam batas toleransi operasional.
            2. **Stabilitas pola**: prediksi tidak sering melenceng besar dari aktual.
            3. **Anomali**: rasio anomali wajar dan dapat dijelaskan secara domain.
            4. **Interpretabilitas**: fitur dominan SHAP masuk akal secara fisik/operasional.

            Jika salah satu poin di atas tidak terpenuhi, lakukan retraining atau audit data (khususnya fitur lag, cuaca, dan label libur).
            """
        )

with tab2:
    st.header("Prediksi Masa Depan & Penjelasan Keputusan (Local XAI)")
    st.markdown("Masukkan input kondisi lingkungan. Sistem otomatis mendeteksi hari libur nasional Indonesia untuk memperkuat presisi prediksi.")
    
    col1, col2 = st.columns(2)
    
    # Inisialisasi kalender Indonesia
    id_holidays = holidays.Indonesia()

    with col1:
        target_date = st.date_input("Pilih Tanggal Prediksi", value=pd.to_datetime("2026-08-17"))
        
        # Auto-deteksi Libur Nasional
        is_auto_holiday = target_date in id_holidays
        holiday_name = id_holidays.get(target_date) if is_auto_holiday else "Hari Kerja/Biasa (Bukan Libur Nasional)"
        
        st.info(f"📅 **Deteksi Sistem Kalender:** {holiday_name}")
        
        # Opsi override untuk What-If Scenario
        is_holiday_override = st.checkbox("Paksa hitung sebagai Hari Libur Nasional (Simulasi Manual)", value=is_auto_holiday)
        is_holiday = 1 if is_holiday_override else 0

    with col2:
        avg_temp = st.number_input("Input Suhu Udara (°C)", min_value=15.0, max_value=45.0, value=28.0, step=0.1, format="%.1f")
        rainfall = st.number_input("Input Curah Hujan (mm)", min_value=0.0, max_value=300.0, value=5.0, step=1.0, format="%.1f")
        
    if st.button("Jalankan Prediksi AI 🚀", use_container_width=True):
        day_of_week = target_date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        last_known_data = df_clean.iloc[-1]
        
        custom_data = pd.DataFrame({
            'Day_of_Week': [day_of_week],
            'Is_Weekend': [is_weekend],
            'Is_Holiday': [is_holiday],
            'Avg_Temp': [avg_temp],
            'Rainfall': [rainfall],
            'Lag_1': [last_known_data['Demand_MWh']], 
            'Lag_7': [last_known_data.get('Lag_7', last_known_data['Demand_MWh'])], 
            'Lag_30': [last_known_data.get('Lag_30', last_known_data['Demand_MWh'])],
            'Rolling_7': [last_known_data.get('Rolling_7', last_known_data['Demand_MWh'])],
        }).astype(float)
        
        prophet_df = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
        base_demand = prophet_model.predict(prophet_df)['yhat'].values[0]
        weather_residual = lgbm_model.predict(custom_data)[0]
        final_demand = base_demand + weather_residual
        anomaly_flag = iso_forest.predict(custom_data)[0]
        
        explainer = shap.TreeExplainer(lgbm_model)
        shap_vals = explainer.shap_values(custom_data)[0]
        
        feature_impacts = list(zip(available_features, shap_vals))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_3_impacts = feature_impacts[:3]
        
        st.divider()
        st.success(f"### ⚡ Hasil Prediksi: {final_demand:,.0f} MWh")
        st.write(f"*(Baseline Tren: {base_demand:,.0f} MWh | Penyesuaian AI: {weather_residual:+,.0f} MWh)*")
        
        if anomaly_flag == -1:
            st.error("🚨 **Peringatan:** Kombinasi fitur pada hari ini terdeteksi sebagai Anomali Jaringan ekstrem (Black-Swan Event).")
            
        st.subheader("1. Narasi Eksekutif (Top 3 Faktor Penentu)")
        status_arah = "TINGGI" if final_demand > base_demand else "RENDAH"
        st.markdown(f"Pada tanggal **{target_date.strftime('%d %B %Y')}**, prediksi model lebih **{status_arah}** dibandingkan tren baseline. 3 pendorong utamanya adalah:")
        
        for i, (feat, impact) in enumerate(top_3_impacts):
            direction_text = "Meningkatkan Prediksi (Positif 📈)" if impact > 0 else "Menurunkan Prediksi (Negatif 📉)"
            st.markdown(f"**{i+1}. {feature_dict.get(feat, feat)}** ➔ {direction_text} berdampak **{abs(impact):,.0f} MWh**.")

        st.subheader("2. Grafik Analisis Dampak (Local SHAP Plot)")
        plot_data = sorted(feature_impacts, key=lambda x: x[1])
        colors = ['#EF553B' if val < 0 else '#00CC96' for feat, val in plot_data]
        
        fig_local = go.Figure(go.Bar(
            x=[val for feat, val in plot_data],
            y=[feature_dict.get(feat, feat) for feat, val in plot_data],
            orientation='h',
            marker_color=colors,
            text=[f"{val:+,.0f}" for feat, val in plot_data],
            textposition='auto'
        ))
        fig_local.update_layout(
            title="Seberapa Besar Setiap Variabel Mengubah Prediksi Hari Ini?",
            xaxis_title="Dampak terhadap Prediksi (MWh)",
            template='plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_local, use_container_width=True)

        st.subheader("3. Rincian Lengkap Variabel, Nilai Input, dan Artinya")
        explanation_data = []
        for feat, impact in feature_impacts:
            actual_val = custom_data[feat].values[0]
            
            if feat == 'Avg_Temp': val_str = f"{actual_val:.1f} °C"
            elif feat in ['Is_Holiday', 'Is_Weekend']: val_str = "Ya" if actual_val == 1 else "Tidak"
            elif feat == 'Rainfall': val_str = f"{actual_val:.1f} mm"
            elif feat == 'Day_of_Week': 
                days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
                val_str = days[int(actual_val)]
            else: val_str = f"{actual_val:,.0f}"
            
            explanation_data.append({
                "Variabel": feature_dict.get(feat, feat),
                "Nilai Aktual Hari Ini": val_str,
                "Dampak AI (MWh)": f"{impact:+,.0f}",
                "Arti & Efek Fisiknya": feature_desc.get(feat, "")
            })
            
        st.table(pd.DataFrame(explanation_data))