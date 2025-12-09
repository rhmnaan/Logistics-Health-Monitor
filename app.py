import streamlit as st
import plotly.express as px
import pandas as pd
import joblib 
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os 
import numpy as np 

# ---------------------------------------------------------
# LOADING ALL MODELS & DATA
# ---------------------------------------------------------
@st.cache_resource
def load_all_resources():
    
    model_anomali = None
    model_klasifikasi = None
    scaler_klasifikasi = None
    model_prediksi_waktu = None

    df_anomali = pd.DataFrame({'Is_Anomaly': ['No', 'Yes', 'No'],
                               'Anomaly_Score': [0.1, -0.5, 0.2],
                               'Sales': [100, 20, 80],
                               'Order_Profit': [20, -5, 15]})

    df_klasifikasi = pd.DataFrame({'Proporsi': ['{"Risk_Level": ["Low","High"], "Count": [80,20]}'],
                                   'Top_Kategori': ['{"Category":["Furniture","Tech"],"Risk_Count":[12,7]}']})

    df_importance = pd.DataFrame({'Fitur': ['Mode Pengiriman', 'Jarak', 'Tipe Barang'],
                                  'Importance': [0.6, 0.4, 0.2]})

    kolom_klasifikasi = {}
    feature_waktu = {}

    def safe_load(path, method, default_value, display_name):
        if not os.path.exists(path):
            st.sidebar.warning(f"⚠️ FILE '{path}' tidak ditemukan ({display_name}).")
            return default_value
        try:
            data = method(path)
            st.sidebar.success(f"Berhasil memuat {display_name}")
            return data
        except Exception as e:
            st.sidebar.error(f"Error memuat {display_name}: {e}")
            return default_value

    st.sidebar.subheader("Status Loading Model & Data")

    # Load Model
    model_anomali = safe_load('model_isolation_forest.pkl', joblib.load, None, "Model Anomali")
    model_klasifikasi = safe_load('model_klasifikasi.pkl', joblib.load, None, "Model Klasifikasi")
    scaler_klasifikasi = safe_load('scaler_klasifikasi.pkl', joblib.load, None, "Scaler Klasifikasi")
    model_prediksi_waktu = safe_load('model_waktu_prediksi.pkl', joblib.load, None, "Model Prediksi Waktu")

    # Load CSV
    df_anomali_loaded = safe_load('hasil_deteksi_anomali.csv',
                                  lambda f: pd.read_csv(f).head(100),
                                  df_anomali, "Data Anomali")

    df_klasifikasi_loaded = safe_load('insight_klasifikasi.csv',
                                      lambda f: pd.read_csv(f).head(100),
                                      df_klasifikasi, "Data Insight Klasifikasi")

    df_importance_loaded = safe_load('feature_importance.csv',
                                     lambda f: pd.read_csv(f).head(100),
                                     df_importance, "Feature Importance")

    # Load JSON
    kolom_klasifikasi = safe_load('kolom_klasifikasi.json',
                                  lambda f: json.load(open(f)),
                                  {}, "Metadata Klasifikasi")

    feature_waktu = safe_load('feature_waktu.json',
                              lambda f: json.load(open(f)),
                              {}, "Metadata Prediksi Waktu")

    if isinstance(df_anomali_loaded, pd.DataFrame): df_anomali = df_anomali_loaded
    if isinstance(df_klasifikasi_loaded, pd.DataFrame): df_klasifikasi = df_klasifikasi_loaded
    if isinstance(df_importance_loaded, pd.DataFrame): df_importance = df_importance_loaded

    return model_anomali, model_klasifikasi, scaler_klasifikasi, model_prediksi_waktu, \
           df_anomali, df_klasifikasi, df_importance, kolom_klasifikasi, feature_waktu


# LOAD
(
    MODEL_ANOMALI,
    MODEL_KLASIFIKASI,
    SCALER_KLASIFIKASI,
    MODEL_PREDIKSI,
    DF_ANOMALI,
    DF_KLASIFIKASI,
    DF_IMPORTANCE,
    META_KLASIFIKASI,
    META_WAKTU
) = load_all_resources()


# ---------------------------------------------------------
# UI CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Logistics Health Monitor", layout="wide")
st.title("🚢 The Logistics Health Monitor: Supply Chain Control Tower")
st.markdown("Dashboard terintegrasi untuk pengambilan keputusan cepat manajer gudang/logistik.")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🚨 Early Warning System (Anomali)",
    "📈 Risk Profiling (Klasifikasi)",
    "⚙️ Optimization & Strategy (Prediksi Waktu)"
])


# ---------------------------------------------------------
# TAB 1 – DETEKSI ANOMALI
# ---------------------------------------------------------
with tab1:

    st.header("1. Early Warning System")
    anomalies_count = 0

    if 'Is_Anomaly' in DF_ANOMALI.columns:
        anomalies_count = DF_ANOMALI[DF_ANOMALI['Is_Anomaly'].astype(str).str.lower() == 'yes'].shape[0]

    if anomalies_count > 0:
        st.error(f"⚠️ Terdeteksi {anomalies_count} anomali operasional.")
    else:
        st.success("Tidak ada anomali terdeteksi.")

    st.subheader("Visualisasi Sales & Profit")
    try:
        st.line_chart(DF_ANOMALI[['Sales', 'Order_Profit']])
    except:
        st.warning("Kolom Sales/Order_Profit tidak ditemukan.")

    st.subheader("Detail Anomali")
    if 'Is_Anomaly' in DF_ANOMALI.columns:
        df_f = DF_ANOMALI[DF_ANOMALI['Is_Anomaly'].astype(str).str.lower() == 'yes']
        st.dataframe(df_f.head(50))
    else:
        st.dataframe(DF_ANOMALI.head())


# ---------------------------------------------------------
# TAB 2 – KLASIFIKASI & PREDIKSI RISIKO
# ---------------------------------------------------------
with tab2:

    st.header("🚦 Klasifikasi Risiko Keterlambatan Pesanan")

    # ============================================================
    # 1️⃣ PROPORSI RISIKO
    # ============================================================
    st.subheader("📊 Proporsi Risiko Keterlambatan")

    try:
        raw = DF_KLASIFIKASI["Proporsi"].iloc[0]
        proporsi = json.loads(raw)
        df_proporsi = pd.DataFrame(proporsi)

        fig = px.pie(df_proporsi, names="Risk_Level", values="Count", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

        st.bar_chart(df_proporsi.set_index("Risk_Level"))

    except Exception as e:
        st.warning(f"Gagal visualisasi Proporsi Risiko. Error: {e}")


    # ============================================================
    # 3️⃣ VISUALISASI REGION BERISIKO
    # ============================================================
    st.subheader("🌍 Visualisasi Region Paling Berisiko")

    try:
        model_features = list(MODEL_KLASIFIKASI.feature_names_in_)
        region_cols = [col for col in model_features if col.startswith("order_region_")]

        if len(region_cols) == 0:
            st.warning("❌ Model tidak memiliki fitur region order_region_*.")        

        else:
            st.info("🔄 Menghitung ulang risiko tiap region berdasarkan MODEL KLASIFIKASI.")

            hasil = []

            dataset_cols = DF_KLASIFIKASI.columns.tolist()
            dataset_has_region = any(col in dataset_cols for col in region_cols)

            for col in region_cols:
                region_name = col.replace("order_region_", "")

                if dataset_has_region and col in DF_KLASIFIKASI.columns:
                    subset = DF_KLASIFIKASI[DF_KLASIFIKASI[col] == 1]
                    if len(subset) > 0:
                        X = subset[model_features]
                        pred = MODEL_KLASIFIKASI.predict_proba(X)[:, 1]
                        hasil.append([region_name, pred.mean()])
                        continue

                X_zero = pd.DataFrame([{f: 0 for f in model_features}])
                X_zero[col] = 1  

                num_cols = [
                    c for c in [
                        "days_for_shipment_scheduled",
                        "days_for_shipping_real",
                        "shipment_delay"
                    ] if c in model_features
                ]

                if len(num_cols) > 0:
                    try:
                        X_zero[num_cols] = SCALER_KLASIFIKASI.transform(X_zero[num_cols])
                    except:
                        pass

                pred = MODEL_KLASIFIKASI.predict_proba(X_zero)[0][1]
                hasil.append([region_name, pred])

            region_risk = pd.DataFrame(hasil, columns=["Region", "Risk_Ratio"])
            region_risk["Risk_Percent"] = (region_risk["Risk_Ratio"] * 100).round(2)
            region_risk = region_risk.sort_values("Risk_Ratio", ascending=False).reset_index(drop=True)

            warna = []
            total = len(region_risk)
            for i in range(total):
                if i < total * 0.33:
                    warna.append("red")
                elif i < total * 0.66:
                    warna.append("orange")
                else:
                    warna.append("green")

            region_risk["Warna"] = warna

            fig2 = px.bar(
                region_risk,
                x="Region",
                y="Risk_Percent",
                color="Warna",
                title="🔥 Persentase Risiko per Region (Berdasarkan Model)",
                text="Risk_Percent"
            )

            fig2.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            max_r = region_risk.iloc[0]
            st.success(f"🌋 Region paling berisiko: **{max_r['Region']} ({max_r['Risk_Percent']}%)**")

    except Exception as e:
        st.warning(f"Gagal menampilkan visualisasi region: {e}")


    
        
    # ============================================================
    # 4️⃣ SIMULASI PESANAN BARU
    # ============================================================
    st.subheader("🎯 Simulasi Risiko Pesanan Baru")

    if META_KLASIFIKASI and MODEL_KLASIFIKASI and SCALER_KLASIFIKASI:

        days_scheduled = st.number_input("Days for shipment scheduled", value=2)
        days_real = st.number_input("Days for shipping real", value=3)

        shipping_mode = st.selectbox("Shipping Mode",
                                     ["Same Day", "Second Class", "Standard Class"])

        region = st.selectbox("Region", [
            "Caribbean","Central Africa","Central America","Central Asia",
            "East Africa","East of USA","Eastern Asia","Eastern Europe",
            "North Africa","Northern Europe","Oceania","South America",
            "South Asia","South of  USA ","Southeast Asia","Southern Africa",
            "Southern Europe","US Center ","West Africa","West Asia",
            "West of USA ","Western Europe"
        ])

        category = st.selectbox("Kategori Produk", [
            k.replace("category_name_", "") for k in META_KLASIFIKASI if k.startswith("category_name_")
        ])

        segment = st.selectbox("Customer Segment", ["Corporate", "Home Office"])


        if st.button("Prediksi Risiko"):
            try:
                X_dict = {col: 0 for col in META_KLASIFIKASI}

                X_dict["days_for_shipment_scheduled"] = days_scheduled
                X_dict["days_for_shipping_real"] = days_real
                X_dict["shipment_delay"] = days_real - days_scheduled

                if f"shipping_mode_{shipping_mode}" in X_dict:
                    X_dict[f"shipping_mode_{shipping_mode}"] = 1

                if f"order_region_{region}" in X_dict:
                    X_dict[f"order_region_{region}"] = 1

                if f"category_name_{category}" in X_dict:
                    X_dict[f"category_name_{category}"] = 1

                if f"customer_segment_{segment}" in X_dict:
                    X_dict[f"customer_segment_{segment}"] = 1

                X_df = pd.DataFrame([X_dict])

                num_cols = [
                    "days_for_shipment_scheduled",
                    "days_for_shipping_real",
                    "shipment_delay"
                ]
                X_df[num_cols] = SCALER_KLASIFIKASI.transform(X_df[num_cols])

                pred = MODEL_KLASIFIKASI.predict(X_df)[0]
                prob = MODEL_KLASIFIKASI.predict_proba(X_df)[0]

                if pred == 1:
                    st.error(f"💥 Risiko Tinggi (Terlambat) – Prob: {prob[1]:.2f}")
                else:
                    st.success(f"🟢 Tidak Terlambat – Prob Risiko Tinggi: {prob[1]:.2f}")

            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")

    else:
        st.warning("⚠️ Model / metadata klasifikasi belum dimuat.")
        
    
    # ============================================================
    # 3️⃣.b TOP 10 PRODUK RISIKO TERTINGGI
    # ============================================================
    st.subheader("🔥 Top 10 Produk Dengan Risiko Keterlambatan Tertinggi")

    try:
        # Pastikan kolom yang diperlukan ada
        if "Top_Kategori" not in DF_KLASIFIKASI.columns or "Proporsi" not in DF_KLASIFIKASI.columns:
            st.warning("❌ Dataset agregasi tidak memiliki kolom 'Top_Kategori' atau 'Proporsi'.")
        else:
            st.info("📊 Menggunakan data agregasi (Proporsi) untuk menentukan Top 10 kategori risiko tertinggi.")

            # Extract nama kategori dari JSON
            DF_KLASIFIKASI["Kategori"] = DF_KLASIFIKASI["Top_Kategori"].apply(
                lambda x: x["Category"] if isinstance(x, dict) and "Category" in x else str(x)
            )

            # Sort berdasarkan Proporsi (risiko)
            top10 = DF_KLASIFIKASI.sort_values("Proporsi", ascending=False).head(10)

            # Buat warna gradasi
            warna = []
            total = len(top10)
            for i in range(total):
                if i < total * 0.33:
                    warna.append("red")
                elif i < total * 0.66:
                    warna.append("orange")
                else:
                    warna.append("green")

            top10["Warna"] = warna

            # Visualisasi
            fig_top = px.bar(
                top10,
                x="Kategori",
                y="Proporsi",
                color="Warna",
                text="Proporsi",
                title="🔥 Top 10 Kategori Dengan Risiko Keterlambatan Tertinggi"
            )

            fig_top.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_top, use_container_width=True)

            # Output kategori tertinggi
            st.success(
                f"🏆 Risiko tertinggi: **{top10.iloc[0]['Kategori']} ({top10.iloc[0]['Proporsi']})**"
            )

    except Exception as e:
        st.warning(f"Gagal membuat visualisasi Top 10 Produk: {e}")
    
    
# ---------------------------------------------------------
# TAB 3 – OPTIMIZATION & STRATEGY (VERSION FOR HUMAN-FRIENDLY UI)
# ---------------------------------------------------------
with tab3:

    st.header("🚀 3. Optimization & Strategy")

    # ============================================================
    # 1️⃣ FEATURE IMPORTANCE – LANGSUNG DARI MODEL
    # ============================================================
    st.subheader("📌 Faktor yang Paling Mempengaruhi Waktu Pengiriman")

    st.write("""
        Grafik ini menunjukkan **10 faktor teratas** yang paling mempengaruhi kecepatan pengiriman 
        berdasarkan model Machine Learning.  
        Nilai **Importance** = seberapa besar pengaruh suatu fitur terhadap hasil prediksi.
    """)

    try:
        if MODEL_PREDIKSI is None:
            st.warning("Model prediksi waktu belum dimuat.")
        else:
            fitur_model = MODEL_PREDIKSI.feature_names_in_
            importance_model = MODEL_PREDIKSI.feature_importances_

            df_imp = pd.DataFrame({
                "Fitur": fitur_model,
                "Importance": importance_model
            })

            df_imp = df_imp.sort_values("Importance", ascending=False).head(10)
            df_imp = df_imp.sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                df_imp,
                x="Importance",
                y="Fitur",
                orientation="h",
                title="🔍 Top 10 Faktor Paling Berpengaruh Terhadap Waktu Pengiriman",
            )

            st.plotly_chart(fig_imp, use_container_width=True)

            st.info("💡 *Semakin ke kanan batangnya, semakin besar pengaruh fitur tersebut terhadap waktu pengiriman.*")

    except Exception as e:
        st.error(f"Gagal mengambil feature importance dari model. Error: {e}")
        
    
    # ============================================================
    #  ACTUAL VS PREDICTED — HANYA VISUALISASI UTAMA
    # ============================================================

    st.subheader("📈 Realita vs Prediksi Waktu Pengiriman")

    try:
        if MODEL_PREDIKSI is None or META_WAKTU is None:
            st.warning("Model prediksi atau metadata belum dimuat.")
        else:

            # 1. Load X_test dan y_test
            X_test = pd.read_csv("X_test.csv")
            y_test = pd.read_csv("y_test.csv")

            # --- FIX DIMENSI Y_TEST ---
            if isinstance(y_test, pd.DataFrame):
                if y_test.shape[1] == 1:
                    y_test = y_test.iloc[:, 0]  # ubah ke Series
                else:
                    raise ValueError("y_test punya lebih dari 1 kolom.")

            # --- FIX URUTAN FITUR SESUAI MODEL ---
            fitur_model = list(META_WAKTU)
            X_test = X_test[fitur_model]

            # pastikan numerik
            X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

            # 2. Prediksi
            pred = MODEL_PREDIKSI.predict(X_test)
            pred = np.array(pred).reshape(-1)

            # 3. Samakan panjang data
            n = min(len(y_test), len(pred))

            df_plot = pd.DataFrame({
                "Actual": y_test.values[:n].reshape(-1),
                "Predicted": pred[:n]
            })

            # ============================================================
            # 10. GRAFIK UTAMA — ACTUAL VS PREDICTED
            # ============================================================

            st.subheader("📈 Akurasi Prediksi (Semakin Dekat Garis Merah Semakin Bagus)")

            fig = px.scatter(
                df_plot,
                x="Actual",
                y="Predicted",
                opacity=0.7,
                title="Actual vs Predicted — Semakin Dekat ke Garis Merah Semakin Akurat"
            )

            fig.add_shape(
                type="line",
                x0=df_plot["Actual"].min(),
                y0=df_plot["Actual"].min(),
                x1=df_plot["Actual"].max(),
                y1=df_plot["Actual"].max(),
                line=dict(color="red", dash="dash")
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membuat grafik Actual vs Predicted: {e}")
    
    
    # ============================================================
    # 3️⃣ SIMULATOR WAKTU PENGIRIMAN
    # ============================================================
    st.subheader("⚙️ Simulator Estimasi Waktu Pengiriman")

    st.write("""
        Gunakan simulator ini untuk melihat perkiraan waktu pengiriman antara **Standard**, 
        **Express**, dan **Same Day** berdasarkan jarak.
    """)

    mode = st.radio("Pilih Mode Pengiriman:", ["Standard", "Express", "Same Day"])
    dist = st.number_input("Masukkan jarak (km)", min_value=1, value=500)

    if st.button("🔮 Hitung Estimasi"):

        # Simulasi berbasis rumus sederhana
        if mode == "Standard":
            pred = 6.5 + (dist / 500)
            delta = "3 hari lebih lambat dari Express"
        elif mode == "Express":
            pred = 3.5 + (dist / 1000)
            delta = "0 hari (baseline)"
        else:  # Same Day
            pred = 1
            delta = "5 hari lebih cepat dari Standard"

        st.metric("Estimasi Waktu Pengiriman", f"{pred:.1f} Hari", delta)

        st.write("---")

        st.subheader("📈 Perbandingan Waktu Antar Mode Pengiriman")

        st.write("""
            Grafik berikut menunjukkan **seberapa cepat masing-masing mode** 
            jika jarak semakin jauh.  
            • **Semakin tinggi garis → semakin lama waktu pengiriman**  
            • Mode **Same Day** selalu 1 hari, tidak tergantung jarak  
        """)

        # ==============================
        # VISUALISASI PERBANDINGAN MODE
        # ==============================
        jarak_list = list(range(0, 1001, 50))

        sim_df = pd.DataFrame({
            "Jarak": jarak_list,
            "Standard": [6.5 + (d/500) for d in jarak_list],
            "Express":  [3.5 + (d/1000) for d in jarak_list],
            "Same Day": [1 for _ in jarak_list]
        })

        sim_df_melt = sim_df.melt(id_vars="Jarak", var_name="Mode", value_name="Waktu")

        fig_sim = px.line(
            sim_df_melt,
            x="Jarak",
            y="Waktu",
            color="Mode",
            markers=True,
            title="📈 Perbandingan Waktu Pengiriman Berdasarkan Jarak",
        )

        st.plotly_chart(fig_sim, use_container_width=True)

        st.info("💡 *Dari grafik, terlihat bahwa Express lebih cepat dari Standard, dan Same Day tetap paling cepat untuk semua jarak.*")
