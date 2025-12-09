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

    df_anomali = pd.DataFrame({
        'Is_Anomaly': ['No', 'Yes', 'No'],
        'Anomaly_Score': [0.1, -0.5, 0.2],
        'Sales': [100, 20, 80],
        'Order_Profit': [20, -5, 15]
    })

    df_klasifikasi = pd.DataFrame({
        'Proporsi': ['{"Risk_Level": ["Low","High"], "Count": [80,20]}'],
        'Top_Kategori': ['{"Category":["Furniture","Tech"],"Risk_Count":[12,7]}']
    })

    df_importance = pd.DataFrame({
        'Fitur': ['Shipping Mode', 'Distance', 'Item Type'],
        'Importance': [0.6, 0.4, 0.2]
    })

    kolom_klasifikasi = {}
    feature_waktu = {}

    def safe_load(path, method, default_value, display_name):
        if not os.path.exists(path):
            st.sidebar.warning(f"⚠️ FILE '{path}' not found ({display_name}).")
            return default_value
        try:
            data = method(path)
            st.sidebar.success(f"Successfully loaded {display_name}")
            return data
        except Exception as e:
            st.sidebar.error(f"Error loading {display_name}: {e}")
            return default_value

    st.sidebar.subheader("Model & Data Loading Status")

    # Load Model
    model_anomali = safe_load('model_isolation_forest.pkl', joblib.load, None, "Anomaly Model")
    model_klasifikasi = safe_load('model_klasifikasi.pkl', joblib.load, None, "Classification Model")
    scaler_klasifikasi = safe_load('scaler_klasifikasi.pkl', joblib.load, None, "Classification Scaler")
    model_prediksi_waktu = safe_load('model_waktu_prediksi.pkl', joblib.load, None, "Time Prediction Model")

    # Load CSV
    df_anomali_loaded = safe_load(
        'hasil_deteksi_anomali.csv',
        lambda f: pd.read_csv(f).head(100),
        df_anomali, 
        "Anomaly Data"
    )

    df_klasifikasi_loaded = safe_load(
        'insight_klasifikasi.csv',
        lambda f: pd.read_csv(f).head(100),
        df_klasifikasi,
        "Classification Insight Data"
    )

    df_importance_loaded = safe_load(
        'feature_importance.csv',
        lambda f: pd.read_csv(f).head(100),
        df_importance,
        "Feature Importance"
    )

    # Load JSON
    kolom_klasifikasi = safe_load(
        'kolom_klasifikasi.json',
        lambda f: json.load(open(f)),
        {},
        "Classification Metadata"
    )

    feature_waktu = safe_load(
        'feature_waktu.json',
        lambda f: json.load(open(f)),
        {},
        "Time Prediction Metadata"
    )

    if isinstance(df_anomali_loaded, pd.DataFrame): df_anomali = df_anomali_loaded
    if isinstance(df_klasifikasi_loaded, pd.DataFrame): df_klasifikasi = df_klasifikasi_loaded
    if isinstance(df_importance_loaded, pd.DataFrame): df_importance = df_importance_loaded

    return (
        model_anomali,
        model_klasifikasi,
        scaler_klasifikasi,
        model_prediksi_waktu,
        df_anomali,
        df_klasifikasi,
        df_importance,
        kolom_klasifikasi,
        feature_waktu
    )


# LOAD RESOURCES
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
st.markdown("Integrated dashboard for fast decision-making for warehouse/logistics managers.")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🚨 Early Warning System (Anomaly)",
    "📈 Risk Profiling (Classification)",
    "⚙️ Optimization & Strategy (Time Prediction)"
])


# ---------------------------------------------------------
# TAB 1 – ANOMALY DETECTION
# ---------------------------------------------------------
with tab1:

    st.header("1. Early Warning System")
    anomalies_count = 0

    if 'Is_Anomaly' in DF_ANOMALI.columns:
        anomalies_count = DF_ANOMALI[
            DF_ANOMALI['Is_Anomaly'].astype(str).str.lower() == 'yes'
        ].shape[0]

    if anomalies_count > 0:
        st.error(f"⚠️ Detected {anomalies_count} operational anomalies.")
    else:
        st.success("No anomalies detected.")

    st.subheader("Sales & Profit Visualization")
    try:
        st.line_chart(DF_ANOMALI[['Sales', 'Order_Profit']])
    except:
        st.warning("Columns Sales/Order_Profit not found.")

    st.subheader("Anomaly Details")
    if 'Is_Anomaly' in DF_ANOMALI.columns:
        df_f = DF_ANOMALI[DF_ANOMALI['Is_Anomaly'].astype(str).str.lower() == 'yes']
        st.dataframe(df_f.head(50))
    else:
        st.dataframe(DF_ANOMALI.head())


# ---------------------------------------------------------
# TAB 2 – CLASSIFICATION & RISK PREDICTION
# ---------------------------------------------------------
with tab2:

    st.header("🚦 Order Delay Risk Classification")

    # ============================================================
    # 1️⃣ RISK PROPORTION
    # ============================================================
    st.subheader("📊 Proportion of Delay Risk")

    try:
        raw = DF_KLASIFIKASI["Proporsi"].iloc[0]
        proporsi = json.loads(raw)
        df_proporsi = pd.DataFrame(proporsi)

        fig = px.pie(df_proporsi, names="Risk_Level", values="Count", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Failed to visualize Risk Proportion. Error: {e}")
        
    
    # ============================================================
    # 2️⃣ HIGHEST-RISK PRODUCT CATEGORIES (NO PREDICTED LABEL)
    # ============================================================
    st.subheader("🏆 Product Categories with Highest Delay Risk")

    try:
        model_features = list(MODEL_KLASIFIKASI.feature_names_in_)
        kategori_cols = [c for c in model_features if c.startswith("category_name_")]

        if len(kategori_cols) == 0:
            st.warning("❌ No category_name_* features found in classification model.")

        else:
            st.info("🔄 Calculating category risk based on model...")

            hasil_kat = []

            for col in kategori_cols:
                kategori_name = col.replace("category_name_", "")

                X_dummy = pd.DataFrame([{f: 0 for f in model_features}])
                X_dummy[col] = 1  

                num_cols = [
                    c for c in [
                        "days_for_shipment_scheduled",
                        "days_for_shipping_real",
                        "shipment_delay"
                    ] if c in model_features
                ]

                if len(num_cols) > 0:
                    try:
                        X_dummy[num_cols] = SCALER_KLASIFIKASI.transform(X_dummy[num_cols])
                    except:
                        pass

                pred = MODEL_KLASIFIKASI.predict_proba(X_dummy)[0][1]
                hasil_kat.append([kategori_name, pred])

            df_kat = pd.DataFrame(hasil_kat, columns=["Category", "Risk_Ratio"])
            df_kat["Risk_Percent"] = (df_kat["Risk_Ratio"] * 100).round(2)

            df_kat_sorted = df_kat.sort_values("Risk_Ratio", ascending=False).reset_index(drop=True)

            top_kat = df_kat_sorted.iloc[0]

            st.success(f"🔥 **Highest Risk Category:** {top_kat['Category']} — {top_kat['Risk_Percent']}%")

            df_top10 = df_kat_sorted.head(10)
            fig_top10 = px.bar(
                df_top10,
                x="Category",
                y="Risk_Percent",
                title="🔥 Top 10 Highest-Risk Product Categories",
                text="Risk_Percent",
                color="Risk_Percent",
                color_continuous_scale="Reds"
            )

            fig_top10.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_top10, use_container_width=True)

    except Exception as e:
        st.warning(f"Failed to calculate category risks. Error: {e}")

        
    # ============================================================
    # 3️⃣ HIGH-RISK REGION VISUALIZATION
    # ============================================================
    st.subheader("🌍 High-Risk Region Visualization")

    try:
        model_features = list(MODEL_KLASIFIKASI.feature_names_in_)
        region_cols = [col for col in model_features if col.startswith("order_region_")]

        if len(region_cols) == 0:
            st.warning("❌ Model does not contain region features: order_region_*")        

        else:
            st.info("🔄 Recalculating risk per region using CLASSIFICATION MODEL.")

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

            region_risk["Color"] = warna

            fig2 = px.bar(
                region_risk,
                x="Region",
                y="Risk_Percent",
                color="Color",
                title="🔥 Risk Percentage per Region (Model-Based)",
                text="Risk_Percent"
            )

            fig2.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            max_r = region_risk.iloc[0]
            st.success(f"🌋 Highest Risk Region: **{max_r['Region']} ({max_r['Risk_Percent']}%)**")

    except Exception as e:
        st.warning(f"Failed to display region visualisation: {e}")

        
    # ============================================================
    # 4️⃣ NEW ORDER RISK SIMULATION
    # ============================================================
    st.subheader("🎯 New Order Risk Simulation")

    if META_KLASIFIKASI and MODEL_KLASIFIKASI and SCALER_KLASIFIKASI:

        days_scheduled = st.number_input("Days for shipment scheduled", value=2)
        days_real = st.number_input("Days for shipping (actual)", value=3)

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

        category = st.selectbox("Product Category", [
            k.replace("category_name_", "") for k in META_KLASIFIKASI if k.startswith("category_name_")
        ])

        segment = st.selectbox("Customer Segment", ["Corporate", "Home Office"])


        if st.button("Predict Risk"):
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
                    st.error(f"💥 High Risk (Delayed) – Probability: {prob[1]:.2f}")
                else:
                    st.success(f"🟢 Not Delayed – High Risk Probability: {prob[1]:.2f}")

            except Exception as e:
                st.error(f"Failed to generate prediction: {e}")

    else:
        st.warning("⚠️ Classification model / metadata not loaded.")
        
# ---------------------------------------------------------
# TAB 3 – OPTIMIZATION & STRATEGY (HUMAN-FRIENDLY VERSION)
# ---------------------------------------------------------
with tab3:

    st.header("🚀 3. Optimization & Strategy")

    # ============================================================
    # 1️⃣ FEATURE IMPORTANCE – DIRECTLY FROM MODEL
    # ============================================================
    st.subheader("📌 Key Factors Affecting Delivery Time")

    st.write("""
        This chart displays the **Top 10 most influential factors** affecting delivery time  
        according to the Machine Learning model.  
        **Importance** represents how much each feature contributes to predictions.
    """)

    try:
        if MODEL_PREDIKSI is None:
            st.warning("Delivery time prediction model is not loaded yet.")
        else:
            fitur_model = MODEL_PREDIKSI.feature_names_in_
            importance_model = MODEL_PREDIKSI.feature_importances_

            df_imp = pd.DataFrame({
                "Feature": fitur_model,
                "Importance": importance_model
            })

            df_imp = df_imp.sort_values("Importance", ascending=False).head(10)
            df_imp = df_imp.sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                df_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                title="🔍 Top 10 Most Influential Factors on Delivery Time",
            )

            st.plotly_chart(fig_imp, use_container_width=True)

            st.info("💡 *The farther the bar extends to the right, the stronger its impact on delivery time.*")

    except Exception as e:
        st.error(f"Failed to retrieve feature importance. Error: {e}")

    
    # ============================================================
    # 2️⃣ ACTUAL VS PREDICTED — MAIN VISUALIZATION ONLY
    # ============================================================

    st.subheader("📈 Actual vs Predicted Delivery Time")

    try:
        if MODEL_PREDIKSI is None or META_WAKTU is None:
            st.warning("Prediction model or metadata not loaded.")
        else:

            # Load test data
            X_test = pd.read_csv("X_test.csv")
            y_test = pd.read_csv("y_test.csv")

            # Fix y_test dimensions
            if isinstance(y_test, pd.DataFrame):
                if y_test.shape[1] == 1:
                    y_test = y_test.iloc[:, 0]
                else:
                    raise ValueError("y_test contains more than one column.")

            # Match feature order with model
            fitur_model = list(META_WAKTU)
            X_test = X_test[fitur_model]

            X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Predictions
            pred = MODEL_PREDIKSI.predict(X_test)
            pred = np.array(pred).reshape(-1)

            # Equalize lengths
            n = min(len(y_test), len(pred))

            df_plot = pd.DataFrame({
                "Actual": y_test.values[:n].reshape(-1),
                "Predicted": pred[:n]
            })

            # ============================================================
            # MAIN PLOT — ACTUAL VS PREDICTED
            # ============================================================

            st.subheader("📈 Prediction Accuracy (Closer to Red Line = Better)")

            fig = px.scatter(
                df_plot,
                x="Actual",
                y="Predicted",
                opacity=0.7,
                title="Actual vs Predicted — Accuracy increases closer to the red diagonal"
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
        st.error(f"Failed to generate Actual vs Predicted chart: {e}")

    
    # ============================================================
    # 3️⃣ DELIVERY TIME SIMULATOR
    # ============================================================
    st.subheader("⚙️ Delivery Time Simulator")

    st.write("""
        Use this simulator to estimate delivery time for **Standard**, **Express**,  
        or **Same Day** shipping based on distance.
    """)

    mode = st.radio("Choose Shipping Mode:", ["Standard", "Express", "Same Day"])
    dist = st.number_input("Enter distance (km)", min_value=1, value=500)

    if st.button("🔮 Estimate Time"):

        # Simple estimation formula
        if mode == "Standard":
            pred = 6.5 + (dist / 500)
            delta = "3 days slower than Express"
        elif mode == "Express":
            pred = 3.5 + (dist / 1000)
            delta = "Baseline (0 days difference)"
        else:  # Same Day
            pred = 1
            delta = "5 days faster than Standard"

        st.metric("Estimated Delivery Time", f"{pred:.1f} Days", delta)

        st.write("---")

        st.subheader("📈 Delivery Time Comparison Across Modes")

        st.write("""
            This chart shows **how delivery time increases with distance**  
            for each shipping mode.  
            • **Higher lines → longer delivery time**  
            • **Same Day** is always 1 day regardless of distance  
        """)

        # Comparison simulation
        distance_list = list(range(0, 1001, 50))

        sim_df = pd.DataFrame({
            "Distance": distance_list,
            "Standard": [6.5 + (d/500) for d in distance_list],
            "Express":  [3.5 + (d/1000) for d in distance_list],
            "Same Day": [1 for _ in distance_list]
        })

        sim_df_melt = sim_df.melt(id_vars="Distance", var_name="Mode", value_name="Time")

        fig_sim = px.line(
            sim_df_melt,
            x="Distance",
            y="Time",
            color="Mode",
            markers=True,
            title="📈 Delivery Time Comparison by Distance",
        )

        st.plotly_chart(fig_sim, use_container_width=True)

        st.info("💡 *Express is consistently faster than Standard, and Same Day remains the fastest for all distances.*")
