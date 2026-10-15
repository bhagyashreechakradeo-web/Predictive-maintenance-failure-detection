
import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("predictive_maintenance_model.pkl")
model_features = joblib.load("model_features.pkl")
feature_importance = pd.read_csv("feature_importance.csv")
THRESHOLD = 0.30

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

# Dashboard styling
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.hero {
    padding: 1.5rem;
    border-radius: 15px;
    background: linear-gradient(120deg, #14233d, #243b55);
    color: white;
}
.hero p {color: #d8e4f3;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>⚙️ Predictive Maintenance & Failure Explanation</h1>
<p>Machine-learning risk scoring and maintenance indicators</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "AI4I 2020 dataset demonstration. Predictions are estimates, "
    "not guarantees of machine safety."
)

# Sidebar: project and evaluation
with st.sidebar:
    st.header("About the project")
    st.write("Model: Random Forest")
    st.write("Failure alert threshold: 30%")

    st.divider()
    st.subheader("Held-out test performance")
    st.metric("Accuracy", "97.9%")
    st.metric("Failure precision", "66.7%")
    st.metric("Failure recall", "64.7%")
    st.metric("Failure F1-score", "0.657")

    st.caption(
        "These are results from the test split. "
        "The dataset is synthetic; real-world performance may differ."
    )

# Machine input form
st.subheader("1. Machine operating measurements")

with st.form("machine_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        machine_type = st.selectbox("Machine type", ["L", "M", "H"])
        tool_wear = st.number_input(
            "Tool wear (minutes)", 0, 500, 100
        )

    with c2:
        air_temp = st.number_input(
            "Air temperature (K)", 250.0, 350.0, 300.5
        )
        process_temp = st.number_input(
            "Process temperature (K)", 250.0, 400.0, 309.8
        )

    with c3:
        speed = st.number_input(
            "Rotational speed (rpm)", 0, 5000, 1500
        )
        torque = st.number_input(
            "Torque (Nm)", 0.0, 200.0, 40.0
        )

    submitted = st.form_submit_button(
        "Analyze Machine", type="primary", use_container_width=True
    )

if submitted:
    readings = {
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    # Match training-time encoding and feature order
    input_df = pd.DataFrame([readings])
    input_df = pd.get_dummies(
        input_df, columns=["Type"], dtype=int
    )
    input_df = input_df.reindex(
        columns=model_features, fill_value=0
    )

    probability = float(model.predict_proba(input_df)[0, 1])

    prediction = (
        "Potential Machine Failure"
        if probability >= THRESHOLD
        else "No Failure Alert"
    )

    if probability < 0.10:
        risk = "Low Estimated Risk"
    elif probability < THRESHOLD:
        risk = "Moderate Estimated Risk"
    else:
        risk = "High Estimated Risk"

    st.divider()
    st.subheader("2. Prediction report")

    a, b, c = st.columns(3)
    a.metric("Estimated failure probability", f"{probability:.4%}")
    b.metric("Model alert", prediction)
    c.metric("Risk category", risk)

    st.progress(probability)
    st.caption("Progress bar represents estimated failure probability.")

    if probability >= THRESHOLD:
        st.error("The estimated probability meets the alert threshold.")
    elif probability >= 0.10:
        st.warning("Probability is below the alert threshold, but is not zero.")
    else:
        st.success("The model did not trigger a failure alert.")

    # Rule-based indicators; these are separate from the model
    st.subheader("3. Potential risk indicators")
    indicators = []

    if speed < 1500:
        indicators.append(
            "Low rotational speed — inspect operating conditions."
        )
    if torque > 50:
        indicators.append(
            "High torque — check whether the machine is under high load."
        )
    if tool_wear > 200:
        indicators.append(
            "High tool wear — consider checking tool condition."
        )
    if air_temp > 302:
        indicators.append(
            "Elevated air temperature — review the operating environment."
        )
    if process_temp > 312:
        indicators.append(
            "Elevated process temperature — review process conditions."
        )

    if indicators:
        for item in indicators:
            st.warning(item)
    else:
        st.info(
            "No predefined indicators were triggered. "
            "This does not guarantee that the machine is safe."
        )

    with st.expander("View machine measurements"):
        st.dataframe(
            pd.DataFrame([readings]),
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("Enter the readings and click Analyze Machine to view a report.")

# -------------------- FEATURE IMPORTANCE --------------------

st.divider()
st.subheader("4. What influences the model overall?")

st.write(
    "This chart shows which features the Random Forest relies on most "
    "across its predictions. It does not explain an individual prediction."
)

importance_chart = feature_importance.sort_values(
    "Importance",
    ascending=True
).set_index("Feature")

st.bar_chart(importance_chart["Importance"])

st.caption(
    "Feature importance indicates overall model reliance, not causation "
    "or the reason for a specific machine's prediction."
)

st.divider()

st.caption(
    "Portfolio demonstration using a synthetic dataset. Risk indicators "
    "are predefined rules, not causal or individual model explanations. "
    "Do not use this tool as the sole basis for real-world safety decisions."
)
