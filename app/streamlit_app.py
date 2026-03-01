def render_risk_bar(prob):
    percentage = int(prob * 100)

    if percentage < 20:
        color = "#2ecc71"  # green
    elif percentage < 50:
        color = "#f1c40f"  # yellow
    elif percentage < 75:
        color = "#e67e22"  # orange
    else:
        color = "#e74c3c"  # red

    bar_html = f"""
    <div style="
        background-color: #eee;
        border-radius: 5px;
        width: 100%;
        height: 25px;
        margin-top: 10px;
    ">
        <div style="
            background-color: {color};
            width: {percentage}%;
            height: 100%;
            border-radius: 5px;
            text-align: center;
            color: white;
            font-weight: bold;
        ">
            {percentage}%
        </div>
    </div>
    """

    st.markdown(bar_html, unsafe_allow_html=True)

import streamlit as st
import requests

# =========================
# Configuration
# =========================

st.set_page_config(page_title="Nukefraud", layout="centered")

API_URL = "https://nukefraud.onrender.com/predict"

# =========================
# Sample Data
# =========================

LEGIT_SAMPLE = (
    "0.0,-1.3598071336738,-0.0727811733098497,2.53634673796914,"
    "1.37815522427443,-0.338320769942518,0.462387777762292,"
    "0.239598554061257,0.0986979012610507,0.363786969611213,"
    "0.0907941719789316,-0.551599533260813,-0.617800855762348,"
    "-0.991389847235408,-0.311169353699879,1.46817697209427,"
    "-0.470400525259478,0.207971241929242,0.0257905801985591,"
    "0.403992960255733,0.251412098239705,-0.018306777944153,"
    "0.277837575558899,-0.110473910188767,0.0669280749146731,"
    "0.128539358273528,-0.189114843888824,0.133558376740387,"
    "-0.0210530534538215,149.62"
)

FRAUD_SAMPLE = (
    "406.0,-2.3122265423263,1.95199201064158,-1.60985073229769,"
    "3.9979055875468,-0.522187864667764,-1.42654531920595,"
    "-2.53738730624579,1.39165724829804,-2.77008927719433,"
    "-2.77227214465915,3.20203320709635,-2.89990738849473,"
    "-0.595221881324605,-4.28925378244217,0.389724120274487,"
    "-1.14074717980657,-2.83005567450437,-0.0168224681808257,"
    "0.416955705037907,0.126910559061474,0.517232370861764,"
    "-0.0350493686052974,-0.465211076182388,0.320198198514526,"
    "0.0445191674731724,0.177839798284401,0.261145002567677,"
    "-0.143275874698919,0.0"
)

# =========================
# Callback Functions
# =========================

def fill_legit():
    st.session_state.input = LEGIT_SAMPLE

def fill_fraud():
    st.session_state.input = FRAUD_SAMPLE


# =========================
# UI
# =========================

st.title("Nukefraud — Fraud Detection System")
st.markdown(
    "Cost-sensitive credit card fraud detection with optimized decision threshold."
)

st.divider()

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "MLP"],
    key="model_select"
)

st.divider()

st.subheader("Transaction Input")

# Initialize session state before widget creation
if "input" not in st.session_state:
    st.session_state.input = ""

feature_input = st.text_area(
    "Enter 30 numerical features (comma-separated)",
    height=150,
    key="input"
)

col1, col2 = st.columns(2)

with col1:
    st.button(
        "Auto-fill Legit Sample",
        on_click=fill_legit,
        key="legit_btn"
    )

with col2:
    st.button(
        "Auto-fill Fraud Sample",
        on_click=fill_fraud,
        key="fraud_btn"
    )

st.divider()

# =========================
# Prediction
# =========================

if st.button("Predict", key="predict_btn"):

    try:
        features = [float(x.strip()) for x in st.session_state.input.split(",")]

        if len(features) != 30:
            st.error("Exactly 30 features are required.")
        else:
            response = requests.post(
                API_URL,
                json={
                    "features": features,
                    "model": model_choice
                }
            )

            if response.status_code == 200:
                result = response.json()

                prob = result["fraud_probability"]
                label = result["prediction"]
                threshold = result["threshold"]

                st.subheader("Prediction Result")

                st.metric(
                    label="Fraud Probability",
                    value=f"{prob:.4f}"
                )       

                render_risk_bar(prob)

                st.write(f"Decision Threshold: {threshold}")

                if label == 1:
                    st.error("High Risk: Fraudulent Transaction")
                else:
                    st.success("Low Risk: Legitimate Transaction")

            else:
                st.error("API error occurred.")

    except Exception:
        st.error("Invalid input format. Ensure values are numeric and comma-separated.")