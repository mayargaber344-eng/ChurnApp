import pandas as pd

from pathlib import Path
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# App title
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to leave the company."
)

st.divider()

with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1]
        )

        partner = st.selectbox(
            "Partner", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents", ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.subheader("Internet Services")

    col3, col4 = st.columns(2)

    with col3:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col4:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    predict_button = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True
    )


if predict_button:

    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": senior_citizen,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method
    }])

    try:
        prediction = model.predict(input_data)[0]
        result = str(prediction).strip().lower()

        st.divider()
        st.subheader("Prediction Result")

        if result in ["1", "yes", "true", "churn"]:
            st.error(
                "⚠️ This customer is at risk of leaving!"
            )
            st.write(
                "The model predicts that this customer will churn."
            )

        elif result in ["0", "no", "false", "stay"]:
            st.success(
                "✅ This customer is likely to stay!"
            )
            st.write(
                "The model predicts that this customer will not churn."
            )

        else:
            st.warning(f"Model prediction: {prediction}")
            st.write(
                "Please verify the target labels used during training."
            )

        with st.expander("View Customer Details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
