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


# ==========================================
# CSV Batch Prediction
# ==========================================

st.divider()
st.header("📂 Batch Customer Churn Prediction")

st.write(
    "Upload a CSV file containing multiple customers "
    "to predict churn for all of them at once."
)

# The CSV must use the same feature columns as the model
BATCH_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Downloadable CSV template
template_row = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "SeniorCitizen": 0,
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

template_df = pd.DataFrame([template_row])

st.download_button(
    label="📥 Download CSV Template",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="churn_input_template.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader(
    "Upload customer CSV file",
    type=["csv"],
    key="batch_csv_uploader",
)

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        # Remove accidental spaces from column names
        batch_data.columns = batch_data.columns.str.strip()

        # Check required columns
        missing_columns = [
            col for col in BATCH_COLUMNS
            if col not in batch_data.columns
        ]

        if missing_columns:
            st.error(
                "The CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )
            st.info(
                "Download the CSV template above and use "
                "the same column names."
            )

        elif batch_data.empty:
            st.warning("The uploaded CSV file contains no customers.")

        else:
            # Keep only model input columns in the correct order
            input_batch = batch_data[BATCH_COLUMNS].copy()

            # Convert numeric columns to numeric data types
            numeric_columns = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "SeniorCitizen",
            ]

            for col in numeric_columns:
                input_batch[col] = pd.to_numeric(
                    input_batch[col],
                    errors="coerce"
                )

            # Reject rows with missing or invalid numeric values
            invalid_rows = input_batch[
                numeric_columns
            ].isna().any(axis=1)

            if invalid_rows.any():
                st.error(
                    f"{int(invalid_rows.sum())} row(s) contain "
                    "missing or invalid numeric values in "
                    "tenure, MonthlyCharges, TotalCharges, "
                    "or SeniorCitizen. Please correct the CSV "
                    "and upload it again."
                )

            else:
                if st.button(
                    "🔍 Predict All Customers",
                    key="batch_predict_button",
                    use_container_width=True,
                ):
                    
with st.spinner("Predicting customer churn..."):
    predictions = model.predict(input_batch)

    # Calculate churn probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_batch)

        classes = list(model.classes_)
        churn_class = next(
            (
                cls for cls in classes
                if str(cls).strip().lower()
                in ["1", "yes", "true", "churn"]
            ),
            None
        )

        if churn_class is not None:
            churn_index = classes.index(churn_class)
            churn_probabilities = probabilities[:, churn_index] * 100
        else:
            churn_probabilities = None
    else:
        churn_probabilities = None


                    results = batch_data.copy()
                    results["ChurnPrediction"] = predictions

if churn_probabilities is not None:
    results["ChurnProbability (%)"] = churn_probabilities.round(2)

                    def get_churn_status(value):
                        label = str(value).strip().lower()

                        if label in ["1", "yes", "true", "churn"]:
                            return "At Risk"
                        elif label in ["0", "no", "false", "stay"]:
                            return "Likely to Stay"
                        else:
                            return "Check Model Label"

                    results["ChurnStatus"] = [
                        get_churn_status(pred)
                        for pred in predictions
                    ]

                    at_risk = int(
                        (results["ChurnStatus"] == "At Risk").sum()
                                  )
                   likely_to_stay = int(
                              (results["ChurnStatus"] == "Likely to Stay").sum()
                                        )

# Display average churn probability
if churn_probabilities is not None:
    st.metric(
        "Average Churn Probability",
        f"{churn_probabilities.mean():.2f}%"
    )

st.subheader("📊 Batch Prediction Summary")

                    st.subheader("📊 Batch Prediction Summary")

                    metric1, metric2, metric3 = st.columns(3)

                    metric1.metric(
                        "Total Customers",
                        len(results)
                    )
                    metric2.metric(
                        "Customers At Risk",
                        at_risk
                    )
                    metric3.metric(
                        "Likely to Stay",
                        likely_to_stay
                    )

                    st.subheader("Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    csv_output = results.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        label="📥 Download Prediction Results",
                        data=csv_output,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        key="download_batch_results",
                        use_container_width=True,
                    )

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

