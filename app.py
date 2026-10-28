            else:
                # Use only model features, in the expected order.
                input_batch = batch_data[BATCH_COLUMNS].copy()

                # Convert numeric columns; invalid values become NaN.
                for column in NUMERIC_COLUMNS:
                    input_batch[column] = pd.to_numeric(
                        input_batch[column],
                        errors="coerce",
                    )

                invalid_rows = input_batch[NUMERIC_COLUMNS].isna().any(axis=1)

                if invalid_rows.any():
                    st.error(
                        f"{int(invalid_rows.sum())} row(s) contain missing "
                        "or invalid numeric values in one or more of these "
                        "columns: tenure, MonthlyCharges, TotalCharges, "
                        "SeniorCitizen."
                    )
                    st.write(
                        "Correct these rows in Excel, save the file as CSV, "
                        "and upload it again. Row numbers below refer to "
                        "the original CSV data rows (header is row 1)."
                    )

                    invalid_preview = batch_data.loc[
                        invalid_rows,
                        [column for column in NUMERIC_COLUMNS
                         if column in batch_data.columns],
                    ].copy()
                    invalid_preview.insert(
                        0,
                        "CSV Row Number",
                        invalid_preview.index + 2,
                    )
                    st.dataframe(
                        invalid_preview,
                        use_container_width=True,
                    )
                else:
                    st.caption(
                        f"Ready to predict for {len(input_batch)} customer(s)."
                    )

                    if st.button(
                        "🔍 Predict All Customers",
                        key="batch_predict_button",
                        use_container_width=True,
                    ):
                        with st.spinner("Predicting customer churn..."):
                            predictions = model.predict(input_batch)
                            churn_probabilities = get_churn_probabilities(
                                input_batch
                            )

                        results = batch_data.copy()
                        results["ChurnPrediction"] = predictions
                        results["ChurnStatus"] = [
                            get_churn_status(prediction)
                            for prediction in predictions
                        ]

                        if churn_probabilities is not None:
                            results["ChurnProbability (%)"] = (
                                churn_probabilities.round(2)
                            )

                        at_risk = int(
                            (results["ChurnStatus"] == "At Risk").sum()
                        )
                        likely_to_stay = int(
                            (results["ChurnStatus"] == "Likely to Stay").sum()
                        )
                        unknown_labels = int(
                            (results["ChurnStatus"] == "Check Model Label").sum()
                        )

                        st.subheader("📊 Batch Prediction Summary")

                        if churn_probabilities is not None:
                            metric1, metric2, metric3, metric4 = st.columns(4)
                        else:
                            metric1, metric2, metric3 = st.columns(3)

                        metric1.metric("Total Customers", len(results))
                        metric2.metric("Customers At Risk", at_risk)
                        metric3.metric("Likely to Stay", likely_to_stay)

                        if churn_probabilities is not None:
                            metric4.metric(
                                "Average Churn Probability",
                                f"{churn_probabilities.mean():.2f}%",
                            )

                        if unknown_labels:
                            st.warning(
                                f"{unknown_labels} prediction(s) have labels "
                                "that this app does not recognize as churn "
                                "or stay. Check the target labels used when "
                                "training the model."
                            )

                        st.subheader("Prediction Results")
                        st.dataframe(results, use_container_width=True)

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
