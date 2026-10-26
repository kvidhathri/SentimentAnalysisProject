
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Product Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
)

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]


@st.cache_resource
def load_model():
    model_path = "./sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


def analyze_reviews(reviews):
    predictions = []
    confidence_scores = []
    probability_rows = []

    for start in range(0, len(reviews), 16):
        batch = reviews[start:start + 16]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        for probability in probabilities:
            prediction = torch.argmax(probability).item()
            predictions.append(LABELS[prediction])
            confidence_scores.append(
                round(probability[prediction].item() * 100, 2)
            )
            probability_rows.append({
                "Negative probability (%)": round(
                    probability[0].item() * 100, 2
                ),
                "Neutral probability (%)": round(
                    probability[1].item() * 100, 2
                ),
                "Positive probability (%)": round(
                    probability[2].item() * 100, 2
                ),
            })

    return predictions, confidence_scores, probability_rows


def show_analytics(df):
    st.subheader("📊 Sentiment Analytics Dashboard")

    total = len(df)

    counts = (
        df["predicted_sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
    )

    positive_count = int(counts["Positive"])
    neutral_count = int(counts["Neutral"])
    negative_count = int(counts["Negative"])

    positive_pct = positive_count / total * 100 if total else 0
    neutral_pct = neutral_count / total * 100 if total else 0
    negative_pct = negative_count / total * 100 if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", total)
    col2.metric("😊 Positive", positive_count, f"{positive_pct:.1f}%")
    col3.metric("😐 Neutral", neutral_count, f"{neutral_pct:.1f}%")
    col4.metric("😞 Negative", negative_count, f"{negative_pct:.1f}%")

    st.markdown("#### Sentiment Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("Review count by sentiment")
        st.bar_chart(counts)

    with chart_col2:
        st.write("Percentage of reviews")
        percentages = pd.DataFrame({
            "Sentiment": SENTIMENT_ORDER,
            "Percentage": [
                positive_pct,
                neutral_pct,
                negative_pct,
            ],
        }).set_index("Sentiment")

        st.bar_chart(percentages)

    if "confidence_percent" in df.columns:
        st.markdown("#### Prediction Confidence")

        average_confidence = df["confidence_percent"].mean()
        st.metric(
            "Average Prediction Confidence",
            f"{average_confidence:.2f}%",
        )

    st.caption(
        "These charts summarize model predictions, not verified customer opinions. "
        "Confidence is the model's estimated probability, not a guarantee of correctness."
    )


st.title("💬 Product Review Sentiment Analyzer")
st.write(
    "Analyze individual product reviews or upload a CSV file "
    "to analyze multiple reviews and explore sentiment analytics."
)

single_tab, bulk_tab, dashboard_tab = st.tabs([
    "📝 Single Review",
    "📁 Bulk CSV Analysis",
    "📊 Analytics Dashboard",
])


with single_tab:
    st.subheader("Analyze a Single Review")

    review = st.text_area(
        "Enter your product review",
        placeholder="Example: This product is excellent and useful!",
        height=130,
    )

    if st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        key="single_analyze",
    ):
        if not review.strip():
            st.warning("Please enter a product review first.")
        else:
            with st.spinner("Analyzing your review..."):
                predictions, confidences, probability_rows = analyze_reviews(
                    [review]
                )

            sentiment = predictions[0]
            confidence = confidences[0]

            st.subheader("Analysis Result")

            if sentiment == "Positive":
                st.success(f"😊 Sentiment: {sentiment}")
            elif sentiment == "Negative":
                st.error(f"😞 Sentiment: {sentiment}")
            else:
                st.info(f"😐 Sentiment: {sentiment}")

            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            st.write("**Sentiment Probabilities**")

            for class_id, label in LABELS.items():
                probability = probability_rows[0][
                    f"{label} probability (%)"
                ]
                st.write(f"{label}: {probability:.2f}%")
                st.progress(probability / 100)

            st.caption(
                "Confidence is the model's estimated probability for its "
                "predicted class, not a guarantee that the prediction is correct."
            )


with bulk_tab:
    st.subheader("Bulk Review Analysis")
    st.write("Upload a CSV file containing a column named `review`.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="bulk_csv",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if "review" not in uploaded_df.columns:
                st.error(
                    "Your CSV must contain a column named 'review'."
                )
            else:
                valid_mask = (
                    uploaded_df["review"].notna()
                    & uploaded_df["review"].astype(str).str.strip().ne("")
                )
                results_df = uploaded_df.loc[valid_mask].copy()
                results_df["review"] = results_df["review"].astype(str)

                if results_df.empty:
                    st.warning("No valid reviews were found in the CSV.")
                elif len(results_df) > 10000:
                    st.warning(
                        "Please upload no more than 10,000 reviews at a time."
                    )
                elif st.button(
                    "📊 Analyze All Reviews",
                    type="primary",
                    key="bulk_analyze",
                ):
                    with st.spinner(
                        f"Analyzing {len(results_df)} reviews..."
                    ):
                        predictions, confidences, probability_rows = (
                            analyze_reviews(results_df["review"].tolist())
                        )

                    results_df["predicted_sentiment"] = predictions
                    results_df["confidence_percent"] = confidences

                    for column in probability_rows[0]:
                        results_df[column] = [
                            row[column] for row in probability_rows
                        ]

                    st.session_state["bulk_results"] = results_df

        except Exception as error:
            st.error(f"Could not read the CSV file: {error}")


with dashboard_tab:
    if "bulk_results" not in st.session_state:
        st.info(
            "Upload a CSV file in the Bulk CSV Analysis tab and click "
            "'Analyze All Reviews'. Your dashboard will appear here."
        )
    else:
        results_df = st.session_state["bulk_results"]

        show_analytics(results_df)

        st.markdown("#### Analyzed Reviews")
        st.dataframe(results_df, use_container_width=True)

        csv_data = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Analysis Results",
            data=csv_data,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
        )


st.divider()
st.caption("Built with Python, Streamlit, pandas, and DistilBERT.")
