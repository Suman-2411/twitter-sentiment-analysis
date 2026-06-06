import streamlit as st
from vader_analysis import analyze_vader
from roberta_analysis import analyze_roberta
from charts import make_comparison_chart
from charts import make_pie_chart
from metrics import calculate_metrics
import pandas as pd
import os

bearer_token = os.environ.get("BEARER_TOKEN")

page = st.sidebar.selectbox(
    "Navigate",
    ["ℹ About", "🔍 Live Analysis", "📊 Model Comparison","📈 Model Metrics"]
)

# dark color looks good for this app
st.markdown("""
<style>
.stApp { background-color: #111111; }
h1, h2, h3 { color: #FF0000 !important; }
p { color: #CCCCCC !important; }
</style>
""", unsafe_allow_html=True)


#------- Live Analysis ----------
if page == "🔍 Live Analysis":
    st.title("🔍 Live Tweet Analysis")
    st.divider()

    # ask user - live tweets or kaggle?
    source = st.radio(
        "Where should I get the tweets from?",
        ["🐦 Live Tweets (Tweepy)", "📁 Kaggle Dataset"],
        horizontal=True
    )

    keyword = st.text_input("Enter a keyword:")

    if st.button("Analyze"):
        if keyword == "":
            st.warning("Please enter something!")

        else:
            # get tweets from twitter
            if source == "🐦 Live Tweets (Tweepy)":
                with st.spinner(f"Getting tweets about '{keyword}'..."):
                    try:
                        from fetch_tweets import fetch_tweets
                        from clean_data import clean_tweet
                        df = fetch_tweets(keyword, max_results=10)
                        df['clean_tweet'] = df['text'].apply(clean_tweet)
                        # remove empty ones after cleaning
                        df = df[df['clean_tweet'].str.len() > 0]
                    except Exception as e:
                        st.error(f"tweepy error: {e}")
                        st.stop()

            # search keyword in kaggle data
            # only take 20 because roberta is slow
            else:
                try:
                    df = pd.read_csv("cleaned_tweets.csv")

                    # FIX: split input into individual words so "i love cricket"
                    # searches for "cricket" not the full sentence
                    keywords = keyword.lower().split()

                    def contains_any_keyword(tweet):
                        for word in keywords:
                            if word in tweet:
                                return True
                        return False

                    mask = df['clean_tweet'].apply(contains_any_keyword)
                    df = df[mask].head(20)

                    if df.empty:
                        st.warning(f"No tweets found for '{keyword}', try another word.")
                        st.stop()
                except FileNotFoundError:
                    st.error("cant find cleaned_tweets.csv, did you run clean_data.py?")
                    st.stop()

            # TODO: add progress bar here later, looks empty while waiting
            with st.spinner("running both models... roberta takes time sorry"):
                df['vader_sentiment'], df['vader_score'] = zip(
                    *df['clean_tweet'].apply(analyze_vader)
                )
                df['roberta_sentiment'] = df['clean_tweet'].apply(analyze_roberta)

            st.success(f"Done! Analyzed {len(df)} tweets about '{keyword}'")

            # show total count for each model
            col1, col2 = st.columns(2)
            with col1:
                st.write("⚡ VADER")
                st.caption(
                    f"Positive: {(df['vader_sentiment']=='positive').sum()}  |  "
                    f"Negative: {(df['vader_sentiment']=='negative').sum()}  |  "
                    f"Neutral: {(df['vader_sentiment']=='neutral').sum()}"
                )
            with col2:
                st.write("🧠 RoBERTa")
                st.caption(
                    f"Positive: {(df['roberta_sentiment']=='positive').sum()}  |  "
                    f"Negative: {(df['roberta_sentiment']=='negative').sum()}  |  "
                    f"Neutral: {(df['roberta_sentiment']=='neutral').sum()}"
                )

            st.divider()

            # show each tweet one by one
            for i, row in df.iterrows():
                with st.expander(f"Tweet {i+1} — VADER: {row['vader_sentiment']} | RoBERTa: {row['roberta_sentiment']}"):
                    st.write(row['text'] if 'text' in df.columns else row['clean_tweet'])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("⚡ VADER")
                        if row['vader_sentiment'] == 'positive':
                            st.success(f"😊 Positive ({row['vader_score']:.2f})")
                        elif row['vader_sentiment'] == 'negative':
                            st.error(f"😠 Negative ({row['vader_score']:.2f})")
                        else:
                            st.warning(f"😐 Neutral ({row['vader_score']:.2f})")

                    with col2:
                        st.write("🧠 RoBERTa")
                        if row['roberta_sentiment'] == 'positive':
                            st.success("😊 Positive")
                        elif row['roberta_sentiment'] == 'negative':
                            st.error("😠 Negative")
                        else:
                            st.warning("😐 Neutral")

                    # both models gave different answer
                    if row['vader_sentiment'] != row['roberta_sentiment']:
                        st.info("💡 Models disagree on this one")

#------Model comparison-----
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.divider()

    try:
        df_vader = pd.read_csv('vader_results.csv')
        df_roberta = pd.read_csv('roberta_result.csv')

        # Bar chart comparison
        st.subheader("VADER vs RoBERTa — Side by Side")
        fig = make_comparison_chart(df_vader, df_roberta)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Pie charts
        st.subheader("Sentiment Distribution")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚡ VADER")
            fig_vader = make_pie_chart(df_vader, 'VADER')
            st.plotly_chart(fig_vader, use_container_width=True)

        with col2:
            st.markdown("### 🧠 RoBERTa")
            fig_roberta = make_pie_chart(df_roberta, 'RoBERTa')
            st.plotly_chart(fig_roberta, use_container_width=True)

    except FileNotFoundError:
        st.error("Results not found!")


#-------- About -------
elif page == "ℹ About":
    st.title("👋 Hi, I'm Suman")
    st.caption("AI Developer | NLP Enthusiast | Builder")
    st.divider()

    st.markdown("""
I'm **Suman Thangadurai**, passionate about AI, Data Science and building real-world apps.

**How This App Works:**
- VADER → rule-based, fast
- RoBERTa → deep learning, understands context
""")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("🔗 GitHub", "https://github.com/Suman-2411")
    with col2:
        st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/suman-thangadurai-152861247")

    st.markdown("📧 suman.thangadurai@edu.dsti.institute")


#--------Model Metrics-----------
elif page == "📈 Model Metrics":
    st.title("📈 Model Metrics")
    st.divider()

    try:
        df_vader = pd.read_csv('vader_results.csv')
        df_roberta = pd.read_csv('roberta_result.csv')

        # Calculate metrics
        vader_metrics = calculate_metrics(df_vader, 'VADER')
        roberta_metrics = calculate_metrics(df_roberta, 'RoBERTa')

        # Show metrics side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚡ VADER")
            st.metric("Accuracy", f"{vader_metrics['accuracy']:.2%}")
            st.metric("F1 Score", f"{vader_metrics['f1']:.2%}")
            st.metric("Precision", f"{vader_metrics['precision']:.2%}")
            st.metric("Recall", f"{vader_metrics['recall']:.2%}")

        with col2:
            st.markdown("### 🧠 RoBERTa")
            st.metric("Accuracy", f"{roberta_metrics['accuracy']:.2%}")
            st.metric("F1 Score", f"{roberta_metrics['f1']:.2%}")
            st.metric("Precision", f"{roberta_metrics['precision']:.2%}")
            st.metric("Recall", f"{roberta_metrics['recall']:.2%}")

    except FileNotFoundError:
        st.error("Results not found!")