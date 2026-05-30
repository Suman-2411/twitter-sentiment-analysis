import streamlit as st
from vader_analysis import analyze_vader
from roberta_analysis import analyze_roberta


# Sidebar Navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["ℹ️ About","🔍 Live Analysis", "📊 Model Comparison"]
)


# ============ CSS STYLING ============
st.markdown("""
<style>
/* Main Background */
.stApp {
    background-color: #0D0D0D;
}

/* Title */
h1 {
    color: #FF0000 !important;
    text-align: center;
    border-bottom: 2px solid #FF0000;
    padding-bottom: 10px;
}

/* Subheaders */
h2, h3 {
    color: #FFFFFF !important;
}

/* Button */
.stButton>button {
    background-color: #FF0000;
    color: white !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-weight: bold !important;
    border: none !important;
}

/* Text Input */
.stTextInput>div>input {
    background-color: #1A1A1A !important;
    color: white !important;
    border: 2px solid #FF0000 !important;
    border-radius: 8px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1A1A1A;
    border-right: 2px solid #FF0000;
}

/* Text */
p {
    color: #CCCCCC !important;
}
</style>
""", unsafe_allow_html=True)

# ============ PAGE 1 — LIVE ANALYSIS ============
if page == "🔍 Live Analysis":
    st.markdown("<h1>🔍 Live Tweet Analysis</h1>",
                unsafe_allow_html=True)
    st.divider()

    tweet = st.text_input("", 
                          placeholder="Enter your tweet here...")

    if st.button("🔍 Analyze Tweet"):
        if tweet == "":
            st.warning("⚠️ Please enter a tweet!")
        else:
            with st.spinner("🤖 Analyzing your tweet..."):
                vader_result, compound_score = analyze_vader(tweet)
                roberta_result = analyze_roberta(tweet)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 VADER Result")
                if vader_result == 'positive':
                    st.success("😊 Positive Tweet!")
                elif vader_result == 'negative':
                    st.error("😞 Negative Tweet!")
                else:
                    st.warning("😐 Neutral Tweet!")
                st.metric("Compound Score", 
                         f"{compound_score:.2f}")

            with col2:
                st.markdown("### 🤖 RoBERTa Result")
                if roberta_result == 'positive':
                    st.success("😊 Positive Tweet!")
                elif roberta_result == 'negative':
                    st.error("😞 Negative Tweet!")
                else:
                    st.warning("😐 Neutral Tweet!")

            st.divider()
            if vader_result == roberta_result:
                st.success("✅ Both models agree!")
            else:
                st.info("💡 Models disagree — RoBERTa understands context better!")

# ============ PAGE 2 — COMPARISON ============
elif page == "📊 Model Comparison":
    st.markdown("<h1>📊 Model Comparison</h1>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📈 VADER vs RoBERTa")
    st.image("model_comparison.png", 
             use_column_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 VADER Distribution")
        st.image("VADER_distribution.png",
                use_column_width=True)
    with col2:
        st.markdown("### 🤖 RoBERTa Distribution")
        st.image("RoBERTa_distribution.png",
                use_column_width=True)

# ============ PAGE 3 — ABOUT ============
elif page == "ℹ️ About":
    st.markdown("""
    <h1 style='text-align: center; color: #FF0000;'>
    👋 Hi, I'm Suman
    </h1>
    """, unsafe_allow_html=True)

    st.caption("AI Developer | NLP Enthusiast | Builder")
    st.divider()

    # ✅ About Me
    st.markdown("""
### 🚀 About Me

I'm **Suman Thangadurai**, passionate about:

- 🤖 Artificial Intelligence  
- 📊 Data Science  
- 🌐 Building real-world applications  
    """)

    st.divider()

    # ✅ Technical part
    st.markdown("""
### ⚙️ How This App Works

- 📊 VADER → rule-based, fast  
- 🤖 RoBERTa → deep learning, contextual  

👉 This app compares both models.
    """)

    st.divider()

    # ✅ ✅ ADD YOUR LINKS HERE
    st.markdown("### 📫 Connect With Me")

    col1, col2 = st.columns(2)

    with col1:
        st.link_button("🔗 GitHub", "https://github.com/Suman-2411")

    with col2:
        st.link_button(
            "💼 LinkedIn",
            "https://www.linkedin.com/in/suman-thangadurai-152861247"
        )

    st.markdown("📧 Email: suman.thangadurai@edu.dsti.institute")

    st.divider()
    st.info("👉 Try 'Live Analysis'")