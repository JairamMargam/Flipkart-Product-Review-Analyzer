import streamlit as st
from review_analysis1 import run_full_analysis

st.set_page_config(page_title="Flipkart Review Analyzer", layout="wide")
st.title("📦 Flipkart Product Review Analyzer")

st.markdown("""
Welcome! Paste a Flipkart **product review page URL**, and this app will:
- 🔎 Scrape reviews
- 🧹 Clean and preprocess text
- 😊 Analyze sentiment
- 🧠 Extract frequent phrases & topics
- 🤖 Summarize insights using Gemini
""")

# --- Input Section ---
with st.sidebar:
    st.header("🔧 Configuration")
    product_url = st.text_input("🔗 Enter Flipkart Product Review URL")
    num_pages = st.slider("📄 Number of Pages to Scrape", 1, 50, 3)
    grammar_correction = st.toggle("📝 Apply Grammar Correction", value=False)

# --- Main Action ---
if st.button("🚀 Run Analysis"):
    if not product_url or "flipkart.com" not in product_url:
        st.error("❗ Please enter a valid Flipkart product review URL.")
    else:
        with st.spinner("⏳ Scraping and analyzing... Please wait..."):
            try:
                result = run_full_analysis(product_url, num_pages=num_pages, use_grammar=grammar_correction)
                st.success("✅ Analysis Complete!")

                # --- Reviews Table with toggle and download ---
                with st.expander(f"📝 Extracted Reviews ({len(result['raw'])} total)", expanded=True):
                    show_all = st.checkbox("🔍 Show full review table", value=True)

                    if show_all:
                        st.dataframe(result['raw'], use_container_width=True)
                    else:
                        st.dataframe(result['raw'].head(10), use_container_width=True)

                    csv_data = result['raw'].to_csv(index=False).encode()
                    st.download_button(
                        label="📥 Download All Reviews as CSV",
                        data=csv_data,
                        file_name="flipkart_reviews.csv",
                        mime="text/csv"
                    )

                # --- Sentiment Counts ---
                with st.expander("📊 Sentiment Summary"):
                    st.write(result['sentiment_counts'])

                # --- WordClouds (future)
                # with st.expander("☁️ WordClouds"):
                #     st.image(result['wordcloud_title'])
                #     st.image(result['wordcloud_desc'])

                # --- LDA Topics ---
                with st.expander("🧠 LDA Topics"):
                    st.text(result['lda_topics'])

                # --- Bigrams / Trigrams ---
                with st.expander("🔍 Frequent Phrases (N-Grams)"):
                    st.text(result['bigrams_text'])

                # --- Gemini Summary ---
                with st.expander("💬 Gemini Summary for Business Users", expanded=True):
                    st.info(result['summary'])

            except Exception as e:
                st.error(f"❌ An error occurred:\n```\n{e}\n```")
