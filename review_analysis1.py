# review_analysis.py

import pandas as pd
import numpy as np
import re
import spacy
import nltk
import time
import emoji
from wordcloud import WordCloud
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Setup
nltk.download('punkt')
nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

# Gemini config
genai.configure(api_key="AIzaSyAqJbm53OsGX3zy7o_q1dX564Kbd0NMT1c")
model = GenerativeModel("gemini-2.5-flash")

# Load sentiment model once
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def scrape_flipkart_reviews(base_url, num_pages=2):
    chrome_driver_path = r"C:\Users\Jai Ram\Downloads\chromedriver-win64 (1)\chromedriver-win64\chromedriver.exe"
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

    all_reviews = []
    for page in range(1, num_pages + 1):
        driver.get(base_url + f"&page={page}")
        time.sleep(3)

        # Expand all "READ MORE"
        try:
            read_more_buttons = driver.find_elements("xpath", "//span[text()='READ MORE']")
            for btn in read_more_buttons:
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.5)
        except:
            pass

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews = soup.find_all('div', class_='ZmyHeo')


        for review in reviews:
            rating_tag = review.find_previous('div', class_='XQDdHH Ga3i8K')
            rating = rating_tag.get_text(strip=True) if rating_tag else None
            title_tag = review.find_previous('p', class_='z9E0IG')
            title = title_tag.get_text(strip=True) if title_tag else None
            description = review.get_text(" ", strip=True).replace("READ MORE", "")
            name_tag = review.find_next('p', class_='_2NsDsF AwS1CA')
            name = name_tag.get_text(strip=True) if name_tag else None
            location_tag = review.find_next('p', class_='MztJPv')
            location = location_tag.find_all('span')[1].get_text(strip=True) if location_tag else None
            date_tag_candidates = review.find_all_next('p', class_='_2NsDsF')
            date = next((dt.get_text(strip=True) for dt in date_tag_candidates if dt.get_text(strip=True) != name), None)

            all_reviews.append({
                'name': name,
                'rating': rating,
                'title': title,
                'description': description,
                'date': date,
                'location': location
            })

    driver.quit()
    df = pd.DataFrame(all_reviews)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

# Function to remove emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if token.is_alpha and (not token.is_stop or token.text.lower() in ['not', 'no', 'never', 'only'])])

def correct_grammar(text):
    prompt = f"Correct the grammar in this sentence and return only the corrected sentence:\n\"{text}\""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return text

def get_sentiment(text):
    result = sentiment_pipeline(str(text))[0]
    return result['label'], result['score']

def run_lda(text_series, n_topics=5, label=""):
    tokenized = text_series.apply(lambda x: x.split())
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=42, passes=10)
    topics = f"\nLDA Topics for {label} Reviews:\n"
    for idx, topic in lda.print_topics(num_words=10):
        topics += f"Topic {idx + 1}: {topic}\n"
    return topics

def get_top_ngrams(corpus, ngram_range=(2, 3), n=20):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return sorted_words[:n]

def explain_insights(bigrams, lda_topics):
    prompt = f"""
I extracted review insights using these two:
- Top Bigrams/Trigrams:
{bigrams}

- LDA Topics:
{lda_topics}

Now give me a *very short and clear summary* in bullet points:
-  What customers like
-  What customers dislike
-  What the business should improve

Keep it simple, brief, and to the point — suitable for busy users or product managers. No fluff, just insights.
"""
    response = model.generate_content(prompt)
    return response.text.strip()



def run_full_analysis(url, num_pages=3, use_grammar=False):

    df = scrape_flipkart_reviews(url, num_pages=num_pages)
    df['description'] = df['description'].astype(str).str.strip()

    # Optional Grammar Correction
    # Grammar Correction
    if use_grammar:
        df['description_corrected'] = df['description'].apply(correct_grammar)
    else:
        df['description_corrected'] = df['description']

    # Emoji Removal
    df['description_no_emoji'] = df['description_corrected'].apply(remove_emojis)

    # Sentiment Analysis on grammar-corrected (and emoji-free) sentence
    df['description_for_sentiment'] = df['description_corrected'].apply(remove_emojis)
    sentiment_result = df['description_for_sentiment'].apply(get_sentiment)
    df['sentiment'] = sentiment_result.apply(lambda x: x[0])
    df['confidence'] = sentiment_result.apply(lambda x: x[1])

    # Cleaned for NLP
    df['description_cleaned'] = df['description_corrected'].apply(clean_text)

    # LDA Topics
    lda_pos = run_lda(df[df['sentiment'] == 'POSITIVE']['description_cleaned'], label="Positive")
    lda_neg = run_lda(df[df['sentiment'] == 'NEGATIVE']['description_cleaned'], label="Negative")
    lda_summary = lda_pos + "\n" + lda_neg

    # Bigrams/Trigrams
    bigrams = get_top_ngrams(df['description_cleaned'])
    bigram_str = '\n'.join([f"{phrase} ({count})" for phrase, count in bigrams])

    # Gemini Summary
    summary = explain_insights(bigram_str, lda_summary)

    # Sentiment Summary
    sentiment_counts = df['sentiment'].value_counts()

    return {
        'summary': summary,
        'sentiment_counts': sentiment_counts,
        'raw': df,
        'lda_topics': lda_summary,
        'bigrams_text': bigram_str
    }

